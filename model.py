from abc import ABC, abstractmethod
from finaltransformer import TransforMAP
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import defaultdict

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# _____36 ____6____6
# Constants
TOTAL_BITS = 64
BLOCK_BITS = 6
PAGE_BITS = 36
d_model = PAGE_BITS
num_heads = 9
drop_prob = 0.1
ffn_hidden = 1024
batch_size = 3000
num_layers = 7
save_path = 'trained_model.pth'

class MLPrefetchModel(ABC):
    def __init__(self):
        self.page_size = PAGE_BITS
        self.block_size = BLOCK_BITS
        self.model = TransforMAP(
            d_model=d_model, num_heads=num_heads, num_layers=num_layers,
            ffn_hidden=ffn_hidden, drop_prob=drop_prob, block_size=self.block_size
        ).to(device)  # Move the model to the GPU

        self.mask = None

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def _create_mask(self, max_length):
        mask = torch.full([max_length, max_length], float('-inf')).to(device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def ensure_48bit_address(self, address):
        binary_address = bin(address)[2:].zfill(48)
        page = binary_address[:PAGE_BITS]
        block = binary_address[PAGE_BITS:PAGE_BITS + BLOCK_BITS]
        offset = binary_address[PAGE_BITS + BLOCK_BITS:]
        return page, block, offset

    def preprocessor(self, data, batch_size):
        num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)

        for i in range(num_batches):
            batch_data = data[i * batch_size:(i + 1) * batch_size]
            input_features = []
            page_bitmaps = defaultdict(lambda: np.zeros(2 ** self.block_size, dtype=int))

            for line in batch_data:
                instr_id, cycle_count, load_address, instr_ptr, llc_hit_miss = line
                page, block, offset = self.ensure_48bit_address(load_address)
                page_bitmaps[page][int(block, 2)] = 1
                input_features.append((instr_id, page, block, offset))

            tokenized_inputs = [[int(digit) for digit in page] for _, page, _, _ in input_features]
            X = torch.tensor(tokenized_inputs, dtype=torch.float).to(device)

            labels_tensor = torch.zeros((len(batch_data),), dtype=torch.long).to(device)
            for j, line in enumerate(batch_data):
                _, _, load_address, _, _ = line
                _, block, _ = self.ensure_48bit_address(load_address)
                labels_tensor[j] = int(block, 2)  # Unique block within page for label

            yield X, labels_tensor, input_features

    def train(self, data):
        class CustomLearningRateScheduler:
            def __init__(self, optimizer, d_model, warmup_steps=2000):
                self.optimizer = optimizer
                self.warmup_steps = warmup_steps
                self.d_model = d_model
                self.current_step = 0

            def step(self):
                self.current_step += 1
                lr = self.d_model ** -0.5 * min(self.current_step ** -0.5, self.current_step * self.warmup_steps ** -1.5)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            def zero_grad(self):
                self.optimizer.zero_grad()

            def step_optimizer(self):
                self.optimizer.step()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        scheduler = CustomLearningRateScheduler(optimizer, d_model=d_model, warmup_steps=2000)

        self.model.train()
        for epoch in range(10):
            for X, labels, _ in self.preprocessor(data, batch_size):
                optimizer.zero_grad()

                inputs = X.unsqueeze(0)

                if self.mask is None or self.mask.size(-1) != labels.size(0):
                    self.mask = self._create_mask(labels.size(0))

                outputs = self.model(inputs, inputs, self.mask)
                outputs_flat = outputs.view(-1, outputs.shape[-1])

                loss = criterion(outputs_flat, labels.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        self.save(save_path)
        print(f'Model saved to {save_path}')

    def generate(self, data):
        prefetches = []
        for X, labels, input_features in self.preprocessor(data, batch_size):
            inputs = X.unsqueeze(0)
            print(inputs.size())

            if self.mask is None or self.mask.size(-1) != inputs.size(1):
                self.mask = self._create_mask(inputs.size(1))

            with torch.no_grad():
                outputs = self.model.beam_search(inputs, inputs, beam_width=2, max_len=100, mask=self.mask)

            for idx, (instruction_id, page, block, offset) in enumerate(input_features):
                top2_blocks = torch.topk(outputs[0, idx], 2).indices
                for block_idx in top2_blocks:
                    block_str = format(block_idx.item(), f'0{self.block_size}b')
                    prefetch_address = page + block_str + offset
                    prefetch_address_final = self._process_prefetch_address(prefetch_address)
                    prefetches.append((instruction_id, prefetch_address_final))

        return prefetches

    def _process_prefetch_address(self, prefetch_address):
        return int(prefetch_address.zfill(48), 2)

# Instantiate and use the model
Model = MLPrefetchModel
