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

# Constants
TOTAL_BITS = 64
BLOCK_BITS = 6
PAGE_BITS = 36
BLOCK_NUM_BITS = TOTAL_BITS - BLOCK_BITS
SPLIT_BITS = 6
LOOK_BACK = 5
PRED_FORWARD = 2
EXTRA = 6
BITMAP_SIZE = 2 ** (PAGE_BITS - BLOCK_BITS)
d_model = PAGE_BITS  # d_model is the same as page size, intuitively
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
        binary_address = bin(address)[2:]
        address_size = len(binary_address)

        if address_size == 36:
            new_address = '000000000000' + binary_address
        elif address_size == 42:
            new_address = '000000' + binary_address
        elif address_size == 46:
            new_address = '00' + binary_address
        elif address_size == 48:
            new_address = binary_address
        elif address_size == 47:
            new_address = '0' + binary_address
        elif address_size == 44:
            new_address = '0000' + binary_address
        else:
            raise ValueError(f"Unexpected address size: {address_size}")
        return new_address, address_size

  # Return the input features tensor and the label tensor
    def preprocessor(self, data, batch_size):
        num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)

        for i in range(num_batches):
            batch_data = data[i * batch_size:(i + 1) * batch_size]
            input_features = []  # Model training input
            page_bitmaps = defaultdict(lambda: np.zeros(2 ** self.block_size, dtype=int))  # Store bitmaps per page

            for line in batch_data:
                instr_id, cycle_count, load_address, instr_ptr, llc_hit_miss = line

                new_address, address_size = self.ensure_48bit_address(load_address)
                page = new_address[:self.page_size]  # Extract page part of the address
                block = new_address[self.page_size:self.page_size + self.block_size]  # Extract block part of the address
                block_offset = new_address[self.page_size + self.block_size:]  # Extract block offset

            # Update the bitmap for the current page (each page has its own bitmap)
                page_bitmaps[page][int(block, 2)] = 1  # Set the corresponding block bit to 1 in the page bitmap

            # Add the input features including block offset
                input_features.append((instr_id, page, block, block_offset, address_size))

        # Create the input tensor using only the binary page address
            tokenized_inputs = [[int(digit) for digit in page] for _, page, _, _, _ in input_features]
            X = torch.tensor(tokenized_inputs, dtype=torch.float).to(device)

            max_sequence_length = len(batch_data)
            labels_tensor = torch.zeros((max_sequence_length,), dtype=torch.long).to(device)
            for j, line in enumerate(batch_data):
                _, _, load_address, _, _ = line
                new_address, address_size = self.ensure_48bit_address(load_address)
                page = new_address[:self.page_size]  # Extract the page part of the address again
                block = new_address[self.page_size:self.page_size + self.block_size]  # Extract block part again

            # Retrieve the bitmap for the specific page and find the index of the block
                labels_tensor[j] = int(block, 2)

            yield X, labels_tensor, input_features  # Return the input features tensor, the label tensor, and input features

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
            for X, labels,_ in self.preprocessor(data, batch_size):
                optimizer.zero_grad()

                # Adjust inputs for the model
                inputs = X.unsqueeze(0)  # Add batch dimension if necessary

                if self.mask is None or self.mask.size(-1) != labels.size(0):
                    self.mask = self._create_mask(labels.size(0))

                # Forward pass
                outputs = self.model(inputs, inputs, self.mask)
                outputs_flat = outputs.view(-1, outputs.shape[-1])

                # Compute loss
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
        # Use the tensor X directly as it already contains the tokenized page addresses
            inputs = X.unsqueeze(0)  # Add batch dimension
            print(inputs.size())
        # Ensure the mask is set to the correct size
            if self.mask is None or self.mask.size(-1) != inputs.size(1):
                self.mask = self._create_mask(inputs.size(1))

            with torch.no_grad():
            # Generate predictions using beam search
                outputs = self.model.beam_search(inputs, inputs, beam_width=2, max_len=100, mask=self.mask)

        # Process the output to generate prefetch addresses
            for idx, (instruction_id, page, block, block_offset, address_size) in enumerate(input_features):
                top2_blocks = torch.topk(outputs[0, idx], 2).indices  # Get the top 2 block indices
                for block_idx in top2_blocks:
                    block_str = format(block_idx.item(), f'0{self.block_size}b')

                    prefetch_address = page + block_str + block_offset
                    prefetch_address_final = self._process_prefetch_address(prefetch_address, address_size)
                    prefetches.append((instruction_id, prefetch_address_final))

        return prefetches



    def _process_prefetch_address(self, prefetch_address, address_size):
        if address_size == 46:
            prefetch_address_sorted = prefetch_address[2:48]
        elif address_size == 44:
            prefetch_address_sorted = prefetch_address[2:46]
        elif address_size == 36:
            prefetch_address_sorted = prefetch_address[2:38]
        elif address_size == 42:
            prefetch_address_sorted = prefetch_address[2:44]
        else:
            return int(prefetch_address, 2)
        return int(prefetch_address_sorted, 2)

Model = MLPrefetchModel
