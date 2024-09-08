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
# There are three divisions, the page bits, the page offset and the block offset. page wale badalte hai, 
# bake dono 6 pe fixed.
# Constants
TOTAL_BITS = 64
BLOCK_BITS = 6
PAGE_BITS = 36
BLOCK_NUM_BITS = TOTAL_BITS - BLOCK_BITS
SPLIT_BITS = 6
LOOK_BACK = 5
PRED_FORWARD = 2
EXTRA=6
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
                binary_address=bin(address)[2:0]
                address_size=len(binary_address)

                if address_size ==36:
                    new_address='000000000000'+binary_address
                if address_size==42:
                    new_address='000000'+binary_address
                if address_size==46:
                    new_address='00'+binary_address
                if address_size==44:
                    new_address='0000'+binary_address
                return new_address,address_size

    def preprocessor(self, data, batch_size):
        num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    
        for i in range(num_batches):
            batch_data = data[i * batch_size:(i + 1) * batch_size]
            input_features = []  # Model training input
        
            page_bitmaps = defaultdict(lambda: np.zeros(2 ** self.block_size, dtype=int))  # Store bitmaps per page

            for line in batch_data:
                instr_id, cycle_count, load_address, instr_ptr, llc_hit_miss = line

            # Ensure the address is padded to 48 bits and split into page and block

                new_address,address_size=self.ensure_48bit_address(load_address)
                page = new_address[:self.page_size]  # Extract page part of the address
                block = new_address[self.page_size:self.page_size + self.block_size]  # Extract block part of the address
                block_offset=new_address[self.page_size+self.block_size:]
            # Update the bitmap for the current page (each page has its own bitmap)
                page_bitmaps[page][int(block, 2)] = 1  # Set the corresponding block bit to 1 in the page bitmap
            
            # Add the input features (you can expand these as needed)
                input_features.append((instr_id, page, block,block_offset,address_size))

        # Now, create the labels tensor for this batch, using the bitmaps for each page
            max_sequence_length = len(batch_data)
            labels_tensor = torch.zeros((1, max_sequence_length, 2 ** self.block_size), dtype=torch.float32).to(device)
            for j, line in enumerate(batch_data):
                _, _, load_address, _, _ = line
                new_address,address_size = self.ensure_48bit_address(load_address)
                page = new_address[:self.page_size]  # Extract the page part of the address again
                block = new_address[self.page_size:self.page_size + self.block_size]  # Extract block part again
                block_offset=new_address[self.block_size+self.page_size:]
            # Retrieve the bitmap for the specific page and apply it to the labels tensor
                labels_tensor[0, j] = torch.tensor(page_bitmaps[page], dtype=torch.float32).to(device)

            yield input_features, labels_tensor  # Return the input features and the label tensor


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
        optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9)
        scheduler = CustomLearningRateScheduler(optimizer, d_model=d_model, warmup_steps=2000)

        self.model.train()
        for epoch in range(10):
            for input_features, labels in self.preprocessor(data, batch_size):
                tokenized_inputs = [[int(digit) for digit in str(page)] for _, page, _,_, address_size in input_features]
                X = torch.tensor(tokenized_inputs, dtype=torch.float).unsqueeze(0).to(device)

                dataset = TensorDataset(X, labels)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

                if self.mask is None:
                    self.mask = self._create_mask(labels.size(1))

                for batch in dataloader:
                    inputs, targets = batch
                    optimizer.zero_grad()
                    targets_flat = targets.view(-1, targets.shape[-1])

                    outputs = self.model(inputs, inputs, self.mask)
                    outputs_flat = outputs.view(-1, outputs.shape[-1])
                    loss = criterion(outputs_flat, targets_flat)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        self.save(save_path)
        print(f'Model saved to {save_path}')

    def generate(self, data):
        prefetches = []
        for input_features, labels in self.preprocessor(data, batch_size):
            tokenized_inputs = [[int(digit) for digit in str(page)] for _, page, _, _,address_size in input_features]
            X = torch.tensor(tokenized_inputs, dtype=torch.float).unsqueeze(0).to(device)

            if self.mask is None:
                self.mask = self._create_mask(X.size(1))

            with torch.no_grad():
                outputs = self.model.beam_search(X, X, beam_width=2, max_len=10, mask=self.mask)

            for idx, (instruction_id, page, block, extra, address_size) in enumerate(input_features):
               
                top2_blocks = torch.topk(outputs[0, idx], 2).indices  # Adjust indexing for single sequence batch
                for block_idx in top2_blocks:
                    block_str = format(block_idx.item(), f'0{self.block_size}b')

                    prefetch_address = page + block_str +extra
                    prefetch_address_final = self._process_prefetch_address(prefetch_address, address_size)
                    prefetches.append((instruction_id, prefetch_address_final))

        return prefetches

    def _process_prefetch_address(self, prefetch_address, address_size):
        if address_size == 46:
            prefetch_address_sorted = prefetch_address[2:48]
        elif address_size == 44:
            prefetch_address_sorted = prefetch_address[2:46]
        elif address_size==36:
            prefetch_address=prefetch_address[2:38]
        elif address_size ==42:
            prefetch_address=prefetch_address[2:44]
        else:
            return int(prefetch_address, 2)
        return int(prefetch_address_sorted, 2)


Model = MLPrefetchModel
model = Model()
