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

TOTAL_BITS = 64
BLOCK_BITS = 6
PAGE_BITS = 42
BLOCK_NUM_BITS = TOTAL_BITS - BLOCK_BITS
SPLIT_BITS = 6
LOOK_BACK = 5
PRED_FORWARD = 2
BITMAP_SIZE = 2 ** (PAGE_BITS - BLOCK_BITS)
d_model = PAGE_BITS # d_model is same as page size, intuitively
num_heads = 7
drop_prob = 0.1
ffn_hidden = 1024
batch_size = 3000
num_layers = 7
save_path = 'trained_model.pth'

class MLPrefetchModel(object):
    @abstractmethod
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
        binary_address = bin(address)[2:]  # Convert hex to binary and remove '0b' prefix
        return binary_address.zfill(48)  # Pad to 48 bits

    @abstractmethod
    def preprocessor(self, data, batch_size):
        num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
        for i in range(num_batches):
            batch_data = data[i * batch_size:(i + 1) * batch_size]
            input_features = [] #model training ke liye
            bitmaps = {}

            for line in batch_data:
                instr_id, cycle_count, load_address, instr_ptr, llc_hit_miss = line
                # this is done to remove heterogeneity

            # Ensure the page is padded to PAGE_BITS length
                z=bin(load_address)
                if len(z)==48:
                    binary_address='00'+bin(load_address)[2:]
                elif len(z)==46:
                    binary_address='0000'+bin(load_address)[2:]
                elif len(z) <50:
                    binary_address=bin(load_address)[2:]
                    for i in range (0,50-len(z)):
                        binary_address='0'+binary_address
                else:
                    binary_address=bin(load_address)[2:]
                      
                page = binary_address[:  self.page_size].zfill(self.page_size)
                block = binary_address[ self.page_size: self.page_size + self.block_size]
                addr_size=len(binary_address)
                if page not in bitmaps:
                    bitmaps[page] = np.zeros(2 ** self.block_size, dtype=int)

                input_features.append((instr_id, page, block, addr_size))
                bitmaps[page][int(block, 2)] = 1

            max_sequence_length = len(batch_data)
            labels_tensor = torch.zeros((1, max_sequence_length, 2 ** self.block_size), dtype=torch.float32).to(device)
            for j, line in enumerate(batch_data):
                _, _, load_address, _, _ = line
                page = binary_address[:  self.page_size].zfill(self.page_size)
                block = binary_address[self.page_size:self.page_size + self.block_size]

                label_idx = int(block, 2)
                labels_tensor[0, j, label_idx] = 1.0  # Set the corresponding block bit to 1

            yield input_features, labels_tensor


    @abstractmethod
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
                tokenized_inputs = [[int(digit) for digit in str(page)] for _, page, _,address_size in input_features]
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

    @abstractmethod
    def generate(self, data):
        # prefetches = []
        # for input_features, labels in self.preprocessor(data, batch_size):
        #     tokenized_inputs = [[int(digit) for digit in str(page)] for _, page, _, address_size in input_features]
        #     X = torch.tensor(tokenized_inputs, dtype=torch.float).unsqueeze(0).to(device)

        #     if self.mask is None:
        #         self.mask = self._create_mask(X.size(1))

        #     with torch.no_grad():
        #         outputs = self.model(X, X, self.mask)
        #         probs = F.softmax(outputs, dim=-1)

        #     for idx, (instruction_id, page, block, address_size) in enumerate(input_features):
        #     # Get the top 2 blocks with highest probabilities
        #         top2_blocks = torch.topk(probs[0, idx], 2).indices
        #         for block_idx in top2_blocks:
        #         # Format the block index to a binary string of block_size length
        #             block_str = format(block_idx.item(), f'0{self.block_size}b')
        #             print(block_str)
        #         # Concatenate the page and block binary strings
        #             prefetch_address=page+block_str
        #             print(prefetch_address)
        #             if address_size ==46:
        #                 prefetch_address_sorted=''
        #                 for i in range(2,48):

        #                     prefetch_address_sorted=prefetch_address_sorted+prefetch_address[i]
        #                 prefetch_address_final=int(prefetch_address_sorted,2)

        #             elif address_size ==44:
        #                 prefetch_address_sorted=''
        #                 for i in range(2,46):

        #                     prefetch_address_sorted=prefetch_address_sorted+prefetch_address[i]
        #                 prefetch_address_final=int(prefetch_address_sorted,2)

        #             else:
        #                 prefetch_address_final=int(page+block_str,2)


        #             prefetches.append((instruction_id, prefetch_address_final))
        # return prefetches
        prefetches=[]
        prefetch_count=defaultdict(int)
        for line in data:
                instr_id, _,load_address, _,_  =line
            # Limit prefetches to 2 per instr_id
                if prefetch_count[instr_id] < 2:
                # Convert block from binary to integer
                        prefetch_address_final=load_address+64
                        prefetch_count[instr_id]+=1
                        prefetches.append((instr_id,prefetch_address_final))
        return prefetches
    




# Replace this if you create your own model
Model = MLPrefetchModel

# Initialize your model
model = Model()
