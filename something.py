from abc import ABC, abstractmethod
from finaltransformer import TransforMAP  # Ensure this is correctly implemented
import torch
import torch.nn as nn
import numpy as np

# Model hyperparameters
d_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden = 1024
num_layers = int(5)  # Ensure this is an integer
print(type(num_layers))
# Model path
path = 'finaltransformer.py'

class MLPrefetchModel(ABC):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''
    def __init__(self):
        self.model = TransforMAP(d_model, num_heads, ffn_hidden, int(num_layers), max_sequence_length, drop_prob)  # Ensure num_layers is an integer
        self.page_size = 16
        self.block_size = 32

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        '''
        Saves your model to the filepath path
        '''
        torch.save(self.model.state_dict(), path)

    def preprocessor(self, data):
        input_features = []
        labels = []
        bitmaps = {}

        for line in data:
            instr_id, cycle_count, load_address, instr_ptr, llc_hit_miss = line
            page = bin(load_address)[2:2+self.page_size]
            block = bin(load_address)[2+self.page_size:2+self.page_size+self.block_size]

            if page not in bitmaps:
                bitmaps[page] = np.zeros(2 ** self.block_size, dtype=int)
            
            input_features.append((instr_id, page, block))
            bitmaps[page][int(block, 2)] = 1 
            labels.append(bitmaps[page].copy())
        
        return input_features, labels, bitmaps

    def train(self, data):
        def pad_sequence(seq, max_length, pad_value=0):
            return seq + [pad_value] * (max_length - len(seq))

        input_features, labels, _ = self.preprocessor(data)
        max_length_input = max(len(str(page)) for _, page, _ in input_features)
        tokenized_inputs = [pad_sequence([int(digit) for digit in str(page)], max_length_input) for _, page, _ in input_features]
        tokenized_labels = [[int(digit) for digit in label] for label in labels]

        X = torch.tensor(tokenized_inputs, dtype=torch.long)
        y = torch.tensor(tokenized_labels, dtype=torch.long)

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(10):
            for batch in dataloader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = self.model(inputs, inputs)  # Dummy target for training
                loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    @abstractmethod
    def generate(self, data):
        '''
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        '''
        pass

class Prefetcher(MLPrefetchModel):
    '''
    Implementation of the MLPrefetchModel for a specific use case.
    '''
    def generate(self, data):
        self.model.eval()
        prefetches = []

        input_features, _, bitmaps = self.preprocessor(data)
        max_length_input = max(len(str(page)) for _, page, _ in input_features)
        tokenized_inputs = [self.pad_sequence([int(digit) for digit in str(page)], max_length_input) for _, page, _ in input_features]

        X = torch.tensor(tokenized_inputs, dtype=torch.long)

        with torch.no_grad():
            outputs = self.model(X, X)  # Dummy target for inference
            predictions = torch.argmax(outputs, dim=-1).cpu().numpy()

        for (instr_id, page, block), prediction in zip(input_features, predictions):
            prefetch_block = bin(prediction)[2:].zfill(self.block_size)
            prefetch_address = int(page + prefetch_block, 2)
            prefetches.append((instr_id, prefetch_address))
        
        return prefetches

# Initialize model
model = Prefetcher()

# Example training data
data = [(2, 14, 44977833407552, 44977833407552, False), 
        (3, 55, 279264050811968, 279264050811968, False)]

# Train the model
model.train(data)

# Save the model
model.save('prefetcher_model.pth')

# Load the model
model.load('prefetcher_model.pth')

# Generate prefetches
prefetches = model.generate(data)
print(prefetches)
