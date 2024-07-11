from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from finaltransformer import TransforMAP

class MLPrefetchModel(ABC):
    def __init__(self):
        self.model = TransforMAP(input_dim=10000, output_dim=32, d_model=512, nhead=8, num_layers=5, ffn_hidden=2048, drop_prob=0.1)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def preprocessor(self, data):
        page_size = 16
        block_size = 32
        input_features = []
        labels = []
        bitmaps = {}

        for line in data:
            instr_id, cycle_count, load_address, instr_ptr, llc_hit_miss = line
            page = load_address[:page_size]
            block = load_address[page_size:]

            if page not in bitmaps:
                bitmaps[page] = np.zeros(block_size, dtype=int)

            page_number = int(page, 2)  # Convert page to integer
            block_number = int(block, 2)  # Convert block to integer

            input_features.append((instr_id, page_number))
            bitmaps[page][block_number] = 1

            labels.append(format(block_number, '032b'))  # Convert block number to 32-bit binary string

        return input_features, labels

    def train(self, data):
        input_features, labels = self.preprocessor(data)

        # Tokenize and pad sequences
        def pad_sequence(seq, max_length, pad_value=0):
            return seq + [pad_value] * (max_length - len(seq))

        max_length_input = max(len(str(page)) for _, page in input_features)
        tokenized_inputs = [pad_sequence([int(digit) for digit in str(page)], max_length_input) for _, page in input_features]

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

    def generate(self, data):
        self.model.eval()
        input_features, _ = self.preprocessor(data)

        tokenized_inputs = [[int(digit) for digit in str(page)] for _, page in input_features]
        max_length_input = max(len(seq) for seq in tokenized_inputs)
        padded_inputs = [pad_sequence(seq, max_length_input) for seq in tokenized_inputs]

        X = torch.tensor(padded_inputs, dtype=torch.long)
        
        predictions = []
        with torch.no_grad():
            for (instr_id, _), x in zip(input_features, X):
                pred = self.model(x.unsqueeze(0), x.unsqueeze(0))  # Dummy target
                pred_block = torch.argmax(pred.squeeze(), dim=-1).item()
                prefetch_address = format(pred_block, '032b')
                predictions.append((instr_id, prefetch_address))
        
        return predictions

# Example usage
# Replace this if you create your own model
Model = MLPrefetchModel

