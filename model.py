from abc import ABC, abstractmethod
from finaltransformer import TransforMAP
import torch
import torch.nn as nn
import numpy as np 
d_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden = 1024
num_layers = 5
path='finaltransformer.py'
class MLPrefetchModel(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''
    @abstractmethod
    def __init__(self):
        self.model=TransforMAP
        self.page_size=16
        self.block_size=32
        pass

    def load(self, path):
        self.model = torch.load_state_dict(torch.load(path))


    @abstractmethod
    def save(self, path):
        '''
        Saves your model to the filepath path
        
        '''
    
        torch.save(self.model.state_dict(), path)
        pass
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
                
    @abstractmethod

    def train(self, data):
    #I will have to send bitmap(now i understand why bitmap is necessary, it will stop the model from 
        # constantly predicting the same block for the same page) , the split address and the instruction ID.
        #or I can do that splitting within this script, and later concatenate the answers.
        #but that splitting is necessary for both.

        #train model here.
        #Now we want the input addresses to be transformed in such a way that they predict the next block. So split the block.


        # So now the target labels for training are ready


    # Tokenize and pad sequences
        def pad_sequence(seq, max_length, pad_value=0):
            return seq + [pad_value] * (max_length - len(seq))
        input_features, labels, _=self.preprocessor(data)
        max_length_input = max(len(str(page)) for _, page, _ in input_features)
        tokenized_inputs = [pad_sequence([int(digit) for digit in str(page)], max_length_input) for _, page, _ in input_features]

        tokenized_labels = [[int(digit) for digit in label] for label in labels]
        print("Process ran until here 4")
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
                

    '''
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
    '''

    pass

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


'''
# Example PyTorch Model
import torch
import torch.nn as nn

class PytorchMLModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Initialize your neural network here
        # For example
        self.embedding = nn.Embedding(...)
        self.fc = nn.Linear(...)

    def forward(self, x):
        # Forward pass for your model here
        # For example
        return self.relu(self.fc(self.embedding(x)))

class TerribleMLModel(MLPrefetchModel):
    """
    This class effectively functions as a wrapper around the above custom
    pytorch nn.Module. You can approach this in another way so long as the the
    load/save/train/generate functions behave as described above.

    Disclaimer: It's terrible since the below criterion assumes a gold Y label
    for the prefetches, which we don't really have. In any case, the below
    structure more or less shows how one would use a ML framework with this
    script. Happy coding / researching! :)
    """

    def __init__(self):
        self.model = PytorchMLModel()
    
    def load(self, path):
        self.model = torch.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self, data):
        # Just standard run-time here
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = nn.optim.Adam(self.model.parameters())
        scheduler = nn.optim.lr_scheduler.StepLR(optimizer, step_size=0.1)
        for epoch in range(20):
            # Assuming batch(...) is a generator over the data
            for i, (x, y) in enumerate(batch(data)):
                y_pred = self.model(x)
                loss = criterion(y_pred, y)

                if i % 100 == 0:
                    print('Loss:', loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

    def generate(self, data):
        self.model.eval()
        prefetches = []
        for i, (x, _) in enumerate(batch(data, random=False)):
            y_pred = self.model(x)
            
            for xi, yi in zip(x, y_pred):
                # Where instr_id is a function that extracts the unique instr_id
                prefetches.append((instr_id(xi), yi))

        return prefetches
'''

# Replace this if you create your own model
Model=MLPrefetchModel
#I suppose I will have to write a 'Prefetcher' class that is callable consifering the given data
#Inputs will come as and when, and prediction will have to work in real time.
