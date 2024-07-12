from abc import ABC, abstractmethod
from finaltransformer import TransforMAP
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 
d_model = 40
num_heads = 8
drop_prob = 0.1
batch_size = 1
ffn_hidden = 1024
num_layers = 6
save_path='trained_model.pth'
class MLPrefetchModel(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''
    @abstractmethod
    def __init__(self):
        self.page_size = 40
        self.block_size = 8
        self.model = TransforMAP(
            d_model=d_model, num_heads=num_heads, num_layers=num_layers,
            ffn_hidden=ffn_hidden, drop_prob=drop_prob,block_size=self.block_size
        )

        self.mask = None

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def _create_mask(self, max_length):
        mask = torch.full([max_length, max_length], float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        return mask


    @abstractmethod

    def preprocessor(self, data):
        input_features = []
        labelslist = []
        bitmaps = {}
        max_seqeunce_length=len(data)
        for line in data:
            instr_id, cycle_count, load_address, instr_ptr, llc_hit_miss = line
            page = bin(load_address)[2:2+self.page_size]
            block = bin(load_address)[2+self.page_size:2+self.page_size+self.block_size]

            if page not in bitmaps:
                bitmaps[page] = np.zeros(2 ** self.block_size, dtype=int)
            
            input_features.append((instr_id, page, block))
            bitmaps[page][int(block, 2)] = 1 

        labels_tensor = torch.zeros((1, max_seqeunce_length, 2 ** self.block_size), dtype=torch.float32)
        for i, line in enumerate(data):
            _, _, load_address, _, _ = line
            page = bin(load_address)[2:2 + self.page_size]
            block = bin(load_address)[2 + self.page_size:2 + self.page_size + self.block_size]

            label_idx = int(block, 2)
            labels_tensor[0, i, label_idx] = 1.0  # Set the corresponding block bit to 1


        return input_features, labelslist, bitmaps,max_seqeunce_length,labels_tensor
                
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
    #I will have to send bitmap(now i understand why bitmap is necessary, it will stop the model from 
        # constantly predicting the same block for the same page) , the split address and the instruction ID.
        #or I can do that splitting within this script, and later concatenate the answers.
        #but that splitting is necessary for both.
        #train model here.
        #Now we want the input addresses to be transformed in such a way that they predict the next block. So split the block.
        # So now the target labels for training are ready
        input_features, labelslist, _, max_sequence_length,labels = self.preprocessor(data)
        tokenized_inputs = [[int(digit) for digit in str(page)] for _, page, _ in input_features]
        X = torch.tensor(tokenized_inputs, dtype=torch.float)
        print("Shape of X before adding batch dimension:", X.shape)
        print("Shape of y:", labels.shape)
        X=X.unsqueeze(0)
        print("Shape of X after adding batch dimension:", X.shape)
        print("Shape of y:", labels.shape)
        if X.size(0) != labels.size(0):
            raise ValueError(f"Batch size mismatch: X has {X.size(0)}, y has {labels.size(0)}")
        dataset = TensorDataset(X, labels)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9)
        scheduler = CustomLearningRateScheduler(optimizer, d_model=d_model, warmup_steps=2000)

        if self.mask is None:
            self.mask=self._create_mask(max_sequence_length)
        self.model.train()
        for epoch in range(10):
            for batch in dataloader:
                inputs, targets = batch
                optimizer.zero_grad()
                targets_flat = targets.view(-1,targets.shape[-1])


                print("Size of targets_flat is",targets_flat.size())
                outputs = self.model(inputs, inputs,self.mask)  # Dummy target for training
                outputs_flat = outputs.view(-1, outputs.shape[-1])
                if torch.isnan(inputs).any():
                    print("NaN detected in inputs")
                if torch.isnan(outputs).any():
                    print("NaN detected in outputs")
                if torch.isnan(targets).any():
                    print("NaN detected in targets")
                print("Size of outputs_flat is",outputs_flat.size())
                loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1,targets.shape[-1]))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        self.save(save_path)
        print(f'Model saved to {save_path}')

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
        print(data)
        input_features,_,_,max_sequence_length,_=self.preprocessor(data)
        prefetches=[]
        if self.mask==None:
            self.mask=self._create_mask(self,max_sequence_length)
        self.model.eval()
        
        tokenized_inputs = [[int(digit) for digit in str(page)] for _, page, _ in input_features]
        X = torch.tensor(tokenized_inputs, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(X, X, self.mask)
            probs = F.softmax(outputs, dim=-1)
            for idx, (instruction_id, page, _) in enumerate(input_features):
                top2_blocks = torch.topk(probs[0, idx], 2).indices
                for block_idx in top2_blocks:
                    block_str = format(block_idx.item(), f'0{self.block_size}b')
                    prefetch_address = int(page + block_str, 2)
                    prefetches.append((instruction_id, prefetch_address))

        return prefetches

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
