import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the ℓ2-norm calculation for filters
def calculate_l2_norm(filters):
    return torch.sqrt(torch.sum(filters ** 2, dim=(1, 2, 3)))

# Pruning function
def dynamic_filter_pruning(model, train_loader, criterion, optimizer, pruning_rate, emax, device):
    model.to(device)  # Ensure model is on the correct device
    model.train()
    
    for epoch in range(emax):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Prune filters after each epoch
        for layer in model.children():
            if isinstance(layer, nn.Conv2d):
                filters = layer.weight.data
                n_filters = filters.size(0)
                
                l2_norms = calculate_l2_norm(filters)
                
                if hasattr(layer, 'fused') and layer.fused:
                    m = getattr(layer, 'original_filters', n_filters)
                    num_filters_to_prune = n_filters - m
                else:
                    num_filters_to_prune = int(pruning_rate * n_filters)
                    num_filters_to_prune = min(num_filters_to_prune, n_filters)  # Ensure valid number
                
                _, prune_indices = torch.topk(l2_norms, num_filters_to_prune, largest=False)
                filters[prune_indices] = 0
        
        print(f'Epoch {epoch + 1}/{emax}, Loss: {running_loss / len(train_loader)}')

    return model

# Example usage with a dummy model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 32 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create dummy data
data = torch.rand(100, 3, 32, 32)  # 100 samples, 3 channels, 32x32 image
labels = torch.randint(0, 10, (100,))  # 100 labels for 10 classes
train_dataset = TensorDataset(data, labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Instantiate model, criterion, optimizer, and perform dynamic filter pruning
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Perform dynamic filter pruning
device = 'cpu'
pruned_model = dynamic_filter_pruning(model, train_loader, criterion, optimizer, pruning_rate=0.2, emax=10, device=device)
