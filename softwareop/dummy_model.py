import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define the ℓ2-norm calculation for filters
def calculate_l2_norm(filters):
    # Calculate ℓ2-norm for the filters across all dimensions
    return torch.sqrt(torch.sum(filters ** 2, dim=(1, 2, 3)))

# Pruning function
def dynamic_filter_pruning(model, train_loader, criterion, optimizer, pruning_rate, emax, device):
    model.train()
    for epoch in range(emax):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Prune filters after each epoch
        for layer in model.children():
            if isinstance(layer, nn.Conv2d):  # COP type check
                filters = layer.weight.data
                n_filters = filters.size(0)  # Number of filters
                
                l2_norms = calculate_l2_norm(filters)
                
                if hasattr(layer, 'fused') and layer.fused:  # Check if operation is fused
                    m = getattr(layer, 'original_filters', n_filters)
                    # Zero out the lowest `n - m` filters
                    num_filters_to_prune = n_filters - m
                else:
                    # Zero out the lowest `p_ϕ * n` filters
                    num_filters_to_prune = int(pruning_rate * n_filters)
                
                # Identify indices of filters to prune
                _, prune_indices = torch.topk(l2_norms, num_filters_to_prune, largest=False)
                filters[prune_indices] = 0  # Zero out the filters
        
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

# Instantiate model, criterion, optimizer, and data loader
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Assume train_loader is defined elsewhere, representing the training dataset
train_dataset=torch.rand(1,10,10,10)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Perform dynamic filter pruning
pruned_model = dynamic_filter_pruning(model, train_loader, criterion, optimizer, pruning_rate=0.2, emax=10, device='cuda')
