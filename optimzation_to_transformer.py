import torch
import torch.nn as nn

def dynamic_filter_pruning_transformer(model, X, pruning_rate, max_epoch):
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(max_epoch):
        model.train()
        
        for data, target in X:
            # Forward pass
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
        
        # Iterate through the layers of the transformer model
        for name, module in model.named_modules():
            # Prune the FFN layers (fully connected layers in transformer blocks)
            if isinstance(module, nn.Linear):
                weights = module.weight.data.clone()
                n = weights.size(0)  # number of neurons
                
                # Calculate the ℓ2-norm for the neurons (filters)
                l2_norms = torch.norm(weights.view(n, -1), p=2, dim=1)
                
                # Determine how many neurons to prune
                num_neurons_to_prune = int(pruning_rate * n)
                
                # Zero out the neurons with the lowest l2-norms
                _, indices = torch.topk(l2_norms, num_neurons_to_prune, largest=False)
                for i in indices:
                    module.weight.data[i].zero_()
            
            # Prune the attention heads in the Multi-Head Self-Attention layers
            elif isinstance(module, nn.MultiheadAttention):
                # Reshape the weights for attention heads
                weights = module.in_proj_weight.data.clone()
                num_heads = module.num_heads
                head_dim = module.head_dim
                l2_norms = torch.norm(weights.view(num_heads, -1), p=2, dim=1)
                
                # Determine how many heads to prune
                num_heads_to_prune = int(pruning_rate * num_heads)
                
                # Zero out the heads with the lowest l2-norms
                _, indices = torch.topk(l2_norms, num_heads_to_prune, largest=False)
                for i in indices:
                    module.in_proj_weight.data[i*head_dim:(i+1)*head_dim].zero_()
                    module.out_proj.weight.data[:, i*head_dim:(i+1)*head_dim].zero_()
    
    pruned_params = [param for param in model.parameters() if param.requires_grad]
    
    return pruned_params
