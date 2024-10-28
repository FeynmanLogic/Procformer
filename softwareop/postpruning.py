import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
import math
# Load the IMDB dataset and tokenizer
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size
d_model = 128
ffn_hidden = 512
num_heads = 8
drop_prob = 0.1
num_layers = 3
num_classes = 2
# Define tokenize_function, and create train_loader and test_loader as before

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

train_dataset = dataset['train'].select(range(10000)).map(tokenize_function, batched=True)
test_dataset = dataset['test'].select(range(10000)).map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define the model (as in pre_pruning.py)
class AuxiliaryLinear(nn.Module):
    # Same AuxiliaryLinear class definition here
    def __init__(self, in_features, out_features):
        super(AuxiliaryLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            if in_features == out_features:
                self.linear.weight.copy_(torch.eye(in_features))
            else:
                min_dim = min(in_features, out_features)
                self.linear.weight[:, :min_dim] = torch.eye(min_dim)
    def forward(self, x):
        return self.linear(x)

# [Define ModifiedEncoderLayer, ModifiedDecoderLayer, PositionalEncoding, ModifiedTransformerModel here]
class ModifiedEncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(ModifiedEncoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=drop_prob)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(drop_prob)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(drop_prob)
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        # Auxiliary operators
        self.aux_self_attention = AuxiliaryLinear(d_model, d_model)
        self.aux_ffn = AuxiliaryLinear(d_model, d_model)
    
    def forward(self, src, src_mask=None):
        attn_output, _ = self.self_attention(src, src, src, attn_mask=src_mask)
        attn_output_fused = attn_output + self.aux_self_attention(src)
        src = self.norm1(src + self.dropout1(attn_output_fused))
        
        ffn_output = self.ffn(src)
        ffn_output_fused = ffn_output + self.aux_ffn(src)
        src = self.norm2(src + ffn_output_fused)
        return src

# Modified Decoder Layer with Auxiliary Operators
class ModifiedDecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(ModifiedDecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=drop_prob)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(drop_prob)
        
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=drop_prob)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout2 = nn.Dropout(drop_prob)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(drop_prob)
        )
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)

        # Auxiliary operators
        self.aux_self_attention = AuxiliaryLinear(d_model, d_model)
        self.aux_cross_attention = AuxiliaryLinear(d_model, d_model)
        self.aux_ffn = AuxiliaryLinear(d_model, d_model)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        attn_output, _ = self.self_attention(tgt, tgt, tgt, attn_mask=tgt_mask)
        attn_output_fused = attn_output + self.aux_self_attention(tgt)
        tgt = self.norm1(tgt + self.dropout1(attn_output_fused))
        
        cross_attn_output, _ = self.cross_attention(tgt, memory, memory, attn_mask=memory_mask)
        cross_attn_output_fused = cross_attn_output + self.aux_cross_attention(memory)
        tgt = self.norm2(tgt + self.dropout2(cross_attn_output_fused))
        
        ffn_output = self.ffn(tgt)
        ffn_output_fused = ffn_output + self.aux_ffn(tgt)
        tgt = self.norm3(tgt + ffn_output_fused)
        return tgt

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :]
class ModifiedEncoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super(ModifiedEncoder, self).__init__()
        self.layers = nn.ModuleList([
            ModifiedEncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)  # Only 5 arguments
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return self.norm(src)


class ModifiedDecoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super(ModifiedDecoder, self).__init__()
        self.layers = nn.ModuleList([
            ModifiedDecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)  # Only 5 arguments
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)

# Modified Transformer Model
class ModifiedTransformerModel(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, vocab_size, num_classes):
        super(ModifiedTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = ModifiedEncoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        self.decoder = ModifiedDecoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        # BERT's input_ids can directly be used
        src = self.embedding(input_ids) * math.sqrt(d_model)
        src = self.pos_encoder(src)
        memory = self.encoder(src)
        
        output = self.output_layer(memory.mean(dim=1))
        return output
# Model instantiation
model = ModifiedTransformerModel(
    d_model=128,
    ffn_hidden=512,
    num_heads=8,
    drop_prob=0.1,
    num_layers=3,
    vocab_size=tokenizer.vocab_size,
    num_classes=2
)

# Load the saved pre-pruning state
model.load_state_dict(torch.load("model_pre_pruning.pth"))
print("Loaded pre-pruning model state.")

# Pruning function
def dynamic_filter_pruning_fusion(model, train_loader, pruning_rate=0.2, max_epoch=3):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(max_epoch):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids']
            labels = batch['label']
            
            optimizer.zero_grad()
            predictions = model(input_ids)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Post-Pruning Training - Epoch {epoch+1}/{max_epoch}, Loss: {epoch_loss/len(train_loader):.4f}")

        # Pruning step
        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and isinstance(module, (nn.Linear, AuxiliaryLinear)):
                    weight = module.weight if isinstance(module, nn.Linear) else module.linear.weight
                    l2_norm = weight.norm(2, dim=1)
                    threshold = torch.quantile(l2_norm, pruning_rate)
                    mask = l2_norm > threshold
                    pruned_indices = (~mask).nonzero(as_tuple=False).squeeze()

                    if pruned_indices.numel() > 0:
                        print(f"Pruning {pruned_indices.numel()} neurons in {name}")
                        weight[pruned_indices] = 0  

    print("Model pruning completed.")
    return model

# Apply pruning
pruned_model = dynamic_filter_pruning_fusion(model, train_loader)
def log_predictions_to_file(batch_inputs, batch_predictions, tokenizer, filename="predictions.txt"):
    """
    Logs the model inputs and predictions to a text file.

    Args:
        batch_inputs (torch.Tensor): The input_ids batch from the model.
        batch_predictions (torch.Tensor): The model's predicted classes for the inputs.
        tokenizer (BertTokenizer): The tokenizer to decode input_ids to text.
        filename (str): The name of the file to log the predictions.
    """
    with open(filename, "a") as file:
        for input_ids, prediction in zip(batch_inputs, batch_predictions):
            # Decode input_ids to text
            text = tokenizer.decode(input_ids, skip_special_tokens=True)
            # Get the predicted label (0 or 1)
            label = "Positive" if prediction == 1 else "Negative"
            # Write to file
            file.write(f"Input: {text}\nPrediction: {label}\n\n")


# Evaluation functions
def evaluate_and_log_predictions(model, test_loader, tokenizer, filename="predictions.txt"):
    """
    Evaluates the model on the test set and logs inputs and predictions to a file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        tokenizer (BertTokenizer): Tokenizer used to decode input_ids to text.
        filename (str): The name of the file to log the predictions.
    
    Returns:
        float: The accuracy of the model on the test set.
    """
    model.eval()
    correct = 0
    total = 0
    
    # Clear previous contents of the file
    with open(filename, "w") as file:
        file.write("Model Predictions\n\n")
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            labels = batch['label']
            # Get model predictions
            output = model(input_ids)
            _, predicted = torch.max(output, dim=1)

            # Log inputs and predictions to file
            log_predictions_to_file(input_ids, predicted, tokenizer, filename)

            # Calculate accuracy
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total if total > 0 else 0  # Avoid division by zero
    return accuracy

# Evaluate pruned model and log predictions
accuracy = evaluate_and_log_predictions(pruned_model, test_loader, tokenizer, filename="predictions_after_pruning.txt")
print(f"Pruned Model Accuracy: {accuracy:.2f}%")

# Save pruned model state
torch.save(pruned_model.state_dict(), "model_post_pruning.pth")
print("Model state saved (post-pruning).")
