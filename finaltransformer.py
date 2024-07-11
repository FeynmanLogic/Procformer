import torch
import torch.nn as nn
import numpy
import math
import torch.nn.functional as F
import time 

#isme aisa kuch karna hai, ki finally sirf ek address hi piche jaye
#to ek data splitter aur ek concatenator likh deta hun
#assume 4 pages, having 12 blocks block size of 12
starttime=time.time()
# class Animal:
#     def __init__(self, name):
#         self.name = name
    
#     def speak(self):
#         return f"{self.name} makes a sound"

# class Dog(Animal):
#     def __init__(self, name, breed):
#         super(Dog, self).__init__(name)  # Calls the __init__ method of Animal
#         self.breed = breed
    
#     def speak(self):
#         return f"{self.name}, the {self.breed}, barks"

# dog = Dog("Rex", "Labrador")
# print(dog.speak())
#the above is how the super statement works, we want the init method from the nnModule class.
# we have also included an additional feed forward layer, so as to learn additional information. Here it will have size of 2048

#bitmap creation is necessary
def scaled_dot_product(q,k,v,mask=None):
    d_k=q.size()[-1]
    scaled=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        scaled=scaled+mask
    attention=F.softmax(scaled)
    values=torch.matmul(attention,v)
    return values, attention
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob):
        super(PositionwiseFeedForward,self).__init__()
        self.linear1=nn.Linear(d_model,hidden)
        self.linear2=nn.Linear(hidden,d_model)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=drop_prob)
    def forward(self,x):
        x=self.linear1(x)
        x=self.linear2(x)
        x=self.dropout(x)

        return x

class MultiHeadCrossAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model , 2 * d_model) # 1024
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask=None):
        batch_size, sequence_length, d_model = x.size() # 30 x 200 x 512
        print(f"x.size(): {x.size()}")
        kv = self.kv_layer(x) # 30 x 200 x 1024
        print(f"kv.size(): {kv.size()}")
        q = self.q_layer(y) # 30 x 200 x 512
        print(f"q.size(): {q.size()}")
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)  # 30 x 200 x 8 x 128
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)  # 30 x 200 x 8 x 64
        kv = kv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 128
        q = q.permute(0, 2, 1, 3) # 30 x 8 x 200 x 64
        k, v = kv.chunk(2, dim=-1) # K: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64
        values, attention = scaled_dot_product(q, k, v, mask) #  30 x 8 x 200 x 64
        print(f"values: {values.size()}, attention:{attention.size()}")
        values = values.permute(0,2,1,3).reshape(batch_size, sequence_length, d_model) #  30 x 200 x 512
        out = self.linear_layer(values)  #  30 x 200 x 512
        print(f"out after passing through linear layer: {out.size()}")
        return out  #  30 x 200 x 512

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_sequence_length):
        self.d_model=d_model
        self.max_sequence_length=max_sequence_length
    def forward(self):
        even_i=torch.arrange(0,self.d_model,2).float()
        odd_i=torch.arrange(1,self.d_model,2).float()
        #look up the formula for the positional encodings, for both even and odd i's, the denominator numerically will be same
        #when we say that the dimension of a particular word is x, that means x numbers are used to represent the word, which means there were x activation functions used to define the word
        denominator=torch.pow(10000,even_i/self.d_model)
        position=torch.arrange(self.max_sequence_length,dtype=torch.float).reshape(self.max_sequence_length,1)
        #this shows the position of each word, and a one dimensipnal matrix
        #now the following will create a two dimensional matrix which will have sine and cosine relations
        even_PE=torch.sin(position/denominator)
        odd_PE=torch.sin(position/denominator)
        stacked=torch.stack([even_PE, odd_PE ],dim=2)
        PE=torch.flatten(stacked,start_dim=1,end_dim=2)
        return PE


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, max_sequence_length, d_model = x.size()
        print(f"x.size(): {x.size()}")
        qkv = self.qkv_layer(x)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv.size(): {qkv.size()}")
        q, k, v = qkv.chunk(3, dim=-1)
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}, ")
        values, attention = scaled_dot_product(q, k, v, mask)
        print(f"values.size(): {values.size()}, attention.size:{ attention.size()} ")
        values = values.permute(0,2,1,3).reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim)
        print(f"values.size(): {values.size()}")
        out = self.linear_layer(values)
        print(f"out.size(): {out.size()}")
        return out
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))] #perform layer normalization on
        mean = inputs.mean(dim=dims, keepdim=True)
        print(f"Mean ({mean.size()})")
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        print(f"Standard Deviation  ({std.size()})")
        y = (inputs - mean) / std
        print(f"y: {y.size()}")
        out = self.gamma * y  + self.beta
        print(f"self.gamma: {self.gamma.size()}, self.beta: {self.beta.size()}")
        print(f"out: {out.size()}")
        return out
class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):
        super(EncoderLayer,self).__init__()
        self.attention=MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual_x = x
        print("------- ATTENTION 1 ------")
        
        x = self.attention(x, mask=None)
        print("input is",x)
        print("------- DROPOUT 1 ------")
        x = self.dropout1(x)
        print("input is",x)
        print("------- ADD AND LAYER NORMALIZATION 1 ------")
        x = self.norm1(x + residual_x)
        residual_x = x
        print("input is",x)
        print("------- ATTENTION 2 ------")
        x = self.ffn(x)
        print("input is",x)
        print("------- DROPOUT 2 ------")
        x = self.dropout2(x)
        print("input is",x)
        print("------- ADD AND LAYER NORMALIZATION 2 ------")
        x = self.norm2(x + residual_x)
        print("input is",x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                     for _ in range(num_layers)])

    def forward(self,x):
        x=self.layers(x)
        return x
class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):
        super(DecoderLayer,self).__init__()
        self.self_attention=MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn=PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3=LayerNormalization(parameters_shape=[d_model])
        self.dropout3=nn.Dropout(p=drop_prob)

    def forward(self, x, y, decoder_mask):
        _y = y # 30 x 200 x 512 used for residual connections
        print("MASKED SELF ATTENTION")
        y = self.self_attention(y, mask=decoder_mask) # 30 x 200 x 512
        print("DROP OUT 1")
        y = self.dropout1(y) # 30 x 200 x 512
        print("ADD + LAYER NORMALIZATION 1")
        y = self.norm1(y + _y) # 30 x 200 x 512

        _y = y # 30 x 200 x 512
        print("CROSS ATTENTION")
        y = self.encoder_decoder_attention(x, y, mask=None) #30 x 200 x 512
        print("DROP OUT 2")  #30 x 200 x 512
        y = self.dropout2(y)
        print("ADD + LAYER NORMALIZATION 2")
        y = self.norm2(y + _y)  #30 x 200 x 512

        _y = y  #30 x 200 x 512
        print("FEED FORWARD 1")
        y = self.ffn(y) #30 x 200 x 512
        print("DROP OUT 3")
        y = self.dropout3(y) #30 x 200 x 512
        print("ADD + LAYER NORMALIZATION 3")
        y = self.norm3(y + _y) #30 x 200 x 512
        return y #30 x 200 x 512


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask) #30 x 200 x 512
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) 
                                          for _ in range(num_layers)])

    def forward(self, x, y, mask):
        #x : 30 x 200 x 512 
        #y : 30 x 200 x 512
        #mask : 200 x 200
        y = self.layers(x, y, mask)
        return y #30 x 200 x 512

class TransforMAP(nn.Module):
    def __init__(self, data_tensor, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        self.linear = nn.Linear(d_model, d_model)


    def forward(self, x, y, mask):

        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output, y, mask)
        output = self.linear(decoder_output)
        answer=F.softmax(output)
        return answer

