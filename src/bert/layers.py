import math
import torch
import torch.nn as nn
import torch.nn.functional as F 

class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, bias = False):
        super().__init__()
        self.fc1 = nn.Linear(in_features=d_model, out_features=d_ffn, bias=True)
        self.fc2 = nn.Linear(in_features=d_ffn, out_features=d_model, bias=True)
        self.activation = F.relu()
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, log_attention=False):
        self.d_model = d_model
        self.n_heads = n_heads
        self.log_attention = log_attention
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj = nn.Linear(d_model, d_model)
    

    @staticmethod 
    def split_heads(x):
        # [batch sequence d_model]
        B, S, D = x.shape
        d_head = self.d_model // self.n_heads
        # [batch sequence n_heads d_head] => [batch n_heads sequence d_head]
        return x.view(B, S, self.n_heads, d_head).transpose(1, 2)

    @staticmethod 
    def merge_heads(x):
        B, N, S, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, N * Dh) 
 
    def attention(self, query, key, value, mask=None):
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_model)

        if mask is not None:
            scores.masked_fill_(mask=torch.tensor(False), float('-inf'))
        
        attention = self.softmax(scores) @ value 

        return attention
    
    def forward(self, x):
        q = self.merge_heads(x)
        k = self.merge_heads(x)
        v = self.merge_heads(x)

        attention_weights = self.attention(q, k, v)

        return self.out_proj(attention_weights)



