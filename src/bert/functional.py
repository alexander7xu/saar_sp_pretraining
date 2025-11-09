import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 


def attention(query, key, value, scale=None, mask=None):

    if scale is None:
        scale = math.sqrt(key.size(-1))

    scores = (query @ key.transpose(-2, -1)) / scale

    if mask is not None:
        scores.masked_fill_(mask=0, value=float('-inf'))
        
    attention = F.softmax(scores, -1) @ value 

    return attention


def split_heads(x, n_heads):
    # [batch sequence d_model]
    B, S, D = x.shape
    d_head = D // n_heads
    # [batch sequence n_heads d_head] => [batch n_heads sequence d_head]
    return x.view(B, S, n_heads, d_head).transpose(1, 2)

def merge_heads(x):
    B, Nh, S, Dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, S, Nh * Dh)
 
 
 
 
 
 
 
 
 
 
 
