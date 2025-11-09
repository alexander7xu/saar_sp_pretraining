import torch
import torch.nn as nn
import torch.nn.functional as F

class BERTConfigTemplate:
    d_model = None
    d_ffn = None
    n_heads = None
    n_layer = None
    dropout = None
    vocab_size = None


class BERTBaseConfig(BERTConfigTemplate):
    d_model = 768
    d_ffn = 3072
    n_heads = 12
    n_layer = 12
    dropout = 0.0
    vocab_size = 30522


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        pass