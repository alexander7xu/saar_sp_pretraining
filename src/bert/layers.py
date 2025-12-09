import math

import torch
from torch import Tensor
import torch.nn as nn
from torchtyping import TensorType as T
from typeguard import typechecked

from .functional import attention, split_heads, merge_heads


@typechecked
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, bias: bool = True):
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(
            in_features=d_model, out_features=d_ffn, bias=bias
        )
        self.fc2: nn.Linear = nn.Linear(
            in_features=d_ffn, out_features=d_model, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        """Feed-forward layer with ReLU activation

        Args:
            x: `T["batch", "d_model"]` Input.

        Returns:
            `T["batch", "d_model"]` Output.
        """
        return self.fc2(torch.relu(self.fc1(x)))


@typechecked
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, log_attention: bool = False):
        super().__init__()
        # TODO : add functionality to log attention scores for interp

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_heads = self.d_model // self.n_heads
        self.log_attention: bool = log_attention
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj: nn.Linear = nn.Linear(self.d_model, self.d_model)
        self.Wq: nn.Linear = nn.Linear(self.d_model, self.d_model)
        self.Wk: nn.Linear = nn.Linear(self.d_model, self.d_model)
        self.Wv: nn.Linear = nn.Linear(self.d_model, self.d_model)

    def forward(
        self,
        x: T["batch", "sequence", "d_model"],
        mask: T["batch", "sequence", int] | None,
    ) -> Tensor:
        """Multi-head attention

        Args:
            x: `T["batch", "sequence", "d_model"]` Input.
            mask: `T["batch", "sequence", int]` Mask of input.

        Returns:
            `T["batch", "sequence", "d_model"]` Output.
        """

        # BUG: Seems like bug? It actually works as single-head.
        """
        q = split_heads(self.Wq(x), self.n_heads)
        k = split_heads(self.Wk(x), self.n_heads)
        v = split_heads(self.Wv(x), self.n_heads)

        if mask is not None:
            mask = mask[:, None, None, :] # autobroadcast to [B, H, S, S]
            
        att, weights = attention(q, k, v, mask=mask)

        attention_out = merge_heads(att)
        """
        # T["batch", "sequence", "d_model"]
        q, k, v = map(
            lambda w: w(x.reshape(-1, self.d_model)), (self.Wq, self.Wk, self.Wv)
        )
        if mask is not None:
            mask = mask.reshape(-1)
        att, weights = attention(q, k, v, mask=mask)
        attention_out: T["batch", "sequence", "d_model"] = att.reshape(x.shape)
        return self.out_proj(attention_out)


@typechecked
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding_table: nn.Embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: T["batch", int]) -> T["batch", "d_model"]:
        """Embedding layer for word embeddings.

        Does not have positional information.
        Add Positional embeddings yourself.

        Args:
            token_ids: `T["batch", int]` Indices in the vocabulary.

        Returns:
            `T["batch", "d_model"]` Word embeddings.
        """
        embeddings = self.embedding_table(token_ids)
        return math.sqrt(self.d_model) * embeddings


@typechecked
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, block_size: int):
        super().__init__()
        position: T[block_size, 1] = torch.arange(block_size)[:, None]
        div_terms: T[1, d_model // 2] = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )[None]
        pe = torch.zeros(block_size, d_model)
        pe[:, 0::2] = torch.sin(position * div_terms)
        pe[:, 1::2] = torch.cos(position * div_terms)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(
        self, x: T["batch", "sequence", "d_model"]
    ) -> T["batch", "sequence", "d_model"]:
        """Generate Sinosoidal Positional Embeddings

        Args:
            x: `T["batch", "sequence", "d_model"]` Input.

        Returns:
            `T["batch", "sequence", "d_model"]` Output (x + positional encoding).
        """
        return x + self.pe[:, : x.size(1), :]


@typechecked
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_heads = n_heads
        self.ln1: nn.LayerNorm = nn.LayerNorm(self.d_model)
        self.ln2: nn.LayerNorm = nn.LayerNorm(self.d_model)
        self.mha: MultiHeadAttention = MultiHeadAttention(self.d_model, self.n_heads)
        self.ffn: FFN = FFN(self.d_model, self.d_ffn)

    def forward(
        self,
        x: T["batch", "sequence", "d_model"],
        mask: T["batch", "sequence", int] | None,
    ) -> T["batch", "sequence", "d_model"]:
        """Encoder layer / block for encoder models

        Args:
            x `T["batch", "sequence", "d_model"]`: Input.
            mask: `T["batch", "sequence", int]` Mask of input.

        Returns:
            `T["batch", "sequence", int]` Output.
        """
        B, S, D = x.shape
        mha_out: T[B, S, D] = self.mha(x, mask)
        x: T[B, S, D] = self.ln1(x + mha_out)
        ffn_out: T[B * S, D] = self.ffn(x.reshape(-1, D))
        x: T[B, S, D] = self.ln2(x + ffn_out.reshape(B, S, D))
        return x
