# ------ Testing layers
import torch
from torchtyping import patch_typeguard

from bert.layers import (
    FFN,
    MultiHeadAttention,
    EmbeddingLayer,
    SinusoidalPositionalEncoding,
    EncoderLayer,
)


patch_typeguard()


BATCH_SIZE = 16
SEQ_SIZE = 64
EMBED_SIZE = 42

# -------- FFN --------
ffn = FFN(EMBED_SIZE, 24)
in_ = torch.randn((BATCH_SIZE, EMBED_SIZE))
out = ffn(in_)
assert out.shape == in_.shape
print(f"[FNN]  IN: {in_.shape}  OUT: {out.shape}")


# -------- MultiHeadAttention --------
mha = MultiHeadAttention(EMBED_SIZE, 8)
mask = torch.ones((BATCH_SIZE, SEQ_SIZE), dtype=int)
in_ = torch.randn((BATCH_SIZE, SEQ_SIZE, EMBED_SIZE))
out = mha(in_, mask)
assert out.shape == in_.shape
assert torch.allclose(out, mha(in_, None))
print(f"[MultiHeadAttention]  IN: {in_.shape}  OUT: {out.shape}")
# TODO: Compare with PyTorch


# -------- EmbeddingLayer --------
embedding = EmbeddingLayer(256, EMBED_SIZE)
in_ = torch.randint(0, 256, (BATCH_SIZE * SEQ_SIZE,))
out = embedding(in_)
assert (*out.shape,) == (BATCH_SIZE * SEQ_SIZE, EMBED_SIZE)
print(f"[EmbeddingLayer]  IN: {in_.shape}  OUT: {out.shape}")


# -------- SinusoidalPositionalEncoding --------
spe = SinusoidalPositionalEncoding(EMBED_SIZE, SEQ_SIZE)
in_ = torch.rand((BATCH_SIZE, SEQ_SIZE, EMBED_SIZE))
out = spe(in_)
assert out.shape == in_.shape
print(f"[SinusoidalPositionalEncoding]  IN: {in_.shape}  OUT: {out.shape}")


# -------- EncoderLayer --------
spe = EncoderLayer(EMBED_SIZE, 8, 64)
mask = torch.ones((BATCH_SIZE, SEQ_SIZE), dtype=int)
in_ = torch.rand((BATCH_SIZE, SEQ_SIZE, EMBED_SIZE))
out = spe(in_, mask)
assert out.shape == in_.shape
assert torch.allclose(out, spe(in_, None))
print(f"[EncoderLayer]  IN: {in_.shape}  OUT: {out.shape}")
