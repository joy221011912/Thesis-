"""
============================================================
Step 2: Transformer Encoder-Decoder (EN -> ZH)
============================================================

Standard Transformer from "Attention is All You Need"
(Vaswani et al., 2017), built from scratch in PyTorch.

Architecture:
    - d_model=512, n_heads=8, d_ff=2048
    - 6 encoder layers + 6 decoder layers
    - Sinusoidal positional encoding
    - Shared output projection with target embedding weights

Step 3 will extend this by inserting a VisualCrossAttention
sublayer into each decoder layer (between self-attn and
encoder cross-attn).
============================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_ID = 0


# ============================================================
# Positional Encoding (sinusoidal, fixed)
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ============================================================
# Multi-Head Attention (manual implementation)
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (B, L, D)
        mask:    (B, 1, Lq, Lk) boolean — True = mask OUT
        """
        B, Lq, _ = q.shape
        Lk = k.size(1)
        Q = self.q_proj(q).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(k).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(v).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, H, Lq, d_k)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(out)


# ============================================================
# Feed-Forward
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ============================================================
# Encoder Layer
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


# ============================================================
# Decoder Layer
# ============================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


# ============================================================
# Full Transformer
# ============================================================
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size=16000,
        tgt_vocab_size=16000,
        d_model=512,
        n_heads=8,
        n_enc_layers=6,
        n_dec_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=128,
    ):
        super().__init__()
        self.d_model = d_model

        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len + 64)
        self.drop = nn.Dropout(dropout)

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec_layers)]
        )

        self.out_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        # Weight tying: share target embeddings with output projection
        self.out_proj.weight = self.tgt_emb.weight

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def make_src_mask(src_pad_mask):
        # src_pad_mask: (B, S) True = pad
        # return (B, 1, 1, S) broadcastable
        return src_pad_mask.unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_tgt_mask(tgt_pad_mask):
        # tgt_pad_mask: (B, T)
        B, T = tgt_pad_mask.shape
        pad = tgt_pad_mask.unsqueeze(1).unsqueeze(2)           # (B,1,1,T)
        causal = torch.triu(torch.ones(T, T, device=tgt_pad_mask.device), diagonal=1).bool()
        causal = causal.unsqueeze(0).unsqueeze(0)              # (1,1,T,T)
        return pad | causal                                    # (B,1,T,T)

    def encode(self, src, src_pad_mask):
        x = self.src_emb(src) * math.sqrt(self.d_model)
        x = self.drop(self.pos_enc(x))
        src_mask = self.make_src_mask(src_pad_mask)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x, src_mask

    def decode(self, tgt, enc_out, src_mask, tgt_pad_mask):
        x = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        x = self.drop(self.pos_enc(x))
        tgt_mask = self.make_tgt_mask(tgt_pad_mask)
        for layer in self.decoder:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.out_proj(x)

    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask):
        enc_out, src_mask = self.encode(src, src_pad_mask)
        return self.decode(tgt, enc_out, src_mask, tgt_pad_mask)

    @torch.no_grad()
    def greedy_decode(self, src, src_pad_mask, max_len=64, bos_id=1, eos_id=2):
        self.eval()
        B = src.size(0)
        enc_out, src_mask = self.encode(src, src_pad_mask)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            tgt_pad = (ys == PAD_ID)
            logits = self.decode(ys, enc_out, src_mask, tgt_pad)
            next_tok = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(-1) == eos_id)
            if finished.all():
                break
        return ys


# ============================================================
# Label-Smoothed Cross-Entropy
# ============================================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=PAD_ID):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        # logits: (B, T, V) — target: (B, T)
        V = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)
        nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        smooth = -logp.mean(dim=-1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        mask = (target != self.ignore_index).float()
        return (loss * mask).sum() / mask.sum().clamp(min=1)


if __name__ == "__main__":
    # Smoke test
    model = Transformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.1f}M")

    B, S, T = 2, 10, 8
    src = torch.randint(4, 16000, (B, S))
    tgt = torch.randint(4, 16000, (B, T))
    src_pad = torch.zeros(B, S, dtype=torch.bool)
    tgt_pad = torch.zeros(B, T, dtype=torch.bool)
    out = model(src, tgt, src_pad, tgt_pad)
    print(f"Output shape: {out.shape}  (expected: ({B}, {T}, 16000))")
    loss = LabelSmoothingLoss()(out, tgt)
    print(f"Loss: {loss.item():.4f}")
