"""
============================================================
Step 3: CVT (Contextual-Vision Transformer)
============================================================

Extends the Step 2 Transformer by inserting a
VisualCrossAttention sublayer into every decoder layer.

Decoder layer sublayers:
    1) MaskedSelfAttention(tgt, tgt, tgt)           [from Step 2]
    2) VisualCrossAttention(tgt, visual, visual)    [NEW]
    3) EncoderCrossAttention(tgt, enc_out, enc_out) [from Step 2]
    4) FeedForward                                   [from Step 2]

The model can load Step 2 weights as initialization:
    - Matching params (embeddings, encoder, self/cross attn,
      FFN) are copied
    - New visual-attention sublayer is randomly initialized
============================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_baseline import (
    PAD_ID,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    EncoderLayer,
    LabelSmoothingLoss,
)


CLIP_DIM = 512


class CVTDecoderLayer(nn.Module):
    """Decoder layer with THREE attention sublayers (self + visual + cross)."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn   = MultiHeadAttention(d_model, n_heads, dropout)
        self.visual_attn = MultiHeadAttention(d_model, n_heads, dropout)  # NEW
        self.cross_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)  # NEW
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, visual, tgt_mask, src_mask, vis_mask):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm_v(x + self.drop(self.visual_attn(x, visual, visual, vis_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


class CVT(nn.Module):
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
        clip_dim=CLIP_DIM,
    ):
        super().__init__()
        self.d_model = d_model

        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len + 64)
        self.drop = nn.Dropout(dropout)

        # Project CLIP features (512) -> d_model (also 512, but keep explicit for flexibility)
        self.visual_proj = nn.Sequential(
            nn.Linear(clip_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc_layers)]
        )
        self.decoder = nn.ModuleList(
            [CVTDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec_layers)]
        )

        self.out_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.out_proj.weight = self.tgt_emb.weight  # weight tying

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Zero-init visual-attention output projections. The residual
        # connection then passes the warm-started state through untouched
        # at init; the visual branch learns to contribute from zero.
        for layer in self.decoder:
            nn.init.zeros_(layer.visual_attn.out_proj.weight)
            if layer.visual_attn.out_proj.bias is not None:
                nn.init.zeros_(layer.visual_attn.out_proj.bias)

    @staticmethod
    def make_src_mask(src_pad_mask):
        return src_pad_mask.unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_tgt_mask(tgt_pad_mask):
        B, T = tgt_pad_mask.shape
        pad = tgt_pad_mask.unsqueeze(1).unsqueeze(2)
        causal = torch.triu(
            torch.ones(T, T, device=tgt_pad_mask.device), diagonal=1
        ).bool()
        causal = causal.unsqueeze(0).unsqueeze(0)
        return pad | causal

    @staticmethod
    def make_visual_mask(visual_pad_mask):
        return visual_pad_mask.unsqueeze(1).unsqueeze(2)

    def encode(self, src, src_pad_mask):
        x = self.src_emb(src) * math.sqrt(self.d_model)
        x = self.drop(self.pos_enc(x))
        src_mask = self.make_src_mask(src_pad_mask)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x, src_mask

    def encode_visual(self, visual, visual_pad_mask):
        v = self.visual_proj(visual)
        vis_mask = self.make_visual_mask(visual_pad_mask)
        return v, vis_mask

    def decode(self, tgt, enc_out, visual, src_mask, vis_mask, tgt_pad_mask):
        x = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        x = self.drop(self.pos_enc(x))
        tgt_mask = self.make_tgt_mask(tgt_pad_mask)
        for layer in self.decoder:
            x = layer(x, enc_out, visual, tgt_mask, src_mask, vis_mask)
        return self.out_proj(x)

    def forward(self, src, tgt, visual, src_pad_mask, tgt_pad_mask, visual_pad_mask):
        enc_out, src_mask = self.encode(src, src_pad_mask)
        vis, vis_mask = self.encode_visual(visual, visual_pad_mask)
        return self.decode(tgt, enc_out, vis, src_mask, vis_mask, tgt_pad_mask)

    @torch.no_grad()
    def greedy_decode(self, src, visual, src_pad_mask, visual_pad_mask,
                      max_len=64, bos_id=1, eos_id=2):
        self.eval()
        B = src.size(0)
        enc_out, src_mask = self.encode(src, src_pad_mask)
        vis, vis_mask = self.encode_visual(visual, visual_pad_mask)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            tgt_pad = (ys == PAD_ID)
            logits = self.decode(ys, enc_out, vis, src_mask, vis_mask, tgt_pad)
            next_tok = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(-1) == eos_id)
            if finished.all():
                break
        return ys


def load_baseline_weights(cvt_model, baseline_ckpt_path, verbose=True):
    """
    Copy weights from Step 2 Transformer into CVT where names match.
    New visual-attention parameters are left at their random init.
    """
    ckpt = torch.load(baseline_ckpt_path, map_location="cpu")
    baseline_state = ckpt["model_state_dict"]
    cvt_state = cvt_model.state_dict()

    copied, skipped_new, shape_mismatch = [], [], []
    for k, v in baseline_state.items():
        if k in cvt_state and cvt_state[k].shape == v.shape:
            cvt_state[k] = v
            copied.append(k)
        elif k in cvt_state:
            shape_mismatch.append(k)

    for k in cvt_state:
        if k not in baseline_state:
            skipped_new.append(k)

    cvt_model.load_state_dict(cvt_state)

    if verbose:
        print(f"  [warm-start] Copied from baseline: {len(copied)} tensors")
        print(f"  [warm-start] New (random init):  {len(skipped_new)} tensors")
        if shape_mismatch:
            print(f"  [warm-start] Shape mismatch:    {len(shape_mismatch)} tensors")

    return cvt_model


if __name__ == "__main__":
    # Smoke test
    model = CVT()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CVT parameters: {n_params/1e6:.1f}M")

    B, S, T, N = 2, 10, 8, 10
    src = torch.randint(4, 16000, (B, S))
    tgt = torch.randint(4, 16000, (B, T))
    visual = torch.randn(B, N, 512)
    src_pad = torch.zeros(B, S, dtype=torch.bool)
    tgt_pad = torch.zeros(B, T, dtype=torch.bool)
    vis_pad = torch.zeros(B, N, dtype=torch.bool)

    out = model(src, tgt, visual, src_pad, tgt_pad, vis_pad)
    print(f"Output: {out.shape}  (expected ({B}, {T}, 16000))")

    # Warm-start test
    import os
    if os.path.exists("checkpoints/baseline/best_model.pt"):
        print("\nLoading Step 2 weights:")
        load_baseline_weights(model, "checkpoints/baseline/best_model.pt")
    else:
        print("\n(No baseline checkpoint found — skipping warm-start test)")
