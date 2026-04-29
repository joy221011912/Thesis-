"""
============================================================
Step 4a: Generate translations on TEST set
============================================================

Runs beam-search decoding for all three models on the held-out
test set and saves translations to JSON for metric computation.

Outputs (one per model):
    results/translations_baseline.json        (text-only baseline)
    results/translations_cvt_warmstart.json   (CVT warm-started)
    results/translations_cvt_fromscratch.json (CVT from scratch)

Each JSON entry: {"video_id", "src", "ref", "hyp"}

USAGE:
    # Full test set, all 3 models, beam=5
    python3 eval_generate.py

    # Quick smoke (100 samples, greedy)
    python3 eval_generate.py --max-samples 100 --beam-size 1

    # Only one model
    python3 eval_generate.py --only cvt_warmstart
============================================================
"""

import os
import json
import time
import argparse
import torch
from torch.utils.data import Subset, DataLoader

from dataset_multimodal import (
    VATEXMultimodalDataset, collate_fn,
    PAD_ID, BOS_ID, EOS_ID,
)
from model_baseline import Transformer
from model_cvt import CVT
import sentencepiece as spm


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Beam search (batched, top-1 returned)
# ============================================================

@torch.no_grad()
def beam_search_text(model, src, src_pad_mask, beam_size=5, max_len=64,
                     length_penalty=0.6):
    """Beam search for the text-only Transformer."""
    device = src.device
    B = src.size(0)
    enc_out, src_mask = model.encode(src, src_pad_mask)

    # Expand to beam
    enc_out = enc_out.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(B * beam_size, enc_out.size(1), enc_out.size(2))
    src_mask = src_mask.unsqueeze(1).expand(-1, beam_size, -1, -1, -1).reshape(B * beam_size, 1, 1, src_mask.size(-1))

    ys = torch.full((B * beam_size, 1), BOS_ID, dtype=torch.long, device=device)
    scores = torch.zeros(B, beam_size, device=device)
    scores[:, 1:] = -1e9  # only first beam is active initially
    finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)

    for step in range(max_len - 1):
        tgt_pad = (ys == PAD_ID)
        logits = model.decode(ys, enc_out, src_mask, tgt_pad)
        logp = torch.log_softmax(logits[:, -1], dim=-1)  # (B*beam, V)
        V = logp.size(-1)

        # Mask finished beams so they only keep EOS
        logp_masked = logp.clone()
        if finished.any():
            logp_masked[finished] = -1e9
            logp_masked[finished, EOS_ID] = 0.0

        cand = scores.view(B * beam_size, 1) + logp_masked  # (B*beam, V)
        cand = cand.view(B, beam_size * V)
        top_scores, top_idx = cand.topk(beam_size, dim=-1)  # (B, beam)

        beam_idx = top_idx // V  # (B, beam)
        tok_idx = top_idx % V

        # Reorder ys and finished
        base = torch.arange(B, device=device).unsqueeze(1) * beam_size
        flat_beam = (base + beam_idx).view(-1)
        ys = ys[flat_beam]
        finished = finished[flat_beam]
        ys = torch.cat([ys, tok_idx.view(-1, 1)], dim=1)
        finished = finished | (tok_idx.view(-1) == EOS_ID)
        scores = top_scores

        if finished.all():
            break

    # Length penalty and pick best per example
    lengths = (ys != PAD_ID).sum(dim=1).float().view(B, beam_size)
    lp = ((5.0 + lengths) / 6.0) ** length_penalty
    norm_scores = scores / lp
    best = norm_scores.argmax(dim=-1)  # (B,)
    best_flat = torch.arange(B, device=device) * beam_size + best
    return ys[best_flat]


@torch.no_grad()
def beam_search_cvt(model, src, visual, src_pad_mask, vis_pad_mask
                    beam_size=5, max_len=64, length_penalty=0.6):
    """Beam search for the CVT (multimodal)."""
    device = src.device
    B = src.size(0)
    enc_out, src_mask = model.encode(src, src_pad_mask)
    vis, vis_mask = model.encode_visual(visual, vis_pad_mask)

    enc_out = enc_out.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(B * beam_size, enc_out.size(1), enc_out.size(2))
    src_mask = src_mask.unsqueeze(1).expand(-1, beam_size, -1, -1, -1).reshape(B * beam_size, 1, 1, src_mask.size(-1))
    vis = vis.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(B * beam_size, vis.size(1), vis.size(2))
    vis_mask = vis_mask.unsqueeze(1).expand(-1, beam_size, -1, -1, -1).reshape(B * beam_size, 1, 1, vis_mask.size(-1))

    ys = torch.full((B * beam_size, 1), BOS_ID, dtype=torch.long, device=device)
    scores = torch.zeros(B, beam_size, device=device)
    scores[:, 1:] = -1e9
    finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)

    for step in range(max_len - 1):
        tgt_pad = (ys == PAD_ID)
        logits = model.decode(ys, enc_out, vis, src_mask, vis_mask, tgt_pad)
        logp = torch.log_softmax(logits[:, -1], dim=-1)
        V = logp.size(-1)

        logp_masked = logp.clone()
        if finished.any():
            logp_masked[finished] = -1e9
            logp_masked[finished, EOS_ID] = 0.0

        cand = scores.view(B * beam_size, 1) + logp_masked
        cand = cand.view(B, beam_size * V)
        top_scores, top_idx = cand.topk(beam_size, dim=-1)

        beam_idx = top_idx // V
        tok_idx = top_idx % V

        base = torch.arange(B, device=device).unsqueeze(1) * beam_size
        flat_beam = (base + beam_idx).view(-1)
        ys = ys[flat_beam]
        finished = finished[flat_beam]
        ys = torch.cat([ys, tok_idx.view(-1, 1)], dim=1)
        finished = finished | (tok_idx.view(-1) == EOS_ID)
        scores = top_scores

        if finished.all():
            break

    lengths = (ys != PAD_ID).sum(dim=1).float().view(B, beam_size)
    lp = ((5.0 + lengths) / 6.0) ** length_penalty
    norm_scores = scores / lp
    best = norm_scores.argmax(dim=-1)
    best_flat = torch.arange(B, device=device) * beam_size + best
    return ys[best_flat]


# ============================================================
# Helpers
# ============================================================

def decode_ids(ids, sp, strip_specials=True):
    out = []
    for t in ids:
        if t == EOS_ID:
            break
        if strip_specials and t in (PAD_ID, BOS_ID):
            continue
        out.append(t)
    return sp.decode(out)


def load_text_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = Transformer(
        src_vocab_size=cfg.get("src_vocab_size", 16000),
        tgt_vocab_size=cfg.get("tgt_vocab_size", 16000),
        d_model=cfg.get("d_model", 512),
        n_heads=cfg.get("n_heads", 8),
        n_enc_layers=cfg.get("n_enc_layers", 6),
        n_dec_layers=cfg.get("n_dec_layers", 6),
        d_ff=cfg.get("d_ff", 2048),
        dropout=0.1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def load_cvt_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = CVT(
        src_vocab_size=cfg.get("src_vocab_size", 16000),
        tgt_vocab_size=cfg.get("tgt_vocab_size", 16000),
        d_model=cfg.get("d_model", 512),
        n_heads=cfg.get("n_heads", 8),
        n_enc_layers=cfg.get("n_enc_layers", 6),
        n_dec_layers=cfg.get("n_dec_layers", 6),
        d_ff=cfg.get("d_ff", 2048),
        dropout=0.1,
        clip_dim=cfg.get("clip_dim", 512),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


# ============================================================
# Generation
# ============================================================

def generate(model, model_type, loader, en_sp, zh_sp, device,
             beam_size, max_len):
    assert model_type in ("text", "cvt")
    results = []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        src = batch["src"].to(device)
        tgt = batch["tgt"]
        src_pad = batch["src_pad_mask"].to(device)
        vids = batch["video_ids"]

        if model_type == "text":
            out = beam_search_text(model, src, src_pad,
                                   beam_size=beam_size, max_len=max_len)
        else:
            visual = batch["visual"].to(device)
            vis_pad = batch["visual_pad_mask"].to(device)
            out = beam_search_cvt(model, src, visual, src_pad, vis_pad,
                                  beam_size=beam_size, max_len=max_len)

        for i in range(src.size(0)):
            src_ids = [t for t in src[i].tolist() if t != PAD_ID][1:-1]
            ref_ids = [t for t in tgt[i].tolist() if t != PAD_ID][1:-1]
            hyp_ids = out[i].tolist()[1:]  # drop BOS
            results.append({
                "video_id": vids[i],
                "src": en_sp.decode(src_ids),
                "ref": zh_sp.decode(ref_ids),
                "hyp": decode_ids(hyp_ids, zh_sp),
            })

        if (bi + 1) % 20 == 0:
            done = len(results)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"    batch {bi+1}/{len(loader)}  samples={done}  {rate:.1f}/s")
    return results


# ============================================================
# Main
# ============================================================

MODELS = {
    "baseline":         ("text", "checkpoints/baseline/best_model.pt"),
    "cvt_warmstart":    ("cvt",  "checkpoints/cvt_warmstart/best_model.pt"),
    "cvt_fromscratch":  ("cvt",  "checkpoints/cvt_fromscratch/best_model.pt"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=0,
                   help="0 = full test set")
    p.add_argument("--only", default="", choices=["", *MODELS.keys()],
                   help="Only evaluate one model")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    device = pick_device()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Step 4a: Generating translations on TEST set")
    print(f"  Device: {device}  beam={args.beam_size}  max_len={args.max_len}")
    print("=" * 60)

    # Dataset (same multimodal loader works for both — text model just ignores visual)
    test_ds = VATEXMultimodalDataset("test")
    if args.max_samples > 0:
        test_ds = Subset(test_ds, range(min(args.max_samples, len(test_ds))))
    print(f"  Test samples: {len(test_ds)}")

    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    en_sp = spm.SentencePieceProcessor(model_file="tokenizers/en_tokenizer.model")
    zh_sp = spm.SentencePieceProcessor(model_file="tokenizers/zh_tokenizer.model")

    todo = [args.only] if args.only else list(MODELS.keys())

    for name in todo:
        mtype, ckpt = MODELS[name]
        if not os.path.exists(ckpt):
            print(f"\n  [skip] {name}: {ckpt} not found")
            continue

        print(f"\n  >> {name}  ({mtype}, {ckpt})")
        if mtype == "text":
            model = load_text_model(ckpt, device)
        else:
            model = load_cvt_model(ckpt, device)

        t0 = time.time()
        results = generate(model, mtype, loader, en_sp, zh_sp, device,
                           beam_size=args.beam_size, max_len=args.max_len)
        dt = time.time() - t0

        out_path = os.path.join(args.out_dir, f"translations_{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"     {len(results)} translations  time={dt/60:.1f}m  -> {out_path}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n  Generation complete.")


if __name__ == "__main__":
    main()
