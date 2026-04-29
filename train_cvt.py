"""
============================================================
Step 3: Train CVT (Contextual-Vision Transformer)
============================================================

Trains the multimodal EN->ZH model with visual cross-attention.

By default, initializes from baselines best_model.pt to save
training time (only visual-attention sublayers start random).

USAGE:
    # Full training (30 epochs, warm-started from Step 2)
    python3 train_cvt.py

    # Smoke test
    python3 train_cvt.py --smoke-test

    # Start from scratch (no warm-start)
    python3 train_cvt.py --from-scratch

    # Resume interrupted run
    python3 train_cvt.py --resume
============================================================
"""

import os
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Subset, DataLoader

from dataset_multimodal import VATEXMultimodalDataset, collate_fn, PAD_ID, BOS_ID, EOS_ID
from model_cvt import CVT, LabelSmoothingLoss, load_baseline_weights
import sentencepiece as spm


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FineTuneScheduler:
    """
    Short linear warmup to a low constant LR.

    For fine-tuning a warm-started model (Step 2 weights), Noam's peak LR
    (~7e-4) is too high and destabilizes the pre-trained weights. A small
    constant LR with a brief warmup is the standard fine-tuning recipe.
    """

    def __init__(self, optimizer, peak_lr=1e-4, warmup_steps=500):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup:
            lr = self.peak_lr * (self.step_num / self.warmup)
        else:
            lr = self.peak_lr
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            visual = batch["visual"].to(device)
            src_pad = batch["src_pad_mask"].to(device)
            vis_pad = batch["visual_pad_mask"].to(device)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_in_pad = (tgt_in == PAD_ID)

            logits = model(src, tgt_in, visual, src_pad, tgt_in_pad, vis_pad)
            loss = loss_fn(logits, tgt_out)
            total += loss.item()
            count += 1
    model.train()
    return total / max(count, 1)


def sample_translations(model, loader, en_sp, zh_sp, device, n=3, max_len=64):
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        src = batch["src"][:n].to(device)
        tgt = batch["tgt"][:n]
        visual = batch["visual"][:n].to(device)
        src_pad = batch["src_pad_mask"][:n].to(device)
        vis_pad = batch["visual_pad_mask"][:n].to(device)

        pred = model.greedy_decode(src, visual, src_pad, vis_pad,
                                   max_len=max_len, bos_id=BOS_ID, eos_id=EOS_ID)
        samples = []
        for i in range(src.size(0)):
            src_ids = [t for t in src[i].tolist() if t != PAD_ID][1:-1]
            tgt_ids = [t for t in tgt[i].tolist() if t != PAD_ID][1:-1]
            pred_ids = []
            for t in pred[i].tolist()[1:]:
                if t == EOS_ID:
                    break
                pred_ids.append(t)
            samples.append({
                "src":  en_sp.decode(src_ids),
                "ref":  zh_sp.decode(tgt_ids),
                "pred": zh_sp.decode(pred_ids),
            })
    model.train()
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Peak LR for fine-tuning (low because warm-started)")
    p.add_argument("--warmup", type=int, default=500,
                   help="Linear warmup steps (short for fine-tuning)")
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--ckpt-dir", default="checkpoints/cvt_warmstart")
    p.add_argument("--baseline-ckpt", default="checkpoints/baseline/best_model.pt")
    p.add_argument("--from-scratch", action="store_true",
                   help="Do not warm-start from baseline weights")
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = pick_device()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print("=" * 60)
    print("  Step 3: Training CVT (Visual + Text EN -> ZH)")
    print(f"  Device: {device}")
    print("=" * 60)

    # ── Data ──
    train_ds = VATEXMultimodalDataset("train")
    val_ds   = VATEXMultimodalDataset("val")

    if args.smoke_test:
        train_ds = Subset(train_ds, range(200))
        val_ds   = Subset(val_ds,   range(50))
        args.epochs = 2
        args.batch_size = 16
        args.warmup = 50
        print("  [SMOKE TEST] train=200, val=50, epochs=2, batch=16")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # ── Model ──
    model = CVT(
        src_vocab_size=16000, tgt_vocab_size=16000,
        d_model=args.d_model, n_heads=8,
        n_enc_layers=6, n_dec_layers=6,
        d_ff=2048, dropout=0.1, max_len=128,
    )

    # Warm-start from baseline (unless --from-scratch or --resume)
    last_ckpt_path = os.path.join(args.ckpt_dir, "last_checkpoint.pt")
    warm_started = False
    if not args.from_scratch and not args.resume:
        if os.path.exists(args.baseline_ckpt):
            print(f"\n  Warm-starting from {args.baseline_ckpt}")
            load_baseline_weights(model, args.baseline_ckpt)
            warm_started = True
        else:
            print(f"\n  [Warning] {args.baseline_ckpt} not found - starting from scratch")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params/1e6:.1f}M")

    optimizer = AdamW(model.parameters(), lr=0.0,
                      betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0)
    scheduler = FineTuneScheduler(optimizer, peak_lr=args.lr, warmup_steps=args.warmup)
    loss_fn = LabelSmoothingLoss(smoothing=args.smoothing, ignore_index=PAD_ID)

    en_sp = spm.SentencePieceProcessor(model_file="tokenizers/en_tokenizer.model")
    zh_sp = spm.SentencePieceProcessor(model_file="tokenizers/zh_tokenizer.model")

    # ── Resume ──
    log = {"train_loss": [], "val_loss": [], "epochs": []}
    best_val = float("inf")
    start_epoch = 1

    if args.resume and os.path.exists(last_ckpt_path):
        print(f"\n  [Resume] Loading {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.step_num = ckpt["scheduler_step"]
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        log = ckpt.get("log", log)
        print(f"  [Resume] From epoch {start_epoch} (best_val={best_val:.4f})")
    elif args.resume:
        print("\n  [Resume] No checkpoint — starting fresh")

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n_batches = 0.0, 0

        for step, batch in enumerate(train_loader, 1):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            visual = batch["visual"].to(device)
            src_pad = batch["src_pad_mask"].to(device)
            vis_pad = batch["visual_pad_mask"].to(device)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_in_pad = (tgt_in == PAD_ID)

            logits = model(src, tgt_in, visual, src_pad, tgt_in_pad, vis_pad)
            loss = loss_fn(logits, tgt_out)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            lr = scheduler.step()
            optimizer.step()

            running += loss.item()
            n_batches += 1

            if step % args.log_interval == 0:
                print(f"  [epoch {epoch} step {step}/{len(train_loader)}] "
                      f"loss={running/n_batches:.4f} lr={lr:.2e}")

        train_loss = running / max(n_batches, 1)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        dt = time.time() - t0

        print(f"\n  === Epoch {epoch}/{args.epochs} ===")
        print(f"    train_loss: {train_loss:.4f}")
        print(f"    val_loss:   {val_loss:.4f}")
        print(f"    time:       {dt:.1f}s")

        samples = sample_translations(model, val_loader, en_sp, zh_sp, device, n=3)
        for i, s in enumerate(samples):
            print(f"    [sample {i+1}]")
            print(f"      src:  {s['src']}")
            print(f"      ref:  {s['ref']}")
            print(f"      pred: {s['pred']}")

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)
        log["epochs"].append(epoch)

        model_config = {
            "d_model": args.d_model, "n_heads": 8,
            "n_enc_layers": 6, "n_dec_layers": 6,
            "d_ff": 2048, "dropout": 0.1,
            "src_vocab_size": 16000, "tgt_vocab_size": 16000,
            "clip_dim": 512,
        }

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": model_config,
            }, os.path.join(args.ckpt_dir, "best_model.pt"))
            print(f"    [saved best_model.pt (val_loss={val_loss:.4f})]")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_step": scheduler.step_num,
            "epoch": epoch,
            "best_val": best_val,
            "log": log,
            "config": model_config,
        }, last_ckpt_path)

        with open(os.path.join(args.ckpt_dir, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)

    print("\n  Training complete.")
    print(f"  Best val_loss: {best_val:.4f}")
    print(f"  Checkpoints in: {args.ckpt_dir}/")


if __name__ == "__main__":
    main()
