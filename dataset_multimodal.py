"""
============================================================
Step 3: Multimodal Dataset (Text + Visual Features)
============================================================

Extends dataset_text by loading per-video CLIP features
from features/train/{video_id}.npy.

Per sample returns:
    {
        'src_ids':   English token IDs (with BOS/EOS)
        'tgt_ids':   Chinese token IDs (with BOS/EOS)
        'video_id':  string
        'visual':    (N_frames, 512) float32 tensor (CLIP features)
    }

Collate pads visual features to max N_frames in the batch
and produces a visual_pad_mask.
============================================================
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import sentencepiece as spm


PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
MAX_LEN = 128
CLIP_DIM = 512


class VATEXMultimodalDataset(Dataset):
    def __init__(
        self,
        split_name,
        dataset_dir="vatex",
        splits_file="data_splits.json",
        features_dir="features/train",
        en_tokenizer_path="tokenizers/en_tokenizer.model",
        zh_tokenizer_path="tokenizers/zh_tokenizer.model",
        max_len=MAX_LEN,
    ):
        self.max_len = max_len
        self.features_dir = features_dir
        self.en_sp = spm.SentencePieceProcessor(model_file=en_tokenizer_path)
        self.zh_sp = spm.SentencePieceProcessor(model_file=zh_tokenizer_path)

        with open(splits_file) as f:
            splits = json.load(f)
        split_ids = set(splits[split_name])

        dataset = load_from_disk(os.path.join(dataset_dir, "json"))
        train_data = dataset["train"]

        # Preload feature paths and confirm existence
        self.samples = []
        for i in range(len(train_data)):
            entry = train_data[i]
            vid = entry["videoID"]
            if vid not in split_ids:
                continue
            feat_path = os.path.join(features_dir, f"{vid}.npy")
            if not os.path.exists(feat_path):
                continue
            en_caps = entry["enCap"]
            zh_caps = entry["chCap"]
            n = min(len(en_caps), len(zh_caps))
            for j in range(n):
                self.samples.append((vid, en_caps[j], zh_caps[j], feat_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, en, zh, feat_path = self.samples[idx]
        src = [BOS_ID] + self.en_sp.encode(en, out_type=int) + [EOS_ID]
        tgt = [BOS_ID] + self.zh_sp.encode(zh, out_type=int) + [EOS_ID]
        src = src[: self.max_len]
        tgt = tgt[: self.max_len]

        # CLIP features are stored as float16 — cast to float32 for the model
        visual = np.load(feat_path).astype(np.float32)  # (N_frames, 512)
        return {
            "src_ids": src,
            "tgt_ids": tgt,
            "video_id": vid,
            "visual": visual,
        }


def collate_fn(batch):
    """Pad text sequences AND visual features to max within batch."""
    B = len(batch)
    max_src = max(len(b["src_ids"]) for b in batch)
    max_tgt = max(len(b["tgt_ids"]) for b in batch)
    max_vis = max(b["visual"].shape[0] for b in batch)

    src = torch.full((B, max_src), PAD_ID, dtype=torch.long)
    tgt = torch.full((B, max_tgt), PAD_ID, dtype=torch.long)
    visual = torch.zeros((B, max_vis, CLIP_DIM), dtype=torch.float32)
    visual_pad_mask = torch.ones((B, max_vis), dtype=torch.bool)  # True = pad

    for i, b in enumerate(batch):
        src[i, : len(b["src_ids"])] = torch.tensor(b["src_ids"], dtype=torch.long)
        tgt[i, : len(b["tgt_ids"])] = torch.tensor(b["tgt_ids"], dtype=torch.long)
        n_frames = b["visual"].shape[0]
        visual[i, :n_frames] = torch.from_numpy(b["visual"])
        visual_pad_mask[i, :n_frames] = False

    return {
        "src": src,
        "tgt": tgt,
        "src_pad_mask": (src == PAD_ID),
        "tgt_pad_mask": (tgt == PAD_ID),
        "visual": visual,                    # (B, N_frames, 512)
        "visual_pad_mask": visual_pad_mask,  # (B, N_frames)
        "video_ids": [b["video_id"] for b in batch],
    }


def make_loader(split, batch_size=64, num_workers=2, shuffle=None, **kwargs):
    ds = VATEXMultimodalDataset(split, **kwargs)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )


if __name__ == "__main__":
    ds = VATEXMultimodalDataset("val")
    print(f"Val size: {len(ds)}")
    s = ds[0]
    print(f"Keys: {list(s.keys())}")
    print(f"src_ids len: {len(s['src_ids'])}, tgt_ids len: {len(s['tgt_ids'])}")
    print(f"visual shape: {s['visual'].shape}, dtype: {s['visual'].dtype}")

    loader = make_loader("val", batch_size=4, num_workers=0)
    b = next(iter(loader))
    print(f"\nBatch src: {b['src'].shape}")
    print(f"Batch tgt: {b['tgt'].shape}")
    print(f"Batch visual: {b['visual'].shape}")
    print(f"Batch visual_pad_mask: {b['visual_pad_mask'].shape}")
    print(f"Frames per sample (True=pad): {b['visual_pad_mask'].sum(1).tolist()}")
