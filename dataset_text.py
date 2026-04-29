"""
============================================================
Step 2: VATEX Translation Dataset (Text-Only, Step 2)
============================================================

PyTorch Dataset + DataLoader factory for EN->ZH translation.
Pairs enCap[i] with chCap[i] (1:1 by index).

Returns per sample:
    {
        'src_ids':   list[int],   English token IDs (with BOS/EOS)
        'tgt_ids':   list[int],   Chinese token IDs (with BOS/EOS)
        'video_id':  str,         For Step 3 (not used in Step 2)
    }

Collate produces padded (src, tgt) tensors and padding masks.
============================================================
"""

import os
import json
import torch
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk


PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
MAX_LEN = 128


class VATEXTranslationDataset(Dataset):
    def __init__(
        self,
        split_name,                 # "train" | "val" | "test"
        dataset_dir="vatex",
        splits_file="data_splits.json",
        en_tokenizer_path="tokenizers/en_tokenizer.model",
        zh_tokenizer_path="tokenizers/zh_tokenizer.model",
        max_len=MAX_LEN,
    ):
        self.max_len = max_len
        self.en_sp = spm.SentencePieceProcessor(model_file=en_tokenizer_path)
        self.zh_sp = spm.SentencePieceProcessor(model_file=zh_tokenizer_path)

        with open(splits_file) as f:
            splits = json.load(f)
        split_ids = set(splits[split_name])

        dataset = load_from_disk(os.path.join(dataset_dir, "json"))
        train_data = dataset["train"]  # all splits pulled from VATEX train

        # Flatten: one sample per (video, caption-index)
        self.samples = []
        for i in range(len(train_data)):
            entry = train_data[i]
            vid = entry["videoID"]
            if vid not in split_ids:
                continue
            en_caps = entry["enCap"]
            zh_caps = entry["chCap"]
            n = min(len(en_caps), len(zh_caps))
            for j in range(n):
                self.samples.append((vid, en_caps[j], zh_caps[j]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, en, zh = self.samples[idx]
        src = [BOS_ID] + self.en_sp.encode(en, out_type=int) + [EOS_ID]
        tgt = [BOS_ID] + self.zh_sp.encode(zh, out_type=int) + [EOS_ID]
        src = src[: self.max_len]
        tgt = tgt[: self.max_len]
        return {"src_ids": src, "tgt_ids": tgt, "video_id": vid}


def collate_fn(batch):
    """Pad sequences to max length in the batch."""
    src_lens = [len(b["src_ids"]) for b in batch]
    tgt_lens = [len(b["tgt_ids"]) for b in batch]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src = torch.full((len(batch), max_src), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        src[i, : len(b["src_ids"])] = torch.tensor(b["src_ids"], dtype=torch.long)
        tgt[i, : len(b["tgt_ids"])] = torch.tensor(b["tgt_ids"], dtype=torch.long)

    return {
        "src": src,                                    # (B, S)
        "tgt": tgt,                                    # (B, T)
        "src_pad_mask": (src == PAD_ID),               # (B, S)  True = pad
        "tgt_pad_mask": (tgt == PAD_ID),               # (B, T)  True = pad
        "video_ids": [b["video_id"] for b in batch],
    }


def make_loader(split, batch_size=64, num_workers=2, shuffle=None, **kwargs):
    ds = VATEXTranslationDataset(split, **kwargs)
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
    # Smoke test
    ds = VATEXTranslationDataset("val")
    print(f"Val set size: {len(ds)}")
    print(f"Sample 0 keys: {list(ds[0].keys())}")
    print(f"Sample 0 src len: {len(ds[0]['src_ids'])}, tgt len: {len(ds[0]['tgt_ids'])}")

    loader = make_loader("val", batch_size=4, num_workers=0)
    batch = next(iter(loader))
    print(f"Batch src shape: {batch['src'].shape}")
    print(f"Batch tgt shape: {batch['tgt'].shape}")
    print(f"Video IDs: {batch['video_ids']}")
