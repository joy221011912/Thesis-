"""
============================================================
Step 2a: Train SentencePiece Tokenizers (English + Chinese)
============================================================

Trains two separate SentencePiece BPE tokenizers using captions
from the TRAINING SPLIT only (no leakage from val/test).

OUTPUT:
    tokenizers/en_tokenizer.model   (+ .vocab)
    tokenizers/zh_tokenizer.model   (+ .vocab)

USAGE:
    python3 data_04_train_tokenizers.py
============================================================
"""

import os
import json
import argparse
import sentencepiece as spm
from datasets import load_from_disk


DEFAULT_DATASET_DIR = "vatex"
DEFAULT_SPLITS_FILE = "data_splits.json"
DEFAULT_OUTPUT_DIR  = "tokenizers"
DEFAULT_VOCAB_SIZE  = 16000

# Special token IDs (match what SentencePiece will produce)
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


def write_corpus(captions, path):
    with open(path, "w", encoding="utf-8") as f:
        for c in captions:
            c = c.strip().replace("\n", " ")
            if c:
                f.write(c + "\n")


def train_tokenizer(corpus_path, model_prefix, vocab_size, character_coverage):
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID, unk_id=UNK_ID,
        pad_piece="<pad>", bos_piece="<s>", eos_piece="</s>", unk_piece="<unk>",
        max_sentence_length=8192,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    p.add_argument("--splits-file", default=DEFAULT_SPLITS_FILE)
    p.add_argument("--output-dir",  default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--vocab-size",  type=int, default=DEFAULT_VOCAB_SIZE)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Step 2a: Training SentencePiece tokenizers")
    print("=" * 60)

    # Load split info and annotations
    with open(args.splits_file) as f:
        splits = json.load(f)
    train_ids = set(splits["train"])

    dataset = load_from_disk(os.path.join(args.dataset_dir, "json"))
    train_data = dataset["train"]

    # Collect captions from training videos only
    en_captions, zh_captions = [], []
    for i in range(len(train_data)):
        entry = train_data[i]
        if entry["videoID"] in train_ids:
            en_captions.extend(entry["enCap"])
            zh_captions.extend(entry["chCap"])

    print(f"  EN captions: {len(en_captions)}")
    print(f"  ZH captions: {len(zh_captions)}")

    en_corpus = os.path.join(args.output_dir, "_en_corpus.txt")
    zh_corpus = os.path.join(args.output_dir, "_zh_corpus.txt")
    write_corpus(en_captions, en_corpus)
    write_corpus(zh_captions, zh_corpus)

    print("\n  Training English tokenizer...")
    train_tokenizer(
        en_corpus,
        os.path.join(args.output_dir, "en_tokenizer"),
        args.vocab_size,
        character_coverage=1.0,
    )

    print("  Training Chinese tokenizer...")
    train_tokenizer(
        zh_corpus,
        os.path.join(args.output_dir, "zh_tokenizer"),
        args.vocab_size,
        character_coverage=0.9995,  # Chinese needs slightly < 1.0
    )

    # Clean up corpus files
    os.remove(en_corpus)
    os.remove(zh_corpus)

    # Quick verification
    print("\n  Verification:")
    en_sp = spm.SentencePieceProcessor(model_file=os.path.join(args.output_dir, "en_tokenizer.model"))
    zh_sp = spm.SentencePieceProcessor(model_file=os.path.join(args.output_dir, "zh_tokenizer.model"))
    sample_en = "A man is playing cricket with a bat"
    sample_zh = "一个男人用球棒打板球"
    print(f"  EN vocab size: {en_sp.get_piece_size()}")
    print(f"  ZH vocab size: {zh_sp.get_piece_size()}")
    print(f"  EN sample:  '{sample_en}'")
    print(f"           -> {en_sp.encode(sample_en, out_type=str)}")
    print(f"  ZH sample:  '{sample_zh}'")
    print(f"           -> {zh_sp.encode(sample_zh, out_type=str)}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
