"""
============================================================
Step 4b: Compute metrics on generated translations
============================================================

Reads results/translations_<model>.json and computes:

    Core MT quality
    - BLEU   (sacrebleu, zh tokenizer)
    - chrF++ (sacrebleu)
    - TER    (sacrebleu)
    - METEOR (nltk, over jieba-segmented Chinese)

    Semantic
    - BERTScore (xlm-roberta-base, multilingual)
    - COMET     (Unbabel/wmt22-comet-da)  [--no-comet to skip]

    Diagnostics
    - Length ratio (hyp_chars / ref_chars)
    - Unknown-token rate in hypotheses
    - Per-length-bucket BLEU (short / medium / long by src-word count)

Writes:
    results/metrics_<model>.json
    results/comparison.md    (side-by-side table)

USAGE:
    python3 eval_metrics.py
    python3 eval_metrics.py --no-comet --no-bertscore   # quick
============================================================
"""

import os
import json
import argparse
import numpy as np

import sacrebleu
import jieba
import nltk
from nltk.translate.meteor_score import meteor_score

MODELS = ["baseline", "cvt_warmstart", "cvt_fromscratch"]
PRETTY = {
    "baseline":        "Baseline (text-only Transformer)",
    "cvt_warmstart":   "CVT (warm-started from baseline)",
    "cvt_fromscratch": "CVT (trained from scratch)",
}


def ensure_nltk():
    for res in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)


def seg_zh(s):
    """Jieba-segmented Chinese — list of tokens."""
    return [w for w in jieba.cut(s) if w.strip()]


def compute_corpus_metrics(hyps, refs):
    """hyps, refs: list[str]. Returns dict of corpus-level scores."""
    out = {}

    # sacrebleu handles Chinese tokenization via tokenize="zh"
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
    out["BLEU"] = bleu.score

    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)  # chrF++
    out["chrF++"] = chrf.score

    ter = sacrebleu.corpus_ter(hyps, [refs])
    out["TER"] = ter.score  # lower is better

    # METEOR — per-sentence then average, on jieba-segmented tokens
    meteors = []
    for h, r in zip(hyps, refs):
        h_tok = seg_zh(h)
        r_tok = seg_zh(r)
        if not h_tok or not r_tok:
            meteors.append(0.0)
            continue
        meteors.append(meteor_score([r_tok], h_tok))
    out["METEOR"] = 100.0 * float(np.mean(meteors))

    return out


def compute_bertscore(hyps, refs):
    from bert_score import score as bscore
    P, R, F1 = bscore(hyps, refs, lang="zh", verbose=False,
                      rescale_with_baseline=False)
    return {"BERTScore_F1": 100.0 * float(F1.mean())}


def compute_comet(srcs, hyps, refs):
    from comet import download_model, load_from_checkpoint
    print("    [COMET] downloading model (first run only)...")
    path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(path)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    out = model.predict(data, batch_size=16, gpus=0, num_workers=2,
                        progress_bar=False)
    return {"COMET": 100.0 * float(out["system_score"])}


def compute_diagnostics(srcs, hyps, refs):
    diag = {}
    hyp_chars = sum(len(h) for h in hyps)
    ref_chars = sum(len(r) for r in refs)
    diag["length_ratio_chars"] = hyp_chars / max(ref_chars, 1)

    # Unknown token marker — SentencePiece decodes unk as empty/replacement,
    # so count proxy: hyp strings containing the "⁇" replacement char.
    diag["unk_rate_pct"] = 100.0 * sum("⁇" in h for h in hyps) / max(len(hyps), 1)

    # Per-length-bucket BLEU (by source English word count)
    buckets = {"short(<=5w)": [], "medium(6-12w)": [], "long(>12w)": []}
    for s, h, r in zip(srcs, hyps, refs):
        n = len(s.split())
        if n <= 5:   buckets["short(<=5w)"].append((h, r))
        elif n <= 12: buckets["medium(6-12w)"].append((h, r))
        else:         buckets["long(>12w)"].append((h, r))

    diag["bleu_by_length"] = {}
    for name, pairs in buckets.items():
        if not pairs:
            diag["bleu_by_length"][name] = None
            continue
        hs = [p[0] for p in pairs]
        rs = [p[1] for p in pairs]
        bleu = sacrebleu.corpus_bleu(hs, [rs], tokenize="zh")
        diag["bleu_by_length"][name] = {"n": len(pairs), "BLEU": bleu.score}

    return diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--no-bertscore", action="store_true")
    p.add_argument("--no-comet", action="store_true")
    args = p.parse_args()

    ensure_nltk()

    all_metrics = {}
    for name in MODELS:
        path = os.path.join(args.results_dir, f"translations_{name}.json")
        if not os.path.exists(path):
            print(f"  [skip] {name}: {path} missing")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        srcs = [d["src"] for d in data]
        hyps = [d["hyp"] for d in data]
        refs = [d["ref"] for d in data]

        print(f"\n  >> {PRETTY[name]}  (n={len(data)})")
        m = compute_corpus_metrics(hyps, refs)
        print(f"    BLEU={m['BLEU']:.2f}  chrF++={m['chrF++']:.2f}  "
              f"TER={m['TER']:.2f}  METEOR={m['METEOR']:.2f}")

        if not args.no_bertscore:
            bs = compute_bertscore(hyps, refs)
            m.update(bs)
            print(f"    BERTScore_F1={bs['BERTScore_F1']:.2f}")

        if not args.no_comet:
            cm = compute_comet(srcs, hyps, refs)
            m.update(cm)
            print(f"    COMET={cm['COMET']:.2f}")

        diag = compute_diagnostics(srcs, hyps, refs)
        print(f"    length_ratio={diag['length_ratio_chars']:.3f}  "
              f"unk_rate={diag['unk_rate_pct']:.3f}%")
        for bname, bv in diag["bleu_by_length"].items():
            if bv:
                print(f"    [{bname:<16}] n={bv['n']:>5}  BLEU={bv['BLEU']:.2f}")

        m["diagnostics"] = diag
        out_path = os.path.join(args.results_dir, f"metrics_{name}.json")
        with open(out_path, "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        all_metrics[name] = m

    # ── Comparison table ──
    if all_metrics:
        cols = ["BLEU", "chrF++", "TER", "METEOR"]
        if not args.no_bertscore: cols.append("BERTScore_F1")
        if not args.no_comet:     cols.append("COMET")

        md = ["# Step 4 — Test-set evaluation\n"]
        md.append("| Model | " + " | ".join(cols) + " |")
        md.append("|" + "---|" * (len(cols) + 1))
        for name in MODELS:
            if name not in all_metrics:
                continue
            m = all_metrics[name]
            row = [PRETTY[name]]
            for c in cols:
                v = m.get(c)
                row.append(f"{v:.2f}" if v is not None else "—")
            md.append("| " + " | ".join(row) + " |")
        md.append("\n_Higher = better for all except **TER** (lower is better)._\n")

        md_path = os.path.join(args.results_dir, "comparison.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        print(f"\n  Comparison table -> {md_path}")
        print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
