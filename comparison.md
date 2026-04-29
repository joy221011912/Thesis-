# Step 4 — Test-set evaluation

| Model | BLEU | chrF++ | TER | METEOR | BERTScore_F1 | COMET |
|---|---|---|---|---|---|---|
| Step 2 (text-only) | 6.23 | 6.22 | 101.78 | 16.39 | 69.69 | 54.18 |
| Step 3 CVT (warm-start) | 9.78 | 8.24 | 104.38 | 20.02 | 72.81 | 61.04 |
| Step 3 CVT (from scratch) | 10.68 | 8.65 | 102.50 | 21.55 | 73.09 | 62.18 |

_Higher = better for all except **TER** (lower is better)._