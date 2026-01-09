# Task 1 — Fine-tuning BERT for Text Classification (AG News)

Generated: 2026-01-09 04:09:32

## Setup
- Dataset: sh0416/ag_news
- Model: bert-base-uncased
- Task: 4-class news topic classification (AG News)
- Max length: 128
- Epochs: 3
- Learning rate: 2e-05
- Train batch size: 16
- Eval batch size: 32
- Weight decay: 0.01
- Quick run: False (full training run)

## Results
| Split | Loss | Accuracy | Macro-F1 |
|------|------|----------|----------|
| Validation | 0.3228 | 0.9145 | 0.9125 |
| Test | 0.3331 | 0.9120 | 0.9128 |

## Notes
- Labels were normalized to 0–3 to match `num_labels=4`.
- Recommended next additions:
  - Confusion matrix
  - Per-class precision/recall/F1 (classification report)
  - A few example predictions (correct vs. incorrect) for error analysis
