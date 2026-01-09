# Task 1 — Fine-tuning BERT for Text Classification (AG News)

Generated: 2026-01-09 04:09:32

## Setup
- Dataset: sh0416/ag_news
- Model: bert-base-uncased
- Max length: 128
- Epochs: 3
- Learning rate: 2e-05
- Train batch size: 16
- Eval batch size: 32
- Weight decay: 0.01
- Quick run: True

## Results
| Split | Accuracy | Macro-F1 |
|------|----------|----------|
| Validation | 0.9145 | 0.9125313393670835 |
| Test | 0.912 | 0.9128472773674903 |

## Notes
- This report is a minimal summary. Add more analysis (examples of errors, per-class performance, etc.).
