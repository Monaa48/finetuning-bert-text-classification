# Fine-tuning BERT for Text Classification (AG News)

## Overview
This repository contains an end-to-end pipeline to fine-tune a **BERT-family encoder** model for **4-class news topic classification** using the **AG News** dataset from HuggingFace Datasets.

**Task:** Encoder (BERT-family) — Text Classification (AG News)

## Dataset
- HuggingFace Datasets: `sh0416/ag_news`
- Input text: `title` + `description` (merged into a single `text` field)
- Labels: normalized to **0–3** (to match `num_labels=4`)

### Data sizes used (Full Run)
- Train: 108000
- Validation: 12000
- Test: 7600

## Model
- Checkpoint: `bert-base-uncased`
- Head: `AutoModelForSequenceClassification`
- Objective: 4-class classification

## Training Setup
- Max length: 128
- Epochs: 3
- Learning rate: 2e-5
- Train batch size: 16
- Eval batch size: 32
- Weight decay: 0.01
- Seed: 42
- Mixed precision: fp16 (if CUDA available)

## Results (Full Run)
| Split | Loss | Accuracy | Macro-F1 |
|------|------|----------|----------|
| Validation | 0.1883 | 0.9485 | 0.9483 |
| Test | 0.1965 | 0.9455 | 0.9455 |

## Repository Structure
Minimum required structure:
- `notebooks/` : training and evaluation notebooks
- `reports/`   : experiment results and analysis
- `requirements.txt` : dependencies

(Optional)
- `models/` : saved best model checkpoints
- `outputs/` : training logs / checkpoints

## Notebooks
- `notebooks/02_finetune_bert_ag_news.ipynb`  
  Fine-tuning + evaluation on AG News.

(If you use the Drive-based notebook, you can still keep it under `notebooks/` for submission.)

## How to Run

### Option A — Google Colab (Drive workflow)
If your notebook is set up to save outputs into Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = "/content/drive/MyDrive/finetuning-bert-text-classification"
```
Then run the notebook cells top-to-bottom.

### Option B — Local
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Open and run the notebook:
```bash
jupyter notebook
```

## Notes / Common Issues
- If you hit `CUDA error: device-side assert triggered`, it is usually caused by **label out-of-range**. Ensure labels are **0..3** for `num_labels=4`.
- If you previously hit a CUDA assert in Colab, **Restart runtime** before re-running training.

## Identification
- ANOM NUR MAULID — 1103223193 — TK4601 — GitHub: https://github.com/Monaa48
- DARRELL CHESTA ADABI — 1103223128 — TK4601 — GitHub: https://github.com/chessstaaa
