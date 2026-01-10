# Fine-tuning BERT for Text Classification — Task 1 (AG News + GoEmotions)

## Overview
This repository contains end-to-end pipelines to fine-tune a **BERT-family encoder** model for:
1) **Multi-class text classification** (AG News), and
2) **Multi-label emotion classification** (GoEmotions)

The workflow covers preprocessing, tokenization, fine-tuning, and evaluation.

> Note: **MNLI (NLI)** is submitted in a separate repository `finetuning-bert-nli` (as required by the assignment).

---

## Datasets

### 1) AG News (Multi-class)
- Dataset: `sh0416/ag_news`
- Input: `title + description` (merged into a single `text` field)
- Labels: normalized to **0–3** (to match `num_labels=4`)

### 2) GoEmotions (Multi-label)
- Dataset: `google-research-datasets/go_emotions` (config: `raw`)
- Input: `text`
- Labels: multi-label (converted to multi-hot vectors inside the notebook)

---

## Model
- Base checkpoint: `bert-base-uncased`
- Framework: HuggingFace Transformers + Datasets

---

## Training Setup (default)
(Used as the base configuration in the notebooks; adjust if needed)
- Max length: 128 (AG News), 128 (GoEmotions)
- Epochs: 3
- Learning rate: 2e-5
- Train batch size: 16
- Eval batch size: 32
- Weight decay: 0.01
- Seed: 42
- Mixed precision: fp16 (if CUDA available)

---

## Results

### AG News — Full Run
**Data sizes**
- Train: 108000
- Validation: 12000
- Test: 7600

| Split | Loss | Accuracy | Macro-F1 |
|------|------|----------|----------|
| Validation | 0.1883 | 0.9485 | 0.9483 |
| Test | 0.1965 | 0.9455 | 0.9455 |

### GoEmotions — Full Run
**Data sizes**
- Train: 168980
- Validation: 21122
- Test: 21123

| Split | Loss | Micro-F1 | Macro-F1 | Micro-Precision | Micro-Recall |
|------|------|----------|----------|-----------------|--------------|
| Validation | 0.1105 | 0.3760 | 0.2867 | 0.5948 | 0.2749 |
| Test | 0.1109 | 0.3771 | 0.2846 | 0.5989 | 0.2752 |


---

## Repository Structure
Minimum required structure:
- `README.md`
- `notebooks/`
- `reports/`
- `requirements.txt`

(Optional)
- `models/`
- `outputs/`

---

## Notebooks
- `notebooks/finetune_bert_ag_news.ipynb`  
  Fine-tuning + evaluation for AG News.
- `notebooks/finetune_bert_go_emotions.ipynb`  
  Fine-tuning + evaluation for GoEmotions.

---

## How to Run (Google Colab + Drive workflow)
Mount Google Drive and set the project directory:

```python
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = "/content/drive/MyDrive/finetuning-bert-text-classification"
