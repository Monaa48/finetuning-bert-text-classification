# Task 1 — Fine-tuning BERT for AG News (Multi-class)

## Overview
This experiment fine-tunes a BERT-family encoder model (`bert-base-uncased`) for **multi-class text classification** on the AG News dataset (`sh0416/ag_news`). The objective is to classify each news sample into one of **4 topic categories**. The notebook implements an end-to-end pipeline consisting of dataset loading, preprocessing, tokenization, model fine-tuning with HuggingFace Trainer, evaluation, and saving outputs.

The implementation is designed for Google Colab and supports saving outputs to Google Drive for easy export to GitHub.

---

## Environment and Output Management (Google Drive)
The workflow mounts Google Drive and defines a project directory (e.g., `PROJECT_DIR = "/content/drive/MyDrive/finetuning-bert-text-classification"`). Under this directory, the notebook creates consistent subfolders such as:
- `reports/` for result summaries,
- `outputs/` for trainer checkpoints/logs,
- `models/` for exporting the best model.

This helps preserve results across runtime resets and ensures the final artifacts can be uploaded to GitHub without relying on Colab session storage.

---

## Dataset Loading and Schema Inspection
The dataset is loaded via HuggingFace Datasets:
- `load_dataset("sh0416/ag_news")`

A key detail of this dataset variant is that it provides separate text fields:
- `title`
- `description`
and a label field:
- `label`

The notebook inspects available columns and example rows to confirm the schema before preprocessing.

---

## Train/Validation Split
The dataset typically provides `train` and `test`. To evaluate during training, the notebook creates a **validation split** from the training data using `train_test_split(...)` with a fixed random seed. This ensures:
- reproducibility of the split,
- comparable results across runs.

---

## Label Normalization (Critical for Stability)
Some variants of AG News encode labels as **1..4** instead of **0..3**. However, PyTorch’s classification loss expects labels to be in the range:
- `0` to `num_labels - 1`

If labels are out of range (e.g., label `4` when `num_labels=4`), training can crash with CUDA device-side assertion errors. To prevent this, the notebook includes a label normalization step:
- if labels start at 1, shift them by `-1` to become `0..3`.

This step is explicitly validated with a min/max label check before training.

---

## Text Construction (title + description → text)
Because the dataset has `title` and `description` rather than a single `text` column, the notebook constructs a unified input:
- `text = title + " " + description`

This merged text field is then used for tokenization. This ensures the model receives a complete representation of each news sample.

---

## Tokenization
The model uses:
- `AutoTokenizer.from_pretrained("bert-base-uncased")`

Tokenization converts raw text into:
- `input_ids`
- `attention_mask`
(and possibly `token_type_ids` depending on the tokenizer)

Truncation is applied with:
- `MAX_LENGTH = 128`
to keep input length manageable and efficient on GPU.

After tokenization, raw text columns are removed to reduce memory footprint.

---

## Model Initialization
The notebook loads a classification head on top of BERT:
- `AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)`

This creates logits of size 4 for each example, corresponding to the 4 AG News classes.

---

## Metrics (Accuracy + Macro-F1)
Evaluation uses:
- **Accuracy**: overall fraction of correct predictions.
- **Macro-F1**: F1 averaged across classes equally, which helps reflect balanced performance across categories.

Macro-F1 is used as the “best model” selection metric (`metric_for_best_model="f1_macro"`) so the saved checkpoint reflects balanced class performance rather than only accuracy.

---

## Training with HuggingFace Trainer
Training is managed via `TrainingArguments` and `Trainer`, with key settings:
- evaluation each epoch,
- saving checkpoints each epoch,
- `load_best_model_at_end=True`,
- mixed precision fp16 enabled when CUDA is available (useful on T4 GPUs),
- fixed `seed` for reproducibility.

The notebook includes compatibility handling for Transformers version differences (e.g., `eval_strategy` vs `evaluation_strategy`) so it can run reliably on Colab environments with different package versions.

---

## Full Run Data Sizes
This run used full dataset sizes:
- Train: 108000
- Validation: 12000
- Test: 7600

These sizes indicate that the experiment is not a reduced subset run.

---

## Results (Full Run)
| Split | Loss | Accuracy | Macro-F1 |
|------|------|----------|----------|
| Validation | 0.1883 | 0.9485 | 0.9483 |
| Test | 0.1965 | 0.9455 | 0.9455 |

The results show strong performance on AG News after 3 epochs of fine-tuning.

---

## Saved Artifacts
After training:
- the best model checkpoint is saved to `models/`,
- trainer outputs/checkpoints (if enabled) are stored under `outputs/`.

These artifacts can be reused for inference or further experiments.

---

## Next Improvements / Additional Analysis
To make the report stronger:
1) **Confusion matrix** to see which classes are most frequently confused.
2) **Per-class precision/recall/F1** to identify weak classes.
3) **Error analysis** with a few example misclassifications.
4) **Learning curve tracking** (optional) by logging training/eval metrics per epoch.

These additions can help demonstrate deeper understanding beyond the final metrics.
