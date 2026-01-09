# Task 1A — Fine-tuning BERT for GoEmotions (Multi-label)

## Overview
This experiment fine-tunes a BERT-family encoder model (`bert-base-uncased`) for **multi-label emotion classification** using the GoEmotions dataset (`google-research-datasets/go_emotions`, config `raw`). Unlike standard multi-class classification (where each input belongs to exactly one class), GoEmotions is **multi-label**, meaning a single text can express multiple emotions simultaneously. Therefore, both the label representation and the evaluation procedure must be adapted for multi-label learning.

The notebook implements an end-to-end pipeline covering:
1) dataset loading and inspection,
2) preprocessing and label transformation into multi-hot vectors,
3) tokenization and batching,
4) model initialization for multi-label classification,
5) training with HuggingFace `Trainer`,
6) evaluation using multi-label metrics (Micro-F1, Macro-F1, Micro-Precision, Micro-Recall),
7) saving artifacts (model + tokenizer) and summarizing results.

---

## Environment and Output Management (Google Drive)
The notebook is designed for Google Colab and mounts Google Drive to persist outputs. A project directory is defined under Drive (e.g., `PROJECT_DIR = "/content/drive/MyDrive/finetuning-bert-text-classification"`). From this base folder, subdirectories such as `reports/`, `models/`, and `outputs/` are created automatically. This ensures:
- outputs remain available even if the Colab runtime resets,
- exported models can be copied to GitHub (or kept on Drive),
- reports are stored in a consistent location.

---

## Dataset Loading and Inspection
The dataset is loaded with:
- `load_dataset("google-research-datasets/go_emotions", "raw")`

After loading, the notebook inspects:
- the available splits (`train`, `validation`, `test`),
- column names (e.g., `text` and a label-related column),
- feature definitions (to see whether labels are stored as a sequence of label IDs).

This inspection step is critical because different dataset versions/configurations may use slightly different column names for labels. The notebook therefore includes logic to detect the label column robustly.

---

## Multi-label Label Processing (Multi-hot Encoding)
The central preprocessing step is converting the dataset’s label representation into a **multi-hot vector**:
- If the dataset provides a list of label IDs per example (e.g., `[2, 15, 21]`),
- it is converted into a vector of length `num_labels` where positions corresponding to present labels are set to 1, and all others are 0.

Example:
- `labels = [2, 15]` with `num_labels = 28` becomes:
  - `[0,0,1,0,0,...,1,...]`

This transformation is necessary because the model will be trained using a **binary relevance formulation** internally: each label is treated as a binary decision, and the model learns to output independent scores for each emotion label.

The transformation is done using `datasets.map(...)` in a batched manner, producing a new `labels` field that contains multi-hot vectors for each example.

---

## Tokenization
For an encoder model like BERT, raw text must be converted into token IDs:
- `AutoTokenizer.from_pretrained("bert-base-uncased")`

Tokenization steps include:
- truncation to a maximum sequence length (e.g., `MAX_LENGTH=128`),
- producing `input_ids`, `attention_mask` (and `token_type_ids` depending on the tokenizer),
- and optionally removing raw text columns after tokenization to reduce memory usage.

The notebook uses batched tokenization via `datasets.map(...)` for efficiency.

---

## Data Collation (Dynamic Padding)
Rather than padding all sequences to a fixed length globally, the notebook uses a dynamic padding collator:
- `DataCollatorWithPadding(tokenizer=tokenizer)`

This pads each batch to the maximum sequence length within that batch, which is generally more memory-efficient and faster than padding everything to a global maximum.

---

## Model Setup (Multi-label Classification)
The model is loaded as:
- `AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels, problem_type="multi_label_classification")`

The key point is `problem_type="multi_label_classification"`, which ensures the model uses the correct loss formulation for multi-label tasks (typically BCEWithLogitsLoss rather than CrossEntropyLoss).

In multi-label classification:
- the model outputs a vector of logits of shape `[batch_size, num_labels]`,
- each logit corresponds to a label score (not mutually exclusive).

---

## Training Configuration (TrainingArguments + Trainer)
Training is configured through `TrainingArguments` and executed via `Trainer`.

Important configuration choices:
- Evaluation per epoch (to track progress and select the best checkpoint).
- Saving per epoch (to keep checkpoints aligned with evaluation).
- `load_best_model_at_end=True` and selecting a metric (Micro-F1) for “best model”.
- Mixed precision (`fp16=True` when CUDA is available) to reduce VRAM usage and speed training on GPUs like T4.

The notebook includes compatibility handling for Transformers API changes (e.g., `eval_strategy` vs `evaluation_strategy`) to avoid runtime errors due to version differences.

---

## Evaluation Procedure (Multi-label Metrics)
Unlike multi-class classification, evaluation in multi-label classification requires converting logits to binary decisions:
1) Apply sigmoid to logits to obtain probabilities:
   - `probs = sigmoid(logits)`
2) Apply a threshold (default 0.5) to produce binary predictions:
   - `preds = (probs >= 0.5)`

Then metrics are computed:
- **Micro-F1**: emphasizes overall label-instance performance; sensitive to frequent labels.
- **Macro-F1**: averages F1 across labels; highlights performance on rare labels.
- **Micro-Precision** and **Micro-Recall**: useful to see whether the model is conservative (high precision, low recall) or aggressive (high recall, low precision).

---

## Full Run Data Sizes
This run used the full dataset sizes:
- Train: 168980
- Validation: 21122
- Test: 21123

These sizes serve as evidence that the experiment is not a reduced subset run.

---

## Results
| Split | Loss | Micro-F1 | Macro-F1 | Micro-Precision | Micro-Recall |
|------|------|----------|----------|-----------------|--------------|
| Validation | 0.1105 | 0.3760 | 0.2867 | 0.5948 | 0.2749 |
| Test | 0.1109 | 0.3771 | 0.2846 | 0.5989 | 0.2752 |

---

## Interpretation of Results
The model achieves moderate Micro-F1 and lower Macro-F1, which is common for GoEmotions:
- Micro metrics tend to be dominated by more frequent emotions.
- Macro-F1 is typically lower because rare labels are harder to predict reliably.

Precision is higher than recall, indicating the model is somewhat conservative at the default threshold (0.5): it avoids predicting many labels unless confidence is sufficiently high. This suggests that threshold tuning (or label-wise thresholds) could improve recall and potentially improve F1 depending on the target trade-off.

---

## Saved Artifacts
After training, the notebook saves:
- fine-tuned model weights,
- tokenizer files,
to a Drive directory under `models/` (e.g., `models/bert_go_emotions_best`).

These artifacts can be reused later for inference or further fine-tuning.

---

## Next Improvements / Additional Analysis
To strengthen the report further:
1) **Per-label analysis**:
   - identify the most common labels and report per-label precision/recall/F1.
2) **Threshold tuning**:
   - try thresholds like 0.3–0.6 and compare Micro-F1/Macro-F1.
3) **Error analysis samples**:
   - show a few examples where the model misses labels (false negatives) or predicts extra labels (false positives).
4) **Class imbalance handling**:
   - experiment with weighted loss or focal loss if desired (optional).

These additions help move the report beyond metrics into qualitative understanding of model behavior.
