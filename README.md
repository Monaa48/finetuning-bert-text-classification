# Fine-tuning BERT for Text Classification (AG News)

## Overview
This repository contains an end-to-end pipeline to fine-tune a BERT-family encoder model for multi-class text classification using the AG News dataset.

## Dataset
- sh0416/ag_news (HuggingFace Datasets)

## Model
- bert-base-uncased
- Objective: 4-class news topic classification

## Repository Structure
- notebooks/ : training and evaluation notebooks (with explanations)
- reports/   : experiment results and analysis
- requirements.txt : dependencies

## Notebooks
1. notebooks/01_data_and_baseline.ipynb
2. notebooks/02_finetune_bert.ipynb
3. notebooks/03_evaluation_and_analysis.ipynb

## Results (fill after running)
| Model | Epochs | Accuracy | Macro-F1 |
|------|--------|----------|----------|
| bert-base-uncased |  |  |  |

## How to Run
1. Create environment (Colab or local)
2. Install requirements: `pip install -r requirements.txt`
3. Run notebooks in order.

## Identification
- ANOM NUR MAULID - 1103223193 - TK4601
- DARRELL CHESTA ADABI - 1103223128 - TK4601
