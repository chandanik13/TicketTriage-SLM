# TicketTriage-SLM — Project Specification

## Overview

**TicketTriage-SLM** is a research-quality customer support ticket triage
system that compares three machine learning approaches:

| # | Model | Type |
|---|---|---|
| 1 | TF-IDF + Logistic Regression | Traditional ML baseline |
| 2 | Qwen2.5-0.5B-Instruct Zero-Shot | Small language model, no fine-tuning |
| 3 | Qwen2.5-0.5B-Instruct + LoRA | Small language model, fine-tuned with LoRA |

All three models analyze the same customer support ticket and produce
five structured fields:

```json
{
  "category": "billing",
  "priority": "high",
  "sentiment": "negative",
  "language": "english",
  "resolution": "refund_duplicate_charge"
}
```

---

## Label Definitions

### Categories
`billing` · `technical` · `account` · `shipping` · `refund` · `subscription`

### Priorities
`low` · `medium` · `high` · `urgent`

### Sentiments
`positive` · `neutral` · `negative`

### Languages
`english`

### Resolutions
`refund_duplicate_charge` · `reset_password` · `resolve_login_issue` ·
`track_order` · `cancel_subscription` · `technical_troubleshooting`

---

## Dataset

- **Size:** ~5,000 synthetic customer support tickets
- **Split:** 70% train / 15% validation / 15% test
- **Seed:** 42
- **Immutability:** Raw dataset is never modified after generation

### Required Columns
`ticket_id` · `text` · `category` · `priority` · `sentiment` · `language` · `resolution`

---

## Evaluation Metrics

- Category accuracy, macro F1, weighted F1
- Priority accuracy
- Sentiment accuracy
- Language accuracy
- Resolution accuracy
- Average field accuracy
- JSON validity rate
- Inference latency

---

## Hardware Requirements

| Task | Hardware |
|---|---|
| Data generation, baseline, evaluation | CPU (any machine) |
| Qwen zero-shot inference | CPU (slow) or GPU (fast) |
| LoRA training | NVIDIA T4 16 GB (Google Colab) |
| Gradio application | CPU (any machine) |

---

## Project Structure

```
TicketTriage-SLM/
├── data/
│   ├── raw/              # Immutable raw dataset
│   └── processed/        # Train/val/test splits
├── notebooks/            # Jupyter notebooks (Colab training)
├── src/
│   ├── __init__.py
│   ├── config.py         # All constants and paths
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── baseline.py
│   ├── zero_shot.py
│   ├── dataset.py
│   ├── train_lora.py
│   ├── inference.py
│   └── evaluation.py
├── models/
│   ├── baseline/         # TF-IDF + LR artifacts
│   └── qwen_lora/        # LoRA adapter weights
├── evaluation/           # Metrics, reports, CSVs
├── app/
│   ├── __init__.py
│   └── app.py            # Gradio dashboard
├── tests/                # Unit tests
├── requirements.txt
├── .gitignore
├── README.md
└── PROJECT_SPEC.md
```
