# TicketTriage-SLM

> A research-quality customer support ticket triage system comparing
> Traditional ML, Zero-Shot LLM, and LoRA Fine-Tuned LLM approaches.

---

## Quick Start

### 1. Clone & Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python -m src.data_generation
```

### 3. Preprocess & Split

```bash
python -m src.preprocessing
```

### 4. Train Baseline

```bash
python -m src.baseline
```

### 5. Run Zero-Shot Inference

```bash
python -m src.zero_shot
```

### 6. Run Evaluation

```bash
python -m src.evaluation
```

### 7. Launch Gradio App

```bash
python -m app.app
```

---

## Models

| Model | Description |
|---|---|
| **TF-IDF + Logistic Regression** | Traditional ML baseline using scikit-learn |
| **Qwen2.5-0.5B-Instruct (Zero-Shot)** | Small LLM with structured prompt, no training |
| **Qwen2.5-0.5B-Instruct + LoRA** | Same LLM fine-tuned with LoRA on training data |

---

## Project Status

> 🚧 **Under Development** — See `PROJECT_SPEC.md` for full specification.

| Phase | Status |
|---|---|
| Project setup | ✅ Complete |
| Dataset generation | ⬜ Not started |
| Baseline model | ⬜ Not started |
| Zero-shot inference | ⬜ Not started |
| LoRA fine-tuning | ⬜ Not started |
| Evaluation | ⬜ Not started |
| Gradio app | ⬜ Not started |

---

## License

This project is for research and educational purposes.
