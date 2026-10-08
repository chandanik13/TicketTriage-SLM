"""
TicketTriage-SLM — Central Configuration

All project constants, paths, label definitions, and model settings
are defined here.  Every other module imports from this file so that
nothing is hard-coded elsewhere.
"""

from pathlib import Path
from typing import Dict, List


# ──────────────────────────────────────────────
# Project root
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────
# Directory layout
# ──────────────────────────────────────────────
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
MODELS_DIR      = PROJECT_ROOT / "models"
BASELINE_DIR    = MODELS_DIR / "baseline"
QWEN_LORA_DIR   = MODELS_DIR / "qwen_lora"
EVALUATION_DIR  = PROJECT_ROOT / "evaluation"
NOTEBOOKS_DIR   = PROJECT_ROOT / "notebooks"
APP_DIR         = PROJECT_ROOT / "app"
TESTS_DIR       = PROJECT_ROOT / "tests"

# ──────────────────────────────────────────────
# Data files
# ──────────────────────────────────────────────
RAW_TICKETS_PATH  = RAW_DATA_DIR / "tickets.csv"
TRAIN_PATH        = PROCESSED_DIR / "train.csv"
VAL_PATH          = PROCESSED_DIR / "validation.csv"
TEST_PATH         = PROCESSED_DIR / "test.csv"

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED: int = 42

# ──────────────────────────────────────────────
# Dataset size
# ──────────────────────────────────────────────
NUM_SAMPLES: int = 5_000

# ──────────────────────────────────────────────
# Data split ratios
# ──────────────────────────────────────────────
TRAIN_RATIO: float = 0.70
VAL_RATIO: float   = 0.15
TEST_RATIO: float  = 0.15

# ──────────────────────────────────────────────
# Label definitions
# ──────────────────────────────────────────────
CATEGORIES: List[str] = [
    "billing",
    "technical",
    "account",
    "shipping",
    "refund",
    "subscription",
]

PRIORITIES: List[str] = [
    "low",
    "medium",
    "high",
    "urgent",
]

SENTIMENTS: List[str] = [
    "positive",
    "neutral",
    "negative",
]

LANGUAGES: List[str] = [
    "english",
]

RESOLUTIONS: List[str] = [
    "refund_duplicate_charge",
    "reset_password",
    "resolve_login_issue",
    "track_order",
    "cancel_subscription",
    "technical_troubleshooting",
]

# ──────────────────────────────────────────────
# Valid category → resolution mappings
# ──────────────────────────────────────────────
# Each category maps to the resolutions that make sense for it.
# This prevents impossible combinations in the synthetic dataset.
CATEGORY_RESOLUTION_MAP: Dict[str, List[str]] = {
    "billing":       ["refund_duplicate_charge"],
    "refund":        ["refund_duplicate_charge"],
    "shipping":      ["track_order"],
    "subscription":  ["cancel_subscription"],
    "account":       ["reset_password", "resolve_login_issue"],
    "technical":     ["technical_troubleshooting"],
}

# ──────────────────────────────────────────────
# Required columns in the dataset
# ──────────────────────────────────────────────
REQUIRED_COLUMNS: List[str] = [
    "ticket_id",
    "text",
    "category",
    "priority",
    "sentiment",
    "language",
    "resolution",
]

# ──────────────────────────────────────────────
# Structured output fields (what each model must produce)
# ──────────────────────────────────────────────
OUTPUT_FIELDS: List[str] = [
    "category",
    "priority",
    "sentiment",
    "language",
    "resolution",
]

# ──────────────────────────────────────────────
# Label lookup (for validation)
# ──────────────────────────────────────────────
VALID_LABELS: Dict[str, List[str]] = {
    "category":   CATEGORIES,
    "priority":   PRIORITIES,
    "sentiment":  SENTIMENTS,
    "language":   LANGUAGES,
    "resolution": RESOLUTIONS,
}

# ──────────────────────────────────────────────
# Model configuration
# ──────────────────────────────────────────────
QWEN_MODEL_NAME: str = "Qwen/Qwen2.5-0.5B-Instruct"

# LoRA hyper-parameters (T4-friendly defaults)
LORA_R: int            = 16
LORA_ALPHA: int        = 32
LORA_DROPOUT: float    = 0.05
LORA_TARGET_MODULES: List[str] = ["q_proj", "v_proj"]

# Training hyper-parameters (T4 16 GB)
TRAIN_BATCH_SIZE: int          = 2
EVAL_BATCH_SIZE: int           = 4
GRADIENT_ACCUMULATION: int     = 8
LEARNING_RATE: float           = 2e-4
NUM_EPOCHS: int                = 3
MAX_SEQ_LENGTH: int            = 512
WARMUP_RATIO: float            = 0.05
WEIGHT_DECAY: float            = 0.01
LOGGING_STEPS: int             = 25
SAVE_STRATEGY: str             = "epoch"
EVAL_STRATEGY: str             = "epoch"

# ──────────────────────────────────────────────
# Evaluation output files
# ──────────────────────────────────────────────
BASELINE_PREDICTIONS_PATH  = EVALUATION_DIR / "baseline_predictions.csv"
ZEROSHOT_PREDICTIONS_PATH  = EVALUATION_DIR / "zeroshot_predictions.csv"
LORA_PREDICTIONS_PATH      = EVALUATION_DIR / "lora_predictions.csv"
ERROR_ANALYSIS_PATH        = EVALUATION_DIR / "error_analysis.csv"
MODEL_COMPARISON_PATH      = EVALUATION_DIR / "model_comparison.csv"
METRICS_PATH               = EVALUATION_DIR / "metrics.json"

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"
