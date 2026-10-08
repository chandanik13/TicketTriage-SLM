import pytest
import argparse
from unittest.mock import MagicMock
from src.train_lora import parse_args, verify_lora_targets

def test_parse_args(monkeypatch):
    test_args = [
        "train_lora.py",
        "--train-data", "data/processed/train.csv",
        "--val-data", "data/processed/val.csv",
        "--r", "16",
        "--learning-rate", "1e-4",
        "--epochs", "5.0"
    ]
    monkeypatch.setattr("sys.argv", test_args)
    args = parse_args()
    
    assert args.train_data == "data/processed/train.csv"
    assert args.val_data == "data/processed/val.csv"
    assert args.r == 16
    assert args.learning_rate == 1e-4
    assert args.epochs == 5.0
    
    # Defaults
    assert args.lora_alpha == 16
    assert args.lora_dropout == 0.05
    assert args.batch_size == 4
    assert args.max_seq_length == 512
    assert args.seed == 42

def test_verify_lora_targets_success():
    # Mock a model with named_modules
    mock_model = MagicMock()
    mock_model.named_modules.return_value = [
        ("model.layers.0.mlp.up_proj", MagicMock()),
        ("model.layers.0.self_attn.q_proj", MagicMock()),
        ("model.layers.0.self_attn.k_proj", MagicMock()),
        ("model.layers.0.self_attn.v_proj", MagicMock()),
        ("model.layers.0.self_attn.o_proj", MagicMock()),
    ]
    
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    # Should not raise an exception
    verify_lora_targets(mock_model, targets)

def test_verify_lora_targets_failure():
    # Mock a model that has completely different module names
    mock_model = MagicMock()
    mock_model.named_modules.return_value = [
        ("model.layers.0.attention.query", MagicMock()),
        ("model.layers.0.attention.key", MagicMock()),
        ("model.layers.0.attention.value", MagicMock()),
        ("model.layers.0.ffn.dense1", MagicMock()),
    ]
    
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    
    with pytest.raises(RuntimeError, match="No matching LoRA target modules found"):
        verify_lora_targets(mock_model, targets)
