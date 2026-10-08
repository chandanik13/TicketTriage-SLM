import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from app.app import analyze_ticket, compare_all_models, build_ui, MODEL_CHOICES

@pytest.fixture
def mock_manager():
    with patch('app.app.manager') as mock:
        yield mock

def test_app_imports():
    """Verify that UI builds without throwing exceptions."""
    app = build_ui()
    assert app is not None

def test_model_selection_mapping():
    assert "TF-IDF + Logistic Regression" in MODEL_CHOICES
    assert MODEL_CHOICES["TF-IDF + Logistic Regression"] == "baseline"
    assert "Qwen2.5-0.5B-Instruct Zero-Shot" in MODEL_CHOICES
    assert MODEL_CHOICES["Qwen2.5-0.5B-Instruct Zero-Shot"] == "qwen_zero_shot"

def test_input_validation(mock_manager):
    res_json, latency = analyze_ticket("", "TF-IDF + Logistic Regression")
    assert "Error" in res_json
    assert "Please enter a ticket" in res_json["Error"]
    assert latency == "N/A"
    
    comp_df = compare_all_models("   ")
    assert "Error" in comp_df.columns
    assert "Please enter a ticket" in comp_df.iloc[0]["Error"]

def test_unavailable_model_handling(mock_manager):
    # Mock manager to return an unavailable status
    mock_manager.predict.return_value = {
        "status": "unavailable",
        "error_message": "Model not found."
    }
    
    res_json, latency = analyze_ticket("Help me", "Qwen2.5-0.5B-Instruct + LoRA")
    assert "Status" in res_json
    assert "training/integration pending" in res_json["Status"]
    assert latency == "N/A"
    
    comp_df = compare_all_models("Help me")
    assert len(comp_df) == 3
    for idx, row in comp_df.iterrows():
        assert "training/integration pending" in row["Category"]

def test_output_formatting_helpers(mock_manager):
    mock_manager.predict.return_value = {
        "status": "success",
        "category": "billing",
        "priority": "high",
        "sentiment": "negative",
        "language": "english",
        "resolution": "process_refund",
        "json_valid": True,
        "latency_seconds": 1.25
    }
    
    res_json, latency = analyze_ticket("I need a refund", "Qwen2.5-0.5B-Instruct Zero-Shot")
    
    assert res_json["Category"] == "billing"
    assert res_json["Priority"] == "high"
    assert res_json["JSON Valid"] is True
    assert latency == "1.25 seconds"
    
    comp_df = compare_all_models("I need a refund")
    assert len(comp_df) == 3
    assert comp_df.iloc[0]["Model"] == "TF-IDF + Logistic Regression"
    assert comp_df.iloc[0]["Category"] == "billing"
    assert comp_df.iloc[0]["Latency (s)"] == 1.25
