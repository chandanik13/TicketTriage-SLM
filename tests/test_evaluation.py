import pytest
import pandas as pd
import numpy as np
from src.evaluation import TicketEvaluator
import os

@pytest.fixture
def evaluator(tmp_path):
    return TicketEvaluator(output_dir=tmp_path)

def test_perfect_predictions(evaluator):
    df = pd.DataFrame({
        "ticket_id": ["T1", "T2"],
        "text": ["Hello", "World"],
        "true_category": ["billing", "technical"],
        "pred_category": ["billing", "technical"],
        "true_priority": ["low", "high"],
        "pred_priority": ["low", "high"],
        "true_sentiment": ["neutral", "negative"],
        "pred_sentiment": ["neutral", "negative"],
        "true_language": ["english", "english"],
        "pred_language": ["english", "english"],
        "true_resolution": ["explain_charge", "troubleshoot_app"],
        "pred_resolution": ["explain_charge", "troubleshoot_app"],
        "json_valid": [True, True],
        "latency_seconds": [1.0, 1.5]
    })
    
    evaluator.evaluate_model("qwen_zero_shot", df, is_generative=True)
    res = evaluator.results[0]
    
    assert res["Category Accuracy"] == 1.0
    assert res["Average Field Accuracy"] == 1.0
    assert res["Exact Match Accuracy"] == 1.0
    assert res["JSON Validity Rate"] == 1.0
    assert "Average Latency" in res
    assert res["Average Latency"] == 1.25

def test_partially_incorrect_predictions(evaluator):
    df = pd.DataFrame({
        "ticket_id": ["T1"],
        "text": ["Hello"],
        "true_category": ["billing"],
        "pred_category": ["billing"],  # correct
        "true_priority": ["low"],
        "pred_priority": ["high"],     # incorrect
        "true_sentiment": ["neutral"],
        "pred_sentiment": ["neutral"], # correct
        "true_language": ["english"],
        "pred_language": ["english"],  # correct
        "true_resolution": ["explain_charge"],
        "pred_resolution": ["process_refund"], # incorrect
        "json_valid": [True],
        "latency_seconds": [1.0]
    })
    
    evaluator.evaluate_model("qwen_lora", df, is_generative=True)
    res = evaluator.results[0]
    
    assert res["Category Accuracy"] == 1.0
    assert res["Priority Accuracy"] == 0.0
    assert res["Average Field Accuracy"] == 0.6  # 3/5 fields correct
    assert res["Exact Match Accuracy"] == 0.0
    
    # Error analysis checks
    evaluator.save_error_analysis()
    err_df = evaluator.all_errors[0]
    assert len(err_df) == 1
    assert err_df.iloc[0]["correct_field_count"] == 3
    assert err_df.iloc[0]["incorrect_fields"] == "priority|resolution"

def test_completely_incorrect_predictions(evaluator):
    df = pd.DataFrame({
        "ticket_id": ["T1"],
        "text": ["Hello"],
        "true_category": ["billing"],
        "pred_category": ["technical"],
        "true_priority": ["low"],
        "pred_priority": ["high"],
        "true_sentiment": ["neutral"],
        "pred_sentiment": ["negative"],
        "true_language": ["english"],
        "pred_language": ["spanish"],
        "true_resolution": ["explain_charge"],
        "pred_resolution": ["troubleshoot_app"],
        "json_valid": [True],
        "latency_seconds": [1.0]
    })
    
    evaluator.evaluate_model("qwen_lora", df, is_generative=True)
    res = evaluator.results[0]
    
    assert res["Average Field Accuracy"] == 0.0
    assert res["Exact Match Accuracy"] == 0.0

def test_baseline_eval_no_json(evaluator):
    df = pd.DataFrame({
        "ticket_id": ["T1"],
        "text": ["Hello"],
        "true_category": ["billing"],
        "pred_category": ["billing"],
        "true_priority": ["low"],
        "pred_priority": ["low"],
        "true_sentiment": ["neutral"],
        "pred_sentiment": ["neutral"],
        "true_language": ["english"],
        "pred_language": ["english"],
        "true_resolution": ["explain_charge"],
        "pred_resolution": ["explain_charge"]
        # Intentionally missing json_valid and latency
    })
    
    evaluator.evaluate_model("baseline", df, is_generative=False)
    res = evaluator.results[0]
    
    assert res["JSON Validity Rate"] == "N/A"
    assert res["Average Latency"] == "N/A"
    assert res["Invalid Field Rate"] == "N/A"

def test_invalid_field_detection(evaluator):
    df = pd.DataFrame({
        "ticket_id": ["T1", "T2"],
        "text": ["Hello", "World"],
        "true_category": ["billing", "technical"],
        "pred_category": ["billing", np.nan], # missing/invalid prediction for T2
        "true_priority": ["low", "high"],
        "pred_priority": ["low", "high"],
        "true_sentiment": ["neutral", "negative"],
        "pred_sentiment": ["neutral", "negative"],
        "true_language": ["english", "english"],
        "pred_language": ["english", "english"],
        "true_resolution": ["explain_charge", "troubleshoot_app"],
        "pred_resolution": ["explain_charge", "troubleshoot_app"],
    })
    
    evaluator.evaluate_model("qwen_zero_shot", df, is_generative=True)
    res = evaluator.results[0]
    
    assert "category (1/2)" in res["Invalid Field Rate"]

def test_comparison_table_generation(evaluator):
    df = pd.DataFrame({
        "ticket_id": ["T1"],
        "text": ["Hello"],
        "true_category": ["billing"],
        "pred_category": ["billing"],
        "true_priority": ["low"],
        "pred_priority": ["low"],
        "true_sentiment": ["neutral"],
        "pred_sentiment": ["neutral"],
        "true_language": ["english"],
        "pred_language": ["english"],
        "true_resolution": ["explain_charge"],
        "pred_resolution": ["explain_charge"]
    })
    evaluator.evaluate_model("model_1", df)
    evaluator.evaluate_model("model_2", df)
    
    comp_df = evaluator.generate_comparison_table()
    assert len(comp_df) == 2
    assert list(comp_df["Model"]) == ["model_1", "model_2"]
    
    # Check that artifact is saved
    assert (evaluator.output_dir / "model_comparison.csv").exists()
