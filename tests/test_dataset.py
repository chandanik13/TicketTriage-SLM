import pytest
import pandas as pd
import json
from src.dataset import validate_dataframe, format_training_example, create_formatted_dataset

def test_validate_dataframe_valid():
    df = pd.DataFrame([
        {
            "ticket_id": "TKT-001",
            "text": "Please explain this charge.",
            "category": "billing",
            "priority": "low",
            "sentiment": "neutral",
            "language": "english",
            "resolution": "explain_charge"
        }
    ])
    assert validate_dataframe(df) is True

def test_validate_dataframe_missing_columns():
    df = pd.DataFrame([
        {
            "ticket_id": "TKT-001",
            "text": "Missing everything else"
        }
    ])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe(df)

def test_validate_dataframe_missing_values():
    df = pd.DataFrame([
        {
            "ticket_id": "TKT-001",
            "text": None,
            "category": "billing",
            "priority": "low",
            "sentiment": "neutral",
            "language": "english",
            "resolution": "explain_charge"
        }
    ])
    with pytest.raises(ValueError, match="missing values"):
        validate_dataframe(df)

def test_validate_dataframe_invalid_labels():
    df = pd.DataFrame([
        {
            "ticket_id": "TKT-001",
            "text": "Help",
            "category": "billing",
            "priority": "SUPER-URGENT", # Invalid
            "sentiment": "neutral",
            "language": "english",
            "resolution": "explain_charge"
        }
    ])
    with pytest.raises(ValueError, match="Invalid labels found in field 'priority'"):
        validate_dataframe(df)

def test_format_training_example():
    row = pd.Series({
        "ticket_id": "TKT-001",
        "text": "Please explain this charge.",
        "category": "billing",
        "priority": "low",
        "sentiment": "neutral",
        "language": "english",
        "resolution": "explain_charge"
    })
    formatted = format_training_example(row)
    
    assert "messages" in formatted
    messages = formatted["messages"]
    assert len(messages) == 3
    
    # Check system prompt
    assert messages[0]["role"] == "system"
    
    # Check user prompt
    assert messages[1]["role"] == "user"
    assert "Analyze the following customer support ticket" in messages[1]["content"]
    assert "Please explain this charge." in messages[1]["content"]
    
    # Check assistant response formatting (valid JSON)
    assert messages[2]["role"] == "assistant"
    response_json = json.loads(messages[2]["content"])
    assert response_json["category"] == "billing"
    assert response_json["priority"] == "low"
    assert response_json["sentiment"] == "neutral"
    assert response_json["language"] == "english"
    assert response_json["resolution"] == "explain_charge"

def test_create_formatted_dataset():
    df = pd.DataFrame([
        {
            "ticket_id": "TKT-001",
            "text": "Please explain this charge.",
            "category": "billing",
            "priority": "low",
            "sentiment": "neutral",
            "language": "english",
            "resolution": "explain_charge"
        }
    ])
    dataset = create_formatted_dataset(df)
    assert len(dataset) == 1
    assert "messages" in dataset[0]
