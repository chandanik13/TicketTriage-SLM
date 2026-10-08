import pytest
from src.zero_shot import extract_and_parse_json, validate_labels, REQUIRED_FIELDS

def test_extract_and_parse_json_perfect():
    raw = '{"category": "billing", "priority": "high"}'
    parsed, valid = extract_and_parse_json(raw)
    assert valid is True
    assert parsed == {"category": "billing", "priority": "high"}

def test_extract_and_parse_json_with_markdown():
    raw = '''Here is your classification:
```json
{
    "category": "technical",
    "priority": "urgent"
}
```
Hope this helps!'''
    parsed, valid = extract_and_parse_json(raw)
    assert valid is True
    assert parsed == {"category": "technical", "priority": "urgent"}

def test_extract_and_parse_json_surrounded_text():
    raw = 'I think it is {"category": "refund"} based on the text.'
    parsed, valid = extract_and_parse_json(raw)
    assert valid is True
    assert parsed == {"category": "refund"}

def test_extract_and_parse_json_malformed():
    raw = '{"category": "billing", "priority": "high"' # Missing closing brace
    parsed, valid = extract_and_parse_json(raw)
    assert valid is False
    assert parsed is None

def test_validate_labels_all_valid():
    parsed = {
        "category": "billing",
        "priority": "low",
        "sentiment": "neutral",
        "language": "english",
        "resolution": "explain_charge"
    }
    is_valid, invalid_fields = validate_labels(parsed)
    assert is_valid is True
    assert len(invalid_fields) == 0

def test_validate_labels_missing_fields():
    parsed = {
        "category": "billing",
        "sentiment": "neutral"
    }
    is_valid, invalid_fields = validate_labels(parsed)
    assert is_valid is False
    assert set(invalid_fields) == {"priority", "language", "resolution"}

def test_validate_labels_invalid_values():
    parsed = {
        "category": "billing",
        "priority": "super-urgent", # invalid
        "sentiment": "neutral",
        "language": "french", # invalid
        "resolution": "explain_charge"
    }
    is_valid, invalid_fields = validate_labels(parsed)
    assert is_valid is False
    assert set(invalid_fields) == {"priority", "language"}

def test_validate_labels_empty():
    parsed = {}
    is_valid, invalid_fields = validate_labels(parsed)
    assert is_valid is False
    assert set(invalid_fields) == set(REQUIRED_FIELDS)
