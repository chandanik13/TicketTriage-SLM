import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

VALID_LABELS = {
    "category": {"billing", "technical", "account", "shipping", "refund", "subscription"},
    "priority": {"low", "medium", "high", "urgent"},
    "sentiment": {"positive", "neutral", "negative"},
    "language": {"english"},
    "resolution": {
        "explain_charge", "update_payment_method", "dispute_charge", "process_refund",
        "escalate_to_agent", "request_more_info", "deny_refund_policy", "check_refund_status",
        "track_order", "report_lost_package", "change_shipping_address", "cancel_subscription",
        "upgrade_plan", "downgrade_plan", "reset_password", "unlock_account", "update_profile",
        "troubleshoot_app", "report_bug", "request_feature"
    }
}

REQUIRED_COLUMNS = ["ticket_id", "text", "category", "priority", "sentiment", "language", "resolution"]

def validate_dataframe(df: pd.DataFrame):
    """Validates that all required columns exist and labels are perfectly strictly valid."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")
        
    for field, valid_set in VALID_LABELS.items():
        invalid_mask = ~df[field].str.lower().str.strip().isin(valid_set)
        if invalid_mask.any():
            invalid_examples = df[invalid_mask][field].unique()
            raise ValueError(f"Invalid labels found in field '{field}': {invalid_examples}")
            
    return True

def format_training_example(row) -> dict:
    """Formats a single DataFrame row into an instruction-following chat structure."""
    user_prompt = f"""Analyze the following customer support ticket and classify it.

Ticket:
{row['text']}

Return a JSON object containing: category, priority, sentiment, language, resolution."""

    assistant_response = json.dumps({
        "category": row["category"].strip().lower(),
        "priority": row["priority"].strip().lower(),
        "sentiment": row["sentiment"].strip().lower(),
        "language": row["language"].strip().lower(),
        "resolution": row["resolution"].strip().lower()
    }, indent=4)
    
    return {
        "messages": [
            {"role": "system", "content": "You are a customer support ticket triage classifier. Output only valid JSON."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }

def create_formatted_dataset(df: pd.DataFrame) -> list[dict]:
    """Validates a DataFrame and converts it into a list of formatted conversation examples."""
    validate_dataframe(df)
    return [format_training_example(row) for _, row in df.iterrows()]
