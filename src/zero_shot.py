import os
import json
import logging
import time
import re
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Valid Labels
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

REQUIRED_FIELDS = ["category", "priority", "sentiment", "language", "resolution"]

SYSTEM_PROMPT = """You are a highly precise customer support ticket triage classifier.
Analyze the semantic meaning of the provided customer support ticket.
You must classify the ticket into exactly one value for each of the following fields.

ALLOWED LABELS:
- category: billing, technical, account, shipping, refund, subscription
- priority: low, medium, high, urgent
- sentiment: positive, neutral, negative
- language: english
- resolution: explain_charge, update_payment_method, dispute_charge, process_refund, escalate_to_agent, request_more_info, deny_refund_policy, check_refund_status, track_order, report_lost_package, change_shipping_address, cancel_subscription, upgrade_plan, downgrade_plan, reset_password, unlock_account, update_profile, troubleshoot_app, report_bug, request_feature

OUTPUT FORMAT:
You must return ONLY valid JSON.
Do not add explanations.
Do not use Markdown formatting or ```json fences.
Do not invent labels that are not in the allowed list.

EXPECTED JSON STRUCTURE:
{
    "category": "...",
    "priority": "...",
    "sentiment": "...",
    "language": "...",
    "resolution": "..."
}"""

def load_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """Loads the tokenizer and model, automatically using CUDA if available."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    
    logger.info(f"Loading model: {model_name} on {device} (dtype: {dtype})")
    
    # Use device_map="auto" if CUDA is available, else force CPU
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=dtype,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map={"": "cpu"}, 
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
    model.eval()
    logger.info("Model loaded successfully.")
    return tokenizer, model

def extract_and_parse_json(raw_output: str) -> tuple[dict | None, bool]:
    """Extracts JSON from text, handling markdown fences and surrounding text."""
    # Attempt to find JSON block
    json_match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
    
    if not json_match:
        return None, False
        
    json_str = json_match.group(0)
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            return parsed, True
        return None, False
    except json.JSONDecodeError:
        return None, False

def validate_labels(parsed_json: dict) -> tuple[bool, list[str]]:
    """Validates that all required fields exist and contain allowed labels."""
    invalid_fields = []
    
    if not parsed_json:
        return False, REQUIRED_FIELDS
        
    for field in REQUIRED_FIELDS:
        if field not in parsed_json:
            invalid_fields.append(field)
            continue
            
        val = str(parsed_json[field]).strip().lower()
        if val not in VALID_LABELS[field]:
            invalid_fields.append(field)
            
    is_valid = len(invalid_fields) == 0
    return is_valid, invalid_fields

def predict_ticket(text: str, tokenizer, model, generation_config=None) -> dict:
    """Runs zero-shot inference on a single ticket."""
    import torch
    start_time = time.time()
    
    if generation_config is None:
        # Deterministic generation for classification
        generation_config = {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    raw_output = ""
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
            
        # Extract only the generated text (ignoring prompt)
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_length:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
    except Exception as e:
        logger.error(f"Inference exception: {str(e)}")
        return {
            "category": None, "priority": None, "sentiment": None, "language": None, "resolution": None,
            "json_valid": False, "invalid_fields": REQUIRED_FIELDS,
            "latency_seconds": time.time() - start_time,
            "raw_output": f"ERROR: {str(e)}"
        }

    # Parse and validate
    parsed_json, is_json_valid = extract_and_parse_json(raw_output)
    
    if not is_json_valid or parsed_json is None:
        return {
            "category": None, "priority": None, "sentiment": None, "language": None, "resolution": None,
            "json_valid": False, "invalid_fields": REQUIRED_FIELDS,
            "latency_seconds": time.time() - start_time,
            "raw_output": raw_output
        }
        
    is_labels_valid, invalid_fields = validate_labels(parsed_json)
    
    return {
        "category": parsed_json.get("category"),
        "priority": parsed_json.get("priority"),
        "sentiment": parsed_json.get("sentiment"),
        "language": parsed_json.get("language"),
        "resolution": parsed_json.get("resolution"),
        "json_valid": is_labels_valid,  # Overall validity (valid JSON AND valid labels)
        "invalid_fields": invalid_fields,
        "latency_seconds": round(time.time() - start_time, 3),
        "raw_output": raw_output
    }

def predict_batch(texts: list[str], tokenizer, model) -> list[dict]:
    """Runs zero-shot inference on a batch of tickets safely."""
    results = []
    for text in texts:
        res = predict_ticket(text, tokenizer, model)
        results.append(res)
    return results

if __name__ == "__main__":
    logger.info("Running minimal CLI smoke test...")
    # Intentionally wrapped in a confirmation prompt or skipped automatically for CI/Agent workflows.
    # To run, execute: python -m src.zero_shot
    
    print("WARNING: This smoke test requires downloading/loading Qwen2.5-0.5B-Instruct.")
    # We exit immediately to avoid uncontrolled downloads during agent testing, 
    # but the code below serves as the functional CLI test when manually run.
    print("Smoke test auto-aborted to prevent uncontrolled downloads.")
    import sys
    sys.exit(0)
    
    tokenizer, model = load_model()
    
    test_ticket = "I have been overcharged on my last bill by $20. I want this fixed immediately or I am canceling."
    print(f"\n[Test Ticket]: {test_ticket}")
    
    result = predict_ticket(test_ticket, tokenizer, model)
    print("\n[Prediction Result]:")
    print(json.dumps(result, indent=2))
