import os
import sys
import logging
import pandas as pd

# Add project root to path if running directly from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We defer heavy imports until inside the main function to keep the script statically verifiable
# without triggering CUDA initialization or heavy downloads during import.
logger = logging.getLogger(__name__)

def run_smoke_test():
    print("=" * 60)
    print("DEVELOPMENT SMOKE TEST — NOT RESEARCH RESULTS")
    print("=" * 60)
    
    import torch
    from src.zero_shot import load_model, predict_ticket
    
    # 1. Print Environment Info
    cuda_available = torch.cuda.is_available()
    print(f"\n[Environment]")
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    # 2. Load Model
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n[Model]")
    print(f"Identifier: {model_id}")
    print("Loading model (this may take a moment)...")
    
    tokenizer, model = load_model(model_name=model_id)
    
    # 3. Load Data
    test_csv_path = "data/processed/test.csv"
    if not os.path.exists(test_csv_path):
        print(f"ERROR: Could not find {test_csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(test_csv_path)
    
    # Read EXACTLY 5 tickets
    tickets_to_test = 5
    sample_df = df.head(tickets_to_test)
    
    print(f"\n[Data]")
    print(f"Source: {test_csv_path}")
    print(f"Number of tickets to test: {len(sample_df)}")
    
    # 4. Run Inference
    print("\n[Inference]")
    for idx, row in sample_df.iterrows():
        ticket_id = row.get("ticket_id", f"TKT-{idx}")
        text = row.get("text", "")
        
        print(f"\n--- Ticket: {ticket_id} ---")
        print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        
        result = predict_ticket(text, tokenizer, model)
        
        print("\nPredictions:")
        print(f"  Category:   {result.get('category')}")
        print(f"  Priority:   {result.get('priority')}")
        print(f"  Sentiment:  {result.get('sentiment')}")
        print(f"  Language:   {result.get('language')}")
        print(f"  Resolution: {result.get('resolution')}")
        
        print("\nDiagnostics:")
        print(f"  JSON Valid:      {result.get('json_valid')}")
        print(f"  Invalid Fields:  {result.get('invalid_fields')}")
        print(f"  Latency:         {result.get('latency_seconds')} seconds")
        print(f"  Raw Output:      {result.get('raw_output')}")
        
    print("\n" + "=" * 60)
    print("DEVELOPMENT SMOKE TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("\nWARNING: This smoke test requires downloading/loading the Qwen LLM.")
    print("Model execution is strictly prohibited in the current phase.")
    print("Exiting immediately to prevent unauthorized downloads and GPU usage.")
    
    # Uncomment to actually run in Colab
    # run_smoke_test()
    sys.exit(0)
