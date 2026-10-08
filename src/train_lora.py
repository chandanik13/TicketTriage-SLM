import os
import argparse
import logging
import random
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-0.5B-Instruct using LoRA")
    parser.add_argument("--train-data", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--val-data", type=str, required=True, help="Path to validation.csv")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="models/qwen_lora", help="Where to save the LoRA adapter")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def verify_lora_targets(model, target_modules: list[str]):
    """Verifies that the target modules actually exist in the model architecture."""
    model_modules = [name for name, _ in model.named_modules()]
    found_targets = set()
    
    for name in model_modules:
        for target in target_modules:
            if target in name:
                found_targets.add(target)
                
    if len(found_targets) == 0:
        raise RuntimeError("No matching LoRA target modules found in the model architecture! Aborting training.")
        
    logger.info(f"Verified LoRA target modules present: {list(found_targets)}")

def train():
    args = parse_args()
    set_seed(args.seed)
    
    # Deferred imports to prevent heavy downloads/loading when just parsing args or testing
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer
    from src.dataset import create_formatted_dataset
    from datasets import Dataset
    import torch
    
    logger.info("Loading datasets...")
    train_df = pd.read_csv(args.train_data)
    val_df = pd.read_csv(args.val_data)
    
    train_formatted = create_formatted_dataset(train_df)
    val_formatted = create_formatted_dataset(val_df)
    
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA is NOT available. LoRA training will be significantly slower on CPU.")
        dtype = torch.float32
        fp16 = False
        bf16 = False
    else:
        # Avoid enabling both bf16 and fp16 simultaneously
        bf16 = torch.cuda.is_bf16_supported()
        fp16 = not bf16
        dtype = torch.bfloat16 if bf16 else torch.float16
        logger.info(f"CUDA available. Using mixed precision: bf16={bf16}, fp16={fp16}")

    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto" if device == "cuda" else {"": "cpu"},
        torch_dtype=dtype,
        trust_remote_code=True
    )
    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    verify_lora_targets(model, target_modules)
    
    peft_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=True if device == "cuda" else False,
        seed=args.seed,
        report_to="none"
    )
    
    logger.info("Initializing SFTTrainer...")
    
    def formatting_prompts_func(example):
        # Trl expects a list of text strings when using formatting_func if not using chat templates natively
        return [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in example["messages"]]
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving LoRA adapter to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete.")

if __name__ == "__main__":
    logger.warning("LoRA training script invoked. Model execution is strictly prohibited by current rules. Exiting.")
    # To run training, comment out the sys.exit(0) and call train()
    import sys
    sys.exit(0)
    # train()
