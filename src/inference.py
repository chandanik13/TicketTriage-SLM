import os
import time
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """Handles lazy loading and unified inference for all models."""
    
    def __init__(self):
        self._models = {
            "baseline": None,
            "qwen_zero_shot": None,
            "qwen_lora": None
        }
        self.baseline_dir = Path("models/baseline")
        self.lora_dir = Path("models/qwen_lora")

    def _load_baseline(self):
        if self._models["baseline"] is None:
            vec_path = self.baseline_dir / "tfidf_vectorizer.joblib"
            mod_path = self.baseline_dir / "logistic_regression.joblib"
            
            if not vec_path.exists() or not mod_path.exists():
                raise FileNotFoundError("Baseline artifacts not found.")
                
            logger.info("Lazy loading TF-IDF Baseline model...")
            self._models["baseline"] = {
                "vectorizer": joblib.load(vec_path),
                "model": joblib.load(mod_path)
            }
        return self._models["baseline"]

    def _load_qwen_zero_shot(self):
        if self._models["qwen_zero_shot"] is None:
            logger.info("Lazy loading Qwen Zero-Shot model...")
            from src.zero_shot import load_model
            # Note: This will download/load the heavy LLM
            tokenizer, model = load_model()
            self._models["qwen_zero_shot"] = {"tokenizer": tokenizer, "model": model}
        return self._models["qwen_zero_shot"]

    def _load_qwen_lora(self):
        if self._models["qwen_lora"] is None:
            if not (self.lora_dir / "adapter_config.json").exists():
                raise FileNotFoundError("LoRA adapter not found. Training pending.")
                
            logger.info("Lazy loading Qwen LoRA model...")
            from src.zero_shot import load_model
            from peft import PeftModel
            
            # Load base
            tokenizer, base_model = load_model()
            # Load LoRA on top
            logger.info("Applying LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, self.lora_dir)
            self._models["qwen_lora"] = {"tokenizer": tokenizer, "model": model}
        return self._models["qwen_lora"]

    def predict(self, model_id: str, text: str) -> dict:
        """Unified prediction interface."""
        start_time = time.time()
        
        try:
            if model_id == "baseline":
                artifacts = self._load_baseline()
                # TF-IDF inference
                vec = artifacts["vectorizer"].transform([text])
                pred_cat = artifacts["model"].predict(vec)[0]
                
                return {
                    "category": pred_cat,
                    "priority": "N/A (Baseline)",
                    "sentiment": "N/A (Baseline)",
                    "language": "N/A (Baseline)",
                    "resolution": "N/A (Baseline)",
                    "json_valid": "N/A",
                    "latency_seconds": round(time.time() - start_time, 3),
                    "raw_output": pred_cat,
                    "status": "success"
                }
                
            elif model_id == "qwen_zero_shot":
                artifacts = self._load_qwen_zero_shot()
                from src.zero_shot import predict_ticket
                res = predict_ticket(text, artifacts["tokenizer"], artifacts["model"])
                res["status"] = "success"
                return res
                
            elif model_id == "qwen_lora":
                artifacts = self._load_qwen_lora()
                from src.zero_shot import predict_ticket
                res = predict_ticket(text, artifacts["tokenizer"], artifacts["model"])
                res["status"] = "success"
                return res
                
            else:
                return {"status": "error", "error_message": f"Unknown model ID: {model_id}"}
                
        except FileNotFoundError as e:
            return {"status": "unavailable", "error_message": str(e)}
        except Exception as e:
            logger.error(f"Inference failed for {model_id}: {str(e)}")
            return {"status": "error", "error_message": "Inference failed. See application logs for details."}
