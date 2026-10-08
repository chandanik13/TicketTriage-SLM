import os
import sys
import json
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

logger = logging.getLogger(__name__)

# --- UNIFIED EVALUATION FRAMEWORK ---

class TicketEvaluator:
    def __init__(self, output_dir="evaluation"):
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"
        self.predictions_dir = self.output_dir / "predictions"
        self.cm_dir = self.output_dir / "confusion_matrices"
        
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.cm_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.all_errors = []
        
        self.fields = ["category", "priority", "sentiment", "language", "resolution"]

    def _save_confusion_matrix(self, y_true, y_pred, model_name, field):
        """Generate and save confusion matrix."""
        # Get unique labels sorted to ensure consistent matrix rendering
        labels = sorted(list(set(y_true.dropna()).union(set(y_pred.dropna()))))
        if not labels:
            return
            
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'{model_name} Confusion Matrix: {field.capitalize()}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        out_path = self.cm_dir / f"{model_name}_{field}_confusion_matrix.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

    def evaluate_model(self, model_name: str, df: pd.DataFrame, is_generative: bool = True):
        """
        Evaluate a single model on all 5 fields.
        df must contain: ticket_id, text, true_{field}, pred_{field}.
        For generative models, df should optionally contain 'json_valid' and 'latency_seconds'.
        """
        logger.info(f"Evaluating model: {model_name}")
        metrics_summary = {"Model": model_name}
        field_accuracies = []
        
        # Track valid predictions for generative invalid field rate
        invalid_counts = {f: 0 for f in self.fields}
        
        for field in self.fields:
            true_col = f"true_{field}"
            pred_col = f"pred_{field}"
            
            if true_col not in df.columns or pred_col not in df.columns:
                logger.warning(f"Missing columns for field {field} in {model_name} predictions.")
                metrics_summary[f"{field.capitalize()} Accuracy"] = None
                if field == "category":
                    metrics_summary[f"{field.capitalize()} Macro F1"] = None
                    metrics_summary[f"{field.capitalize()} Weighted F1"] = None
                continue
                
            y_true = df[true_col].astype(str).str.lower().str.strip()
            y_pred = df[pred_col].astype(str).str.lower().str.strip()
            
            # Replace nan/none strings with actual NaNs for evaluation logic
            y_true = y_true.replace(['nan', 'none', ''], np.nan)
            y_pred = y_pred.replace(['nan', 'none', ''], np.nan)
            
            # Track invalid/missing predictions
            invalid_counts[field] = y_pred.isna().sum()
            
            # Fill NaNs with a dummy label to ensure sklearn can compute metrics
            y_true_clean = y_true.fillna("missing_true")
            y_pred_clean = y_pred.fillna("missing_pred")
            
            # Field metrics
            acc = accuracy_score(y_true_clean, y_pred_clean)
            macro_f1 = f1_score(y_true_clean, y_pred_clean, average='macro', zero_division=0)
            weighted_f1 = f1_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0)
            
            field_accuracies.append(acc)
            metrics_summary[f"{field.capitalize()} Accuracy"] = acc
            
            if field == "category":
                metrics_summary[f"{field.capitalize()} Macro F1"] = macro_f1
                metrics_summary[f"{field.capitalize()} Weighted F1"] = weighted_f1
                
            # Classification report
            report = classification_report(y_true_clean, y_pred_clean, output_dict=True, zero_division=0)
            report_path = self.reports_dir / f"{model_name}_{field}_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
                
            # Confusion matrix (for category, priority, sentiment, resolution)
            if field in ["category", "priority", "sentiment", "resolution"]:
                self._save_confusion_matrix(y_true_clean, y_pred_clean, model_name, field)

        # Average field accuracy
        if field_accuracies:
            metrics_summary["Average Field Accuracy"] = np.mean(field_accuracies)
        else:
            metrics_summary["Average Field Accuracy"] = None

        # Exact Match Accuracy (all 5 correct simultaneously)
        exact_match_series = pd.Series([True] * len(df))
        for field in self.fields:
            if f"true_{field}" in df.columns and f"pred_{field}" in df.columns:
                exact_match_series &= (
                    df[f"true_{field}"].astype(str).str.lower().str.strip() == 
                    df[f"pred_{field}"].astype(str).str.lower().str.strip()
                )
        metrics_summary["Exact Match Accuracy"] = exact_match_series.mean()

        # Generative Metrics (JSON Validity & Invalid Fields)
        if is_generative:
            if "json_valid" in df.columns:
                metrics_summary["JSON Validity Rate"] = df["json_valid"].mean()
            else:
                metrics_summary["JSON Validity Rate"] = None
                
            # Highest failing field
            worst_field = max(invalid_counts, key=invalid_counts.get)
            worst_count = invalid_counts[worst_field]
            metrics_summary["Invalid Field Rate"] = f"{worst_field} ({worst_count}/{len(df)})"
        else:
            metrics_summary["JSON Validity Rate"] = "N/A"
            metrics_summary["Invalid Field Rate"] = "N/A"

        # Latency Metrics
        if "latency_seconds" in df.columns and not df["latency_seconds"].isna().all():
            metrics_summary["Average Latency"] = df["latency_seconds"].mean()
            metrics_summary["Median Latency"] = df["latency_seconds"].median()
            metrics_summary["P95 Latency"] = df["latency_seconds"].quantile(0.95)
        else:
            metrics_summary["Average Latency"] = "N/A"
            metrics_summary["Median Latency"] = "N/A"
            metrics_summary["P95 Latency"] = "N/A"
            
        self.results.append(metrics_summary)
        
        # Save Predictions
        pred_path = self.predictions_dir / f"{model_name}_predictions.csv"
        df.to_csv(pred_path, index=False)
        
        # Error Analysis Preparation
        incorrect_mask = ~exact_match_series
        if incorrect_mask.any():
            errors_df = df[incorrect_mask].copy()
            errors_df["model"] = model_name
            
            # Determine correct counts and incorrect fields for each row
            correct_counts = []
            incorrect_fields_list = []
            
            for _, row in errors_df.iterrows():
                correct = 0
                incorrect = []
                for field in self.fields:
                    t_col, p_col = f"true_{field}", f"pred_{field}"
                    if t_col in row and p_col in row:
                        if str(row[t_col]).strip().lower() == str(row[p_col]).strip().lower():
                            correct += 1
                        else:
                            incorrect.append(field)
                correct_counts.append(correct)
                incorrect_fields_list.append("|".join(incorrect))
                
            errors_df["correct_field_count"] = correct_counts
            errors_df["incorrect_fields"] = incorrect_fields_list
            
            # Sub-select relevant columns for final error analysis CSV
            keep_cols = ["model", "ticket_id", "text"]
            for f in self.fields:
                if f"true_{f}" in df.columns: keep_cols.append(f"true_{f}")
                if f"pred_{f}" in df.columns: keep_cols.append(f"pred_{f}")
            keep_cols.extend(["correct_field_count", "incorrect_fields"])
            
            # Only keep columns that actually exist
            keep_cols = [c for c in keep_cols if c in errors_df.columns]
            self.all_errors.append(errors_df[keep_cols])

    def generate_comparison_table(self):
        """Build and save the unified comparison table."""
        if not self.results:
            logger.warning("No results to compare.")
            return None
            
        comp_df = pd.DataFrame(self.results)
        # Ensure ordered columns
        cols = ["Model", "Category Accuracy", "Category Macro F1", "Category Weighted F1",
                "Priority Accuracy", "Sentiment Accuracy", "Language Accuracy", "Resolution Accuracy",
                "Average Field Accuracy", "Exact Match Accuracy", "JSON Validity Rate", 
                "Average Latency", "Median Latency", "P95 Latency"]
        
        # Only keep columns that were generated (missing fields might drop some)
        ordered_cols = [c for c in cols if c in comp_df.columns]
        comp_df = comp_df[ordered_cols]
        
        comp_path = self.output_dir / "model_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        return comp_df

    def save_error_analysis(self):
        """Save concatenated error records for all models."""
        if self.all_errors:
            err_df = pd.concat(self.all_errors, ignore_index=True)
            err_path = self.output_dir / "error_analysis.csv"
            err_df.to_csv(err_path, index=False)
            return err_df
        return None

# --- LEGACY BASELINE COMPATIBILITY ---
# Included to preserve backward compatibility for the existing pipeline.

def check_artifacts():
    pass

def load_test_data() -> pd.DataFrame:
    from src.config import TEST_PATH
    return pd.read_csv(TEST_PATH)

def save_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Baseline Confusion Matrix (Test Set)')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    logger.info("Evaluation framework implemented. Legacy execution disabled in this phase.")

if __name__ == "__main__":
    main()
