"""
TicketTriage-SLM — TF-IDF + Logistic Regression Baseline

This module trains a traditional machine learning baseline to predict
the support ticket 'category'.

Design principles:
- No leakage: TF-IDF is fit ONLY on training data.
- Modular: Model saving and a reusable prediction function.
- Validated: Calculates exact metrics purely on the validation set.
"""

import sys
import logging
import joblib
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.config import (
    TRAIN_PATH,
    VAL_PATH,
    BASELINE_DIR,
    RANDOM_SEED,
    LOG_LEVEL,
    LOG_FORMAT,
    CATEGORIES
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Artifact Paths
VECTORIZER_PATH = BASELINE_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = BASELINE_DIR / "logistic_regression.joblib"

def load_data():
    """Loads training and validation datasets."""
    logger.info("Loading training and validation datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    return train_df, val_df

def build_vectorizer():
    """
    Creates the TF-IDF Vectorizer with carefully chosen parameters.
    
    Parameters explained:
    - lowercase=True: "Refund" and "refund" should be treated as the same word.
    - ngram_range=(1,2): Captures single words ("charge") and bigrams ("double charge", "log in").
    - min_df=3: Ignores rare words/typos that appear in fewer than 3 tickets, reducing noise.
    - max_df=0.9: Ignores stop words ("the", "and") appearing in >90% of tickets.
    - sublinear_tf=True: Replaces raw term frequency (tf) with 1 + log(tf). A word appearing
                         10 times isn't strictly 10x more important than appearing 1 time.
    """
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )

def build_model():
    """
    Creates the Logistic Regression model.
    - solver='lbfgs': Good default for multiclass classification.
    - max_iter=1000: Ensures the solver has enough time to converge.
    - random_state=42: Ensures reproducible results.
    - class_weight='balanced': Helps if categories are slightly imbalanced (though ours are balanced).
    """
    return LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )

def predict_category(text: str, vectorizer=None, model=None) -> tuple:
    """
    Reusable prediction function.
    Given raw text, predicts the category and returns probabilities.
    """
    if vectorizer is None or model is None:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        
    # Must be in an iterable (list) for the vectorizer
    X_features = vectorizer.transform([text])
    
    predicted_cat = model.predict(X_features)[0]
    
    # Get probabilities
    probs = model.predict_proba(X_features)[0]
    classes = model.classes_
    prob_dict = {cls: round(prob, 4) for cls, prob in zip(classes, probs)}
    
    return predicted_cat, prob_dict

def main():
    logger.info("=" * 60)
    logger.info("BASELINE MODEL: TF-IDF + Logistic Regression")
    logger.info("=" * 60)
    
    # 1. Load Data
    train_df, val_df = load_data()
    X_train_raw = train_df['text']
    y_train = train_df['category']
    X_val_raw = val_df['text']
    y_val = val_df['category']
    
    logger.info(f"Training records: {len(train_df)}")
    logger.info(f"Validation records: {len(val_df)}")
    
    # 2. Fit and Transform TF-IDF
    logger.info("Fitting TF-IDF exclusively on training data...")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    
    logger.info("Transforming validation data (no fitting)...")
    X_val_tfidf = vectorizer.transform(X_val_raw)
    
    feature_count = X_train_tfidf.shape[1]
    logger.info(f"TF-IDF feature count: {feature_count}")
    
    # 3. Train Logistic Regression
    logger.info("Training Logistic Regression...")
    model = build_model()
    model.fit(X_train_tfidf, y_train)
    logger.info("Training completed.")
    
    # 4. Save Artifacts
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Saved Vectorizer to {VECTORIZER_PATH}")
    logger.info(f"Saved Model to {MODEL_PATH}")
    
    # 5. Validation and Metrics
    logger.info("Evaluating on Validation set...")
    y_pred = model.predict(X_val_tfidf)
    
    acc = accuracy_score(y_val, y_pred)
    prec_macro = precision_score(y_val, y_pred, average='macro')
    rec_macro = recall_score(y_val, y_pred, average='macro')
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    
    logger.info("\nClassification Report:\n" + classification_report(y_val, y_pred))
    
    # 6. Test with Real Examples
    print("\n" + "="*50)
    print("TESTING WITH 10 REAL VALIDATION EXAMPLES")
    print("="*50)
    sample_val = val_df.sample(10, random_state=123)
    
    for idx, row in sample_val.iterrows():
        text = row['text']
        true_cat = row['category']
        pred_cat, probs = predict_category(text, vectorizer, model)
        
        # Format text to fit screen
        display_text = text if len(text) <= 100 else text[:97] + "..."
        
        print(f"\nINPUT: {display_text}")
        print(f"ACTUAL: {true_cat}")
        print(f"PREDICTED: {pred_cat}")
        
        # Sort probabilities highest to lowest for display
        sorted_probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))
        prob_str = " | ".join([f"{k}: {v:.2f}" for k,v in sorted_probs.items()][:3])
        print(f"TOP PROBS: {prob_str}")

    # 7. Sanity Checks
    sanity_pass = True
    if len(vectorizer.vocabulary_) != feature_count:
        sanity_pass = False
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        sanity_pass = False
    if pd.isna(y_pred).any():
        sanity_pass = False
    if not set(y_pred).issubset(set(CATEGORIES)):
        sanity_pass = False
        
    # 8. Output Summary
    print("\n" + "-"*50)
    print("BASELINE MODEL")
    print("-" * 50)
    print("Model:\nTF-IDF + Logistic Regression")
    print(f"\nTraining records:\n{len(train_df)}")
    print(f"\nValidation records:\n{len(val_df)}")
    print(f"\nTF-IDF feature count:\n{feature_count}")
    
    print("\n" + "-"*50)
    print("VALIDATION METRICS")
    print("-" * 50)
    print(f"Accuracy:\n{acc:.4f}")
    print(f"\nMacro Precision:\n{prec_macro:.4f}")
    print(f"\nMacro Recall:\n{rec_macro:.4f}")
    print(f"\nMacro F1:\n{f1_macro:.4f}")
    print(f"\nWeighted F1:\n{f1_weighted:.4f}")
    
    print("\n" + "-"*50)
    print("ARTIFACTS")
    print("-" * 50)
    print(f"TF-IDF:\n{VECTORIZER_PATH}")
    print(f"\nLogistic Regression:\n{MODEL_PATH}")
    
    print("\n" + "-"*50)
    print("STATUS")
    print("-" * 50)
    if sanity_pass:
        print("[PASS] TF-IDF + Logistic Regression baseline completed")
    else:
        print("[FAIL] Sanity checks failed")

if __name__ == "__main__":
    main()
