"""
TicketTriage-SLM — Preprocessing and Splitting

This module loads the raw synthetic dataset, performs strict validation checks,
and safely splits the data into Training, Validation, and Test sets using
stratified splitting to ensure label balance. It also verifies that absolutely
no leakage occurs between the splits.

Usage:
    python -m src.preprocessing
"""

import sys
import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_TICKETS_PATH,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    REQUIRED_COLUMNS,
    VALID_LABELS,
    RANDOM_SEED,
    NUM_SAMPLES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    LOG_LEVEL,
    LOG_FORMAT
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def load_and_validate() -> pd.DataFrame:
    """Loads the raw dataset and validates its integrity."""
    logger.info(f"Loading raw dataset from {RAW_TICKETS_PATH}")
    
    try:
        df = pd.read_csv(RAW_TICKETS_PATH)
    except FileNotFoundError:
        logger.error(f"Raw dataset not found at {RAW_TICKETS_PATH}")
        sys.exit(1)
        
    validation_passed = True
    
    # 2. Validate the required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        validation_passed = False
        
    # 3. Validate no missing values
    missing_vals = df[REQUIRED_COLUMNS].isnull().sum()
    if missing_vals.sum() > 0:
        logger.error(f"Missing values detected:\n{missing_vals[missing_vals > 0]}")
        validation_passed = False
        
    # 4. Validate unique ticket_ids
    if df["ticket_id"].duplicated().any():
        dup_count = df["ticket_id"].duplicated().sum()
        logger.error(f"Found {dup_count} duplicate ticket_ids.")
        validation_passed = False
        
    # 5. Validate unique ticket text
    if df["text"].duplicated().any():
        dup_count = df["text"].duplicated().sum()
        logger.error(f"Found {dup_count} duplicate ticket texts.")
        validation_passed = False
        
    # 6. Validate allowed labels
    for field, allowed in VALID_LABELS.items():
        invalid_mask = ~df[field].isin(allowed)
        if invalid_mask.any():
            invalid_vals = df.loc[invalid_mask, field].unique()
            logger.error(f"Invalid values in '{field}': {invalid_vals}")
            validation_passed = False
            
    # 7 & 8. Check for whitespace-only or empty strings in text
    empty_mask = df["text"].str.strip() == ""
    if empty_mask.any():
        empty_count = empty_mask.sum()
        logger.error(f"Found {empty_count} empty or whitespace-only texts.")
        validation_passed = False
        
    # 9. Check dataset size
    if len(df) != NUM_SAMPLES:
        logger.error(f"Expected {NUM_SAMPLES} samples, but found {len(df)}.")
        validation_passed = False

    if not validation_passed:
        logger.error("Dataset validation failed. Stopping execution.")
        sys.exit(1)
        
    return df

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into train, val, and test sets using stratification."""
    logger.info("Splitting dataset into Train/Validation/Test...")
    
    # Calculate sizes
    # train_test_split works with two splits at a time.
    # First, split into train and temp (val + test)
    temp_ratio = VAL_RATIO + TEST_RATIO
    
    train_df, temp_df = train_test_split(
        df, 
        test_size=temp_ratio, 
        random_state=RANDOM_SEED,
        stratify=df["category"]
    )
    
    # Then split temp into val and test
    # The proportion of val in temp is VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_proportion_in_temp = VAL_RATIO / temp_ratio
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_proportion_in_temp,
        random_state=RANDOM_SEED,
        stratify=temp_df["category"]
    )
    
    return train_df, val_df, test_df

def check_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Verifies that no IDs or text leak across splits."""
    leakage_found = False
    
    # Cross-split Duplicate IDs
    train_val_ids = set(train_df["ticket_id"]).intersection(set(val_df["ticket_id"]))
    train_test_ids = set(train_df["ticket_id"]).intersection(set(test_df["ticket_id"]))
    val_test_ids = set(val_df["ticket_id"]).intersection(set(test_df["ticket_id"]))
    
    if train_val_ids or train_test_ids or val_test_ids:
        logger.error("Leakage detected: Duplicate ticket_ids found across splits!")
        leakage_found = True
        
    # Cross-split Duplicate Text
    train_val_txt = set(train_df["text"]).intersection(set(val_df["text"]))
    train_test_txt = set(train_df["text"]).intersection(set(test_df["text"]))
    val_test_txt = set(val_df["text"]).intersection(set(test_df["text"]))

    if train_val_txt or train_test_txt or val_test_txt:
        logger.error("Leakage detected: Duplicate text found across splits!")
        leakage_found = True
        
    if leakage_found:
        logger.error("Cross-split leakage validation failed. Stopping execution.")
        sys.exit(1)
        
    # Store these counts for the summary printout
    return {
        "train_val_ids": len(train_val_ids),
        "train_test_ids": len(train_test_ids),
        "val_test_ids": len(val_test_ids),
        "train_val_txt": len(train_val_txt),
        "train_test_txt": len(train_test_txt),
        "val_test_txt": len(val_test_txt),
    }

def print_summary(
    df: pd.DataFrame, 
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    leakage_stats: dict
):
    """Prints the final summary exactly as requested by the user."""
    
    def format_dist(column: str) -> str:
        train_dist = train_df[column].value_counts().to_dict()
        val_dist = val_df[column].value_counts().to_dict()
        test_dist = test_df[column].value_counts().to_dict()
        
        return (
            f"Training:   {train_dist}\n"
            f"Validation: {val_dist}\n"
            f"Test:       {test_dist}"
        )

    print("\n" + "-"*50)
    print("DATASET SUMMARY")
    print("-"*50)
    print(f"Total records:      {len(df)}")
    print(f"Training records:   {len(train_df)}")
    print(f"Validation records: {len(val_df)}")
    print(f"Test records:       {len(test_df)}")

    print("\n" + "-"*50)
    print("MISSING VALUE CHECK")
    print("-"*50)
    print(f"Training:   {train_df.isnull().sum().sum()}")
    print(f"Validation: {val_df.isnull().sum().sum()}")
    print(f"Test:       {test_df.isnull().sum().sum()}")

    print("\n" + "-"*50)
    print("DUPLICATE CHECK")
    print("-"*50)
    print(f"Duplicate IDs:   {df['ticket_id'].duplicated().sum()}")
    print(f"Duplicate text:  {df['text'].duplicated().sum()}")

    print("\n" + "-"*50)
    print("CROSS-SPLIT LEAKAGE")
    print("-"*50)
    print(f"Train <-> Validation: {leakage_stats['train_val_ids']} IDs, {leakage_stats['train_val_txt']} texts")
    print(f"Train <-> Test:       {leakage_stats['train_test_ids']} IDs, {leakage_stats['train_test_txt']} texts")
    print(f"Validation <-> Test:  {leakage_stats['val_test_ids']} IDs, {leakage_stats['val_test_txt']} texts")

    print("\n" + "-"*50)
    print("CATEGORY DISTRIBUTION")
    print("-"*50)
    print(format_dist("category"))

    print("\n" + "-"*50)
    print("PRIORITY DISTRIBUTION")
    print("-"*50)
    print(format_dist("priority"))

    print("\n" + "-"*50)
    print("SENTIMENT DISTRIBUTION")
    print("-"*50)
    print(format_dist("sentiment"))

    print("\n" + "-"*50)
    print("RESOLUTION DISTRIBUTION")
    print("-"*50)
    print(format_dist("resolution"))

    print("\n" + "-"*50)
    print("FINAL STATUS")
    print("-"*50)
    print("[PASS] Dataset validation successful")


def main():
    # 1. Load and thoroughly validate raw data
    df = load_and_validate()
    
    # 2. Split data safely
    train_df, val_df, test_df = split_data(df)
    
    # 3. Check for any cross-split leakage
    leakage_stats = check_leakage(train_df, val_df, test_df)
    
    # 4. Save processed datasets
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    logger.info(f"Saved Train: {TRAIN_PATH}")
    logger.info(f"Saved Val:   {VAL_PATH}")
    logger.info(f"Saved Test:  {TEST_PATH}")
    
    # 5. Output summary
    print_summary(df, train_df, val_df, test_df, leakage_stats)


if __name__ == "__main__":
    main()
