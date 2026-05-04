"""
Reproduces the exact MindScope test split and saves a stratified subsample
suitable for LLM benchmarking (~250 rows, min 30 per class).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/sentiment_mental_health_clean.csv"
OUTPUT_PATH = "llm_benchmark/results/test_sample.csv"

MIN_PER_CLASS = 30   # floor for minority classes
TARGET_TOTAL  = 250  # approximate total sample size

LABEL_ORDER = [
    "Normal", "Depression", "Suicidal",
    "Anxiety", "Bipolar", "Stress", "Personality disorder"
]

def main():
    df = pd.read_csv(DATA_PATH)
    df["statement"] = df["statement"].astype(str).str.strip()
    df = df[df["statement"] != ""]

    # --- Reproduce exact train/val/test split from notebook ---
    _, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["status"], random_state=42
    )
    _, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["status"], random_state=42
    )

    print(f"Full test set: {len(test_df)} rows")
    print("\nClass counts in test set:")
    print(test_df["status"].value_counts())

    # --- Stratified subsample with minimum floor ---
    counts = test_df["status"].value_counts()
    total_after_floor = MIN_PER_CLASS * len(LABEL_ORDER)
    remaining = TARGET_TOTAL - total_after_floor

    proportions = counts / counts.sum()
    extra = (proportions * remaining).round().astype(int)

    samples = []
    for label in LABEL_ORDER:
        n = MIN_PER_CLASS + extra.get(label, 0)
        available = test_df[test_df["status"] == label]
        n = min(n, len(available))
        samples.append(available.sample(n=n, random_state=42))

    sample_df = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)
    sample_df = sample_df[["statement", "status"]]

    print(f"\nSubsample: {len(sample_df)} rows")
    print("\nClass counts in subsample:")
    print(sample_df["status"].value_counts())

    sample_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
