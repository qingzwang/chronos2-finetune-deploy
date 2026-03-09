"""
Split data files into train/test sets (85%/15%).
"""

import os
import random
import pandas as pd
from pathlib import Path

# Configuration
DATA_DIR = "/home/ubuntu/efs/to_customer/zendure/data"
OUTPUT_DIR = "/home/ubuntu/efs/to_customer/zendure/zero-shot"
TRAIN_RATIO = 0.85
RANDOM_SEED = 42

def main():
    # Get all CSV files
    data_path = Path(DATA_DIR)
    csv_files = sorted([f.name for f in data_path.glob("*.csv")])

    print(f"Total files: {len(csv_files)}")

    # Shuffle with fixed seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(csv_files)

    # Split
    n_train = int(len(csv_files) * TRAIN_RATIO)
    train_files = csv_files[:n_train]
    test_files = csv_files[n_train:]

    print(f"Train files: {len(train_files)} ({len(train_files)/len(csv_files)*100:.1f}%)")
    print(f"Test files: {len(test_files)} ({len(test_files)/len(csv_files)*100:.1f}%)")

    # Create dataframe
    records = []
    for f in train_files:
        records.append({'filename': f, 'split': 'train'})
    for f in test_files:
        records.append({'filename': f, 'split': 'test'})

    df = pd.DataFrame(records)

    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, "data_split.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSplit saved to: {output_path}")

    # Show examples
    print("\nTrain examples:")
    print(df[df['split'] == 'train'].head())
    print("\nTest examples:")
    print(df[df['split'] == 'test'].head())

if __name__ == "__main__":
    main()
