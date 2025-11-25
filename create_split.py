import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_splits(dataset_name, split_ratios, seed=42, num_folds=1):
    """
    Create train/val/test splits with different ratios
    
    Args:
        dataset_name (str): Name of the dataset (mr, r8, r52, ohsumed, 20ng)
        split_ratios (list): List of tuples with (train_ratio, val_ratio, test_ratio)
        seed (int): Random seed for reproducibility
        num_folds (int): Number of different folds to create for each ratio
    """
    # First load and combine all data
    base_path = f'./inputs/{dataset_name}'
    train_df = pd.read_csv(f'{base_path}/train.csv')
    val_df = pd.read_csv(f'{base_path}/val.csv')
    test_df = pd.read_csv(f'{base_path}/test.csv')
    # Combine all data
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Basic validation: ensure label column exists
    if 'label' not in pd.concat([train_df, val_df, test_df], ignore_index=True).columns:
        raise ValueError("Input CSVs must contain a 'label' column for stratified splitting.")

    for train_ratio, val_ratio, test_ratio in split_ratios:
        # Create directory for this split ratio using the naming convention:
        # inputs/{dataset}_{train}_{val}_{test} (siblings to the original dataset folder)
        train_pct = int(train_ratio * 100)
        val_pct = int(val_ratio * 100)
        test_pct = int(test_ratio * 100)
        split_base_name = f"{dataset_name}_{train_pct}_{val_pct}_{test_pct}"
        # Place the split folder under ./inputs so it's a sibling of the original dataset folder
        split_dir = Path(base_path).parent / split_base_name
        os.makedirs(split_dir, exist_ok=True)

        # Set random seed
        fold_seed = seed

        # First split into train and temp (val+test)
        # Use stratified splitting when possible; otherwise fall back to non-stratified
        try:
            train_df, temp_df = train_test_split(
                full_df,
                train_size=train_ratio,
                stratify=full_df['label'],
                random_state=fold_seed
            )
        except ValueError:
            # Stratify failed (e.g., not enough members of a class for the requested split)
            train_df, temp_df = train_test_split(
                full_df,
                train_size=train_ratio,
                random_state=fold_seed
            )

        # Then split temp into val and test
        # Adjust val_ratio to be relative to what's left after train split
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        try:
            val_df, test_df = train_test_split(
                temp_df,
                train_size=relative_val_ratio,
                stratify=temp_df['label'],
                random_state=fold_seed
            )
        except ValueError:
            val_df, test_df = train_test_split(
                temp_df,
                train_size=relative_val_ratio,
                random_state=fold_seed
            )

        # Save splits
        train_df.to_csv(split_dir / 'train.csv', index=False)
        val_df.to_csv(split_dir / 'val.csv', index=False)
        test_df.to_csv(split_dir / 'test.csv', index=False)

        # Print statistics
        print(f"\nDataset: {dataset_name} -> {split_base_name}:")
        print(f"Train size: {len(train_df)} ({len(train_df)/len(full_df):.2%})")
        print(f"Val size: {len(val_df)} ({len(val_df)/len(full_df):.2%})")
        print(f"Test size: {len(test_df)} ({len(test_df)/len(full_df):.2%})")

        # Print class distribution
        print("\nClass distribution:")
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            class_dist = split_df['label'].value_counts(normalize=True)
            print(f"{split_name}: {dict(class_dist.round(3))}")

if __name__ == "__main__":
    # Define different split ratios to try
    split_ratios = [
        (0.18, 0.02, 0.8),
        (0.27, 0.03, 0.7),
        (0.36, 0.04, 0.6),
        (0.45, 0.05, 0.05),
    ]
    
    # List of datasets
    datasets = ['mr', 'r8', 'r52', 'ohsumed', '20ng']
    
    # Create splits for each dataset
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        create_splits(dataset, split_ratios, seed=42)
