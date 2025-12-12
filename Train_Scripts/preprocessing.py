"""
AIDA Dataset Preprocessing Script

This script performs the complete preprocessing pipeline for the AIDA-CoNLL dataset:
1. Loads data from HuggingFace
2. Adds context to entity mentions
3. Normalizes mention text
4. Removes overlapping entities
5. Saves preprocessed data to parquet files

Usage:
    python preprocessing.py
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from tqdm import tqdm
from huggingface_hub import hf_hub_download

import tqdm as notebook_tqdm
from Utilities.Utils_preprocess import *
    



# MAIN PREPROCESSING PIPELINE

def load_dataset():
    """Load AIDA dataset from HuggingFace using alternative methods."""
    print("=" * 70)
    print("LOADING AIDA DATASET")
    print("=" * 70)
    

    base_url = "hf://datasets/cyanic-selkie/aida-conll-yago-wikidata/"
    splits = {'train': 'train.parquet', 'validation': 'validation.parquet', 'test': 'test.parquet'}

    # Use default PyArrow engine (not fastparquet) to handle nested structures
    df_train = pd.read_parquet(base_url + splits['train'])
    df_val = pd.read_parquet(base_url + splits['validation'])
    df_test = pd.read_parquet(base_url + splits['test'])


    return df_train, df_val, df_test


def preprocess_data(df_train, df_val, df_test, context_window=200):
    """
    Apply complete preprocessing pipeline to datasets.
    
    Args:
        df_train, df_val, df_test: Input dataframes
        context_window: Context window size for entity mentions
    
    Returns:
        Tuple of (df_train_processed, df_val_processed, df_test_processed)
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Step 1: Add context
    print("\n[Step 1/3] Adding context to entities...")
    df_train_processed = apply_add_context(df_train, context_window=context_window, inplace=False)
    df_val_processed = apply_add_context(df_val, context_window=context_window, inplace=False)
    df_test_processed = apply_add_context(df_test, context_window=context_window, inplace=False)
    print("✅ Context added")
    
    # Step 2: Normalize mentions
    print("\n[Step 2/3] Normalizing mentions...")
    df_train_processed = apply_normalize_mentions(df_train_processed, inplace=True)
    df_val_processed = apply_normalize_mentions(df_val_processed, inplace=True)
    df_test_processed =apply_normalize_mentions(df_test_processed, inplace=True)
    print("✅ Mentions normalized")
    
    # Step 3: Remove overlaps
    print("\n[Step 3/3] Removing overlapping entities...")
    df_train_processed = apply_remove_overlaps(df_train_processed, inplace=True)
    df_val_processed = apply_remove_overlaps(df_val_processed, inplace=True)
    df_test_processed = apply_remove_overlaps(df_test_processed, inplace=True)
    print("✅ Overlaps removed")
    
    return df_train_processed, df_val_processed, df_test_processed


def save_preprocessed_data(df_train, df_val, df_test, output_dir='./AIDA/preprocessed'):
    """Save preprocessed data to parquet files."""
    print("\n" + "=" * 70)
    print("SAVING PREPROCESSED DATA")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    df_train.to_parquet(f'{output_dir}/train.parquet', index=False)
    df_val.to_parquet(f'{output_dir}/validation.parquet', index=False)
    df_test.to_parquet(f'{output_dir}/test.parquet', index=False)
    
    print(f"\n✅ Saved processed data to {output_dir}/")
    print(f"   Train: {len(df_train):,} documents")
    print(f"   Val: {len(df_val):,} documents")
    print(f"   Test: {len(df_test):,} documents")
    
    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    if len(df_val) > 0 and len(df_val.iloc[0]['entities']) > 0:
        sample = df_val.iloc[0]['entities'][0]
        print(f"\nSample entity keys: {list(sample.keys())}")
        
        if 'normalized_mention' in sample:
            print("✅ normalized_mention field present!")
            
            # Count unique normalized mentions
            unique_val = set()
            for _, row in df_val.iterrows():
                for entity in row['entities']:
                    unique_val.add(entity['normalized_mention'])
            
            print(f"✅ Validation unique normalized mentions: {len(unique_val):,}")
            
            # Train statistics
            train_normalized = set()
            train_total = 0
            for _, row in df_train.iterrows():
                for entity in row['entities']:
                    train_normalized.add(entity.get('normalized_mention', ''))
                    train_total += 1
            
            print(f"\nTrain Statistics:")
            print(f"  Total entities: {train_total:,}")
            print(f"  Unique normalized: {len(train_normalized):,}")
            print(f"  Reduction: {((train_total - len(train_normalized)) / train_total * 100):.1f}%")
        else:
            print("❌ WARNING: normalized_mention missing!")


def main():
    """Main preprocessing pipeline."""
    print("\n" + "=" * 70)
    print("AIDA DATASET PREPROCESSING")
    print("=" * 70)
    
    # Load dataset
    df_train, df_val, df_test = load_dataset()
    
    print(f"\nLoaded dataset:")
    print(f"  Train: {len(df_train):,} examples")
    print(f"  Validation: {len(df_val):,} examples")
    print(f"  Test: {len(df_test):,} examples")
    
    # Preprocess
    df_train_processed, df_val_processed, df_test_processed = preprocess_data(
        df_train, df_val, df_test, 
        context_window=200
    )
    
    # Save
    save_preprocessed_data(df_train_processed, df_val_processed, df_test_processed)
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
