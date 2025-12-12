"""
T5 Model Training Script for Entity Linking

This script trains T5 models for entity linking with and without clarifications:
1. Loads clarification data from JSON files
2. Creates training datasets (baseline and clarified)
3. Trains baseline T5 model (without clarifications)
4. Trains clarified T5 model (with clarifications)
5. Evaluates both models on test set
6. Generates visualizations and saves results

Usage:
    python train_t5_model.py
"""

import pandas as pd
import json
import os
import sys
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from datetime import datetime
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Utils'))
# from Utils_train_t5_model import *
from Utilities.Utils_train_t5_model import *


# CONFIGURATION
# Get the script's directory and set paths relative to workspace root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)

CONFIG = {
    'clarifications_dir': os.path.join(WORKSPACE_ROOT, 'AIDA', 'clarifications'),
    'output_dir': os.path.join(WORKSPACE_ROOT, 'AIDA', 'experiments', 't5_models'),
    'predictions_dir': os.path.join(WORKSPACE_ROOT, 'AIDA', 'experiments', 't5_models', 'predictions'),
    'model_name': 't5-small',
    'max_length': 512,
    'context_window': 250,
    'train_epochs': 60, #set epochs to train
    'batch_size': 32,
    'learning_rate': 3e-4,
    'eval_batch_size': 16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}



# MAIN PIPELINE

def main():
    """Main training and evaluation pipeline."""
    print("\n" + "="*70)
    print("T5 ENTITY LINKING TRAINING PIPELINE")
    print("="*70)
    
    # Display configuration
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # Create output directories
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['predictions_dir'], exist_ok=True)
    
    # STEP 1: LOAD CLARIFICATION DATA
    print("\n" + "="*70)
    print("LOADING CLARIFICATION DATA")
    print("="*70)
    
    train_clarifications = []
    val_clarifications = []
    test_clarifications = []
    
    with open(f"{CONFIG['clarifications_dir']}/clarifications_train.json", 'r', encoding='utf-8') as f:
        train_clarifications = json.load(f)
    
    with open(f"{CONFIG['clarifications_dir']}/clarifications_val.json", 'r', encoding='utf-8') as f:
        val_clarifications = json.load(f)
    
    with open(f"{CONFIG['clarifications_dir']}/clarifications_test.json", 'r', encoding='utf-8') as f:
        test_clarifications = json.load(f)
    
    print(f"\n[OK] Train: {len(train_clarifications)} documents")
    print(f"[OK] Val: {len(val_clarifications)} documents")
    print(f"[OK] Test: {len(test_clarifications)} documents")
    
    # STEP 2: CREATE TRAINING DATASETS
    print("\n" + "="*70)
    print("CREATING TRAINING DATASETS")
    print("="*70)
    
    train_baseline, train_clarified = process_split_for_training(train_clarifications, 'train')
    val_baseline, val_clarified = process_split_for_training(val_clarifications, 'val')
    test_baseline, test_clarified = process_split_for_training(test_clarifications, 'test')
    
    # Save datasets
    data_dir = f"{CONFIG['output_dir']}/processed_for_training"
    save_samples(train_baseline, f'{data_dir}/train_baseline.jsonl')
    save_samples(train_clarified, f'{data_dir}/train_clarified.jsonl')
    save_samples(val_baseline, f'{data_dir}/val_baseline.jsonl')
    save_samples(val_clarified, f'{data_dir}/val_clarified.jsonl')
    save_samples(test_baseline, f'{data_dir}/test_baseline.jsonl')
    save_samples(test_clarified, f'{data_dir}/test_clarified.jsonl')
    
    # STEP 3: PREPARE TOKENIZER AND DATASETS
    print("\n" + "="*70)
    print("PREPARING TOKENIZER AND DATASETS")
    print("="*70)
    
    print(f"\nLoading T5 tokenizer ({CONFIG['model_name']})")
    tokenizer = T5Tokenizer.from_pretrained(CONFIG['model_name'])
    # Add special tokens
    special_tokens = {
        'additional_special_tokens': ['[START_ENT]', '[END_ENT]', '[CLARIFY:', ']']
    }
    tokenizer.add_special_tokens(special_tokens)
    
    print(f"[OK] T5 Tokenizer ready. Vocabulary size: {len(tokenizer)}")
    
    # Create PyTorch datasets
    train_baseline_dataset = EntityLinkingDataset(train_baseline, tokenizer)
    train_clarified_dataset = EntityLinkingDataset(train_clarified, tokenizer)
    val_baseline_dataset = EntityLinkingDataset(val_baseline, tokenizer)
    val_clarified_dataset = EntityLinkingDataset(val_clarified, tokenizer)
    
    print(f"[OK] Train baseline: {len(train_baseline_dataset)} samples")
    print(f"[OK] Train clarified: {len(train_clarified_dataset)} samples")
    print(f"[OK] Val baseline: {len(val_baseline_dataset)} samples")
    print(f"[OK] Val clarified: {len(val_clarified_dataset)} samples")
    
    # STEP 4: TRAIN BASELINE MODEL
    print("\n" + "="*70)
    print("TRAINING BASELINE MODEL (without clarifications)")
    print("="*70)
    
    print(f"\nLoading T5 model ({CONFIG['model_name']})...")
    baseline_model = T5ForConditionalGeneration.from_pretrained(CONFIG['model_name'])
    baseline_model.resize_token_embeddings(len(tokenizer))
    
    baseline_path = os.path.join(CONFIG['output_dir'], "t5_baseline")
    baseline_trainer = train_model(
        baseline_model, tokenizer,
        train_baseline_dataset, val_baseline_dataset,
        baseline_path, 'baseline'
    )
    
    plot_training_metrics(baseline_trainer, baseline_path, 'baseline')

    # STEP 5: TRAIN CLARIFIED MODEL
    print("\n" + "="*70)
    print("TRAINING CLARIFIED MODEL (with clarifications)")
    print("="*70)
    
    print(f"\nLoading T5 model ({CONFIG['model_name']})...")
    clarified_model = T5ForConditionalGeneration.from_pretrained(CONFIG['model_name'])
    clarified_model.resize_token_embeddings(len(tokenizer))
    
    clarified_path = os.path.join(CONFIG['output_dir'], "t5_clarified")
    clarified_trainer = train_model(
        clarified_model, tokenizer,
        train_clarified_dataset, val_clarified_dataset,
        clarified_path, 'clarified'
    )
    
    plot_training_metrics(clarified_trainer, clarified_path, 'clarified')
    

    # STEP 6: EVALUATION ON TEST SET
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    # Load trained models
    baseline_model_eval = T5ForConditionalGeneration.from_pretrained(baseline_path)
    clarified_model_eval = T5ForConditionalGeneration.from_pretrained(clarified_path)
    
    # Generate predictions
    print("\n Baseline Model:")
    baseline_preds, baseline_gts = generate_predictions(
        baseline_model_eval, test_baseline, tokenizer, CONFIG['eval_batch_size']
    )
    
    print("\n Clarified Model:")
    clarified_preds, clarified_gts = generate_predictions(
        clarified_model_eval, test_clarified, tokenizer, CONFIG['eval_batch_size']
    )
    
    # Calculate accuracies
    baseline_acc, baseline_correct, baseline_total = calculate_accuracy(baseline_preds, baseline_gts)
    clarified_acc, clarified_correct, clarified_total = calculate_accuracy(clarified_preds, clarified_gts)
    
    improvement = clarified_acc - baseline_acc
    improvement_pct = ((clarified_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
    

    # STEP 7: ERROR ANALYSIS
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)
    
    error_analysis = perform_error_analysis(baseline_preds, clarified_preds, baseline_gts, test_baseline)
    
    print(f"\n Prediction Agreement:")
    print(f"   Both correct: {error_analysis['both_correct']} ({error_analysis['both_correct']/len(baseline_preds)*100:.1f}%)")
    print(f"   Both incorrect: {error_analysis['both_incorrect']} ({error_analysis['both_incorrect']/len(baseline_preds)*100:.1f}%)")
    print(f"   Only baseline correct: {error_analysis['only_baseline_correct']}")
    print(f"   Only clarified correct: {error_analysis['only_clarified_correct']}")
    

     # STEP 8: SAVE RESULTS
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save evaluation results
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'test_set_size': len(baseline_preds),
        'baseline': {
            'accuracy': baseline_acc,
            'correct': baseline_correct,
            'total': baseline_total
        },
        'clarified': {
            'accuracy': clarified_acc,
            'correct': clarified_correct,
            'total': clarified_total
        },
        'comparison': {
            'absolute_improvement': improvement,
            'relative_improvement_pct': improvement_pct,
            **error_analysis
        }
    }
    
    results_path = os.path.join(CONFIG['predictions_dir'], 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Results saved to: {results_path}")
    
    # Plot results
    plot_evaluation_results(
        baseline_acc, clarified_acc,
        baseline_correct, clarified_correct,
        baseline_total, CONFIG['output_dir']
    )
    
   
   # FINAL SUMMARY
    print("\n" + "="*70)
    print("TRAINING AND EVALUATION COMPLETE")
    print("="*70)
    
    print(f"\n Final Results:")
    print(f"   Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"   Clarified Accuracy: {clarified_acc:.2f}%")
    print(f"   Improvement: {improvement:+.2f}% ({improvement_pct:+.2f}% relative)")
    
    if improvement > 0:
        print(f"\n   [+] Clarifications improved accuracy by {abs(improvement):.2f} percentage points!")
    elif improvement < 0:
        print(f"\n   [-] Clarifications decreased accuracy by {abs(improvement):.2f} percentage points")
    else:
        print(f"\n   [=] No change in accuracy")
    
    print(f"\n All results saved to: {CONFIG['output_dir']}/")
    print("="*70)


if __name__ == "__main__":
    main()
