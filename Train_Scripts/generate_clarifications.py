"""
AIDA Entity Clarification Generation Script

This script generates entity clarifications using Llama-3.2-1B model:
1. Loads preprocessed AIDA data
2. Authenticates with HuggingFace
3. Loads Llama model for clarification generation
4. Generates clarifications for train/val/test splits
5. Saves results to JSON files

Usage:
    python generate_clarifications.py
"""

from huggingface_hub import login
import torch
import pandas as pd
import json
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from Utilities.Utils_generate_clarifications import *

# CONFIGURATION

CONFIG = {
    'model_name': 'meta-llama/Llama-3.2-1B',
    'batch_size': 32,
    'max_new_tokens': 50,
    'temperature': 0.3,
    'context_window_size': 100,
    'data_dir': './AIDA/preprocessed',
    'output_dir': './AIDA/clarifications',
    'checkpoint_interval': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'hf_token': 'your_huggingface_token_here'  # Replace with your token here and in **UTILITIES**
}





# MAIN PIPELINE

def main():
    """Main clarification generation pipeline."""
    print("\n" + "="*70)
    print("AIDA ENTITY CLARIFICATION GENERATION")
    print("="*70)
    
    # Display configuration
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        if key != 'hf_token':
            print(f"   {key}: {value}")
    
    # Authenticate with HuggingFace
    print("\n" + "="*70)
    print("AUTHENTICATION")
    print("="*70)
    login(token=CONFIG['hf_token'])
    print("✅ Authenticated with HuggingFace!")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    print(f"\n Loading preprocessed AIDA data from {CONFIG['data_dir']}...")
    
    df_train, df_val, df_test = load_aida_data()
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Generate clarifications
    print("\n" + "="*70)
    print("GENERATING CLARIFICATIONS")
    print("="*70)
    
    # Validation split
    val_clarifications = generate_clarifications_for_split(model, tokenizer, df_val, 'val')
    print(f"\n✅ Validation complete: {len(val_clarifications)} documents processed")
    
    # Test split
    test_clarifications = generate_clarifications_for_split(model, tokenizer, df_test, 'test')
    print(f"\n✅ Test complete: {len(test_clarifications)} documents processed")
    
    # Train split
    train_clarifications = generate_clarifications_for_split(model, tokenizer, df_train, 'train')
    print(f"\n✅ Train complete: {len(train_clarifications)} documents processed")
    
    # Clear GPU cache
    if CONFIG['device'] == 'cuda':
        torch.cuda.empty_cache()
        print("\n🧹 GPU memory cleared")
    
    # Preview results
    print("\n" + "="*70)
    print("SAMPLE RESULTS")
    print("="*70)
    
    sample_doc = val_clarifications[0]
    print(f"\n Document ID: {sample_doc['doc_id']}")
    print(f"\n Text (first 200 chars):")
    print(sample_doc['text'][:200] + "...")
    
    print(f"\n Entities and Clarifications:")
    for entity in sample_doc['entities'][:3]:
        mention = entity['mention']
        clarification = sample_doc['clarifications'][mention]
        print(f"\n   • {mention}")
        print(f"     → {clarification}")
    
    print(f"\n Statistics:")
    print(f"   Total entities: {len(sample_doc['entities'])}")
    print(f"   Total clarifications: {len(sample_doc['clarifications'])}")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\n✅ All clarifications saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()


