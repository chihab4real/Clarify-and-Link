from huggingface_hub import login
import torch
import pandas as pd
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

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
    'hf_token': 'your_huggingface_token_here'  # Replace with your token
}


def create_prompt(mention, context_left, context_right):
    """
    Create clarification prompt for entity.
    
    Args:
        mention: Entity text to clarify
        context_left: Text before mention
        context_right: Text after mention
    
    Returns:
        Formatted prompt string for LLM generation
    """
    window_size = CONFIG['context_window_size']
    
    context_left = context_left[-window_size:] if len(context_left) > window_size else context_left
    context_right = context_right[:window_size] if len(context_right) > window_size else context_right
    
    prompt = f"""Based on this context: "{context_left} {mention} {context_right}"

            Provide a brief, factual description for the entity "{mention}".
            Identify what this specific mention refers to.
            Use simple English (max 40 words).

            Description:"""
    
    return prompt


def load_aida_data():
    """
    Load preprocessed AIDA train/val/test splits from parquet files.
    
    Reads parquet files from directory specified in CONFIG['data_dir'].
    Prints dataset statistics including document and entity counts.
    
    Args:
        None (uses CONFIG['data_dir'] for file paths)
    
    Returns:
        Tuple of (df_train, df_val, df_test):
        - df_train: pandas DataFrame with training documents and entities
        - df_val: pandas DataFrame with validation documents and entities
        - df_test: pandas DataFrame with test documents and entities
    """
    print("\n" + "="*70)
    print("LOADING AIDA DATA")
    print("="*70)

    data_dir = CONFIG['data_dir']
    print(f"\n Loading from: {data_dir}")

    df_train = pd.read_parquet(f'{data_dir}/train.parquet')
    df_val = pd.read_parquet(f'{data_dir}/validation.parquet')
    df_test = pd.read_parquet(f'{data_dir}/test.parquet')

    print(f"\n✓ Train: {len(df_train)} documents")
    print(f"✓ Validation: {len(df_val)} documents")
    print(f"✓ Test: {len(df_test)} documents")

    # Count entities
    train_entities = sum(len(row['entities']) for _, row in df_train.iterrows())
    val_entities = sum(len(row['entities']) for _, row in df_val.iterrows())
    test_entities = sum(len(row['entities']) for _, row in df_test.iterrows())

    print(f"\n Total entities:")
    print(f"   Train: {train_entities:,}")
    print(f"   Val: {val_entities:,}")
    print(f"   Test: {test_entities:,}")

    return df_train, df_val, df_test




# Load Llama Model for Clarification Generation
def load_model_and_tokenizer():
    """
    Load HuggingFace model and tokenizer for clarification generation.
    
    Uses configuration from CONFIG dictionary for model selection and device placement.
    Automatically sets pad_token to eos_token if not available.
    
    Args:
        None (uses CONFIG global dictionary)
    
    Returns:
        Tuple of (model, tokenizer):
        - model: AutoModelForCausalLM instance loaded with specified dtype and device
        - tokenizer: AutoTokenizer instance with pad_token configured
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    print(f"\n Model: {CONFIG['model_name']}")
    print(f"  Device: {CONFIG['device']}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.float16 if CONFIG['device'] == 'cuda' else torch.float32,
        device_map='auto'
    )

    model.eval()

    print(f"✓ Model loaded on {CONFIG['device']}")
    if CONFIG['device'] == 'cuda':
        print(f"✓ GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

    return model, tokenizer



# Clarification Generation
def generate_clarifications_batch(model, tokenizer, batch_data):
    """
    Generate clarifications for a batch of mentions using LLM.
    
    Processes multiple mentions in parallel for efficient GPU utilization.
    Creates prompts, generates descriptions, and handles empty outputs with fallbacks.

    Args:
        model: Loaded AutoModelForCausalLM instance for text generation
        tokenizer: AutoTokenizer instance for encoding/decoding text
        batch_data: List of tuples, each containing:
            - mention (str): Entity text to clarify
            - context_left (str): Text before mention
            - context_right (str): Text after mention
            - normalized (str): Normalized mention form (not used in generation)

    Returns:
        List of strings containing generated clarifications, one per input mention.
        Falls back to "Entity: {mention}" if generation produces empty output.
    """
    # Create prompts
    prompts = [
        create_prompt(mention, ctx_left, ctx_right)
        for mention, ctx_left, ctx_right, _ in batch_data
    ]

    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        padding_side='left',
        truncation=True,
        max_length=512
    ).to(CONFIG['device'])

    # Generate batch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG['max_new_tokens'],
            temperature=CONFIG['temperature'],
            do_sample=False,  # Deterministic
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode batch
    clarifications = []
    for i, output in enumerate(outputs):
        # Remove prompt from output
        prompt_length = inputs['input_ids'][i].shape[0]
        generated_ids = output[prompt_length:]

        clarification = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Fallback if empty
        if not clarification:
            clarification = f"Entity: {batch_data[i][0]}"

        clarifications.append(clarification)

    return clarifications



def collect_unique_mentions(df, split_name):
    """
    Collect unique normalized entity mentions from a dataset split.
    
    Deduplicates mentions by normalized form to reduce redundant LLM queries.
    Stores context information for the first occurrence of each unique mention.
    Prints statistics on reduction achieved through deduplication.
    
    Args:
        df: pandas DataFrame with 'entities' column containing entity dictionaries
        split_name: String identifier for the split (e.g., 'train', 'val', 'test')
                   used for logging purposes
    
    Returns:
        Tuple of (unique_mentions, original_case_map):
        - unique_mentions: Dictionary mapping normalized_mention -> {'context_left': str, 'context_right': str}
        - original_case_map: Dictionary mapping normalized_mention -> original case mention string
    """
    print(f"\nCollecting unique mentions from {split_name}...")

    unique_mentions = {}
    original_case_map = {}

    for idx, row in df.iterrows():
        for entity in row['entities']:
            normalized = entity.get('normalized_mention', entity.get('mention', '').lower().strip())
            original = entity.get('mention', '')

            if normalized not in unique_mentions:
                unique_mentions[normalized] = {
                    'context_left': entity.get('context_left', ''),
                    'context_right': entity.get('context_right', '')
                }
                original_case_map[normalized] = original

    total_entities = sum(len(row['entities']) for _, row in df.iterrows())
    reduction = (1 - len(unique_mentions)/total_entities) * 100

    print(f"   Unique mentions: {len(unique_mentions)} (vs {total_entities} total)")
    print(f"   Reduction: {reduction:.1f}%")

    return unique_mentions, original_case_map



def convert_to_serializable(obj):
    """
    Convert numpy/pandas data types to JSON-serializable Python types.
    
    Recursively handles dictionaries, lists, numpy integers/floats, and pandas NA values.
    Essential for saving processed data to JSON format.
    
    Args:
        obj: Object to convert (can be dict, list, numpy type, pandas type, or primitive)
    
    Returns:
        JSON-serializable version of input object:
        - Converts numpy int64 -> int
        - Converts numpy float64 -> float  
        - Converts pandas NA/NaN -> None
        - Recursively processes dict values and list items
        - Returns primitives unchanged
    """
    """Convert numpy arrays and other non-serializable objects to Python types."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj
    

def generate_clarifications_for_split(model, tokenizer, df, split_name):
    """
    Generate clarifications for all unique mentions in a dataset split.
    
    Main orchestration function that:
    1. Collects unique mentions to avoid redundant generation
    2. Generates clarifications in batches for efficiency
    3. Maps clarifications back to original documents
    4. Saves checkpoints periodically and final results
    
    Args:
        model: Loaded AutoModelForCausalLM instance for text generation
        tokenizer: AutoTokenizer instance for encoding/decoding
        df: pandas DataFrame containing documents with 'text' and 'entities' columns
        split_name: String identifier ('train', 'val', or 'test') for output file naming
    
    Returns:
        List of document dictionaries, each containing:
        - doc_id: Document index
        - text: Original document text
        - entities: List of entity dictionaries with position/type info
        - clarifications: Dictionary mapping mention -> clarification text
        - num_entities: Total entity count in document
        - num_clarifications: Count of unique clarified mentions
    """
    """Generate clarifications for entire split using batching."""
    print("\n" + "="*70)
    print(f"PROCESSING: {split_name.upper()}")
    print("="*70)

    checkpoint_dir = f"{CONFIG['output_dir']}/clarifications_checkpoints/{split_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Collect unique mentions
    unique_mentions, original_case_map = collect_unique_mentions(df, split_name)

    # Prepare batch data
    batch_data = [
        (original_case_map[norm], context['context_left'], context['context_right'], norm)
        for norm, context in unique_mentions.items()
    ]

    # Estimate time
    num_batches = len(batch_data) // CONFIG['batch_size'] + 1
    print(f"\ Batched generation:")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Total batches: {num_batches}")
    print(f"   Total mentions: {len(batch_data)}")

    # Generate in batches with single progress bar
    global_clarifications = {}

    with tqdm(total=len(batch_data), desc=f"Generating {split_name}", unit=" mentions", 
              position=0, leave=True, ncols=80) as pbar:
        for i in range(0, len(batch_data), CONFIG['batch_size']):
            batch = batch_data[i:i + CONFIG['batch_size']]

            clarifications = generate_clarifications_batch(model, tokenizer, batch)

            # Store results
            for (mention, _, _, normalized), clarification in zip(batch, clarifications):
                global_clarifications[normalized] = clarification

            # Update progress bar
            pbar.update(len(batch))

            # Save checkpoint
            if (i // CONFIG['batch_size'] + 1) % (CONFIG['checkpoint_interval'] // CONFIG['batch_size']) == 0:
                checkpoint_path = f'{checkpoint_dir}/checkpoint_{i + len(batch)}.json'
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(global_clarifications, f, indent=2, ensure_ascii=False)

    # Map to documents
    print(f"\n Mapping clarifications to documents...")
    results = []

    for idx, row in df.iterrows():
        doc_clarifications = {}
        for entity in row['entities']:
            original_mention = entity['mention']
            normalized = entity.get('normalized_mention', original_mention.lower().strip())
            doc_clarifications[original_mention] = global_clarifications.get(
                normalized,
                f"Entity: {original_mention}"
            )


        serializable_entities = [convert_to_serializable(entity) for entity in row['entities']]

        results.append({
            'doc_id': int(idx),  #
            'text': str(row['text']),  
            'entities': serializable_entities,
            'clarifications': doc_clarifications
        })

    # Save final
    output_path = f"{CONFIG['output_dir']}/clarifications_{split_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n {split_name.upper()} complete!")
    print(f"   Saved to: {output_path}")

    return results