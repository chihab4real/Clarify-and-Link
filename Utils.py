import pandas as pd
import numpy as np
import os
import sys
import warnings
import re
from collections import defaultdict
import json
from tqdm import tqdm

from huggingface_hub import login
import torch
import pandas as pd
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM




def filter_valid_entities(row):
    """
    Filter entities to keep only those with valid KB links (qid or pageid).
    
    Args:
        row: DataFrame row containing 'entities' column with list of entity dictionaries
    
    Returns:
        List of entities that have either 'qid' or 'pageid' attributes
    """
    entities = row['entities'] if 'entities' in row and row['entities'] is not None else []
    
    valid_entities = []
    for entity in entities:
        if entity.get('qid') is not None or entity.get('pageid') is not None:
            valid_entities.append(entity)
    
    return valid_entities


def apply_filter_valid_entities(df, inplace=False):
    """
    Apply entity filtering to entire DataFrame to remove entities without KB links.
    
    Args:
        df: DataFrame with 'entities' column
        inplace: If True, modify df in place; if False, return a copy (default: False)
    
    Returns:
        DataFrame with filtered entities and new 'num_valid_entities' column
    """
    if not inplace:
        df = df.copy()
    
    print("Filtering entities without KB links...")
    df['entities'] = df.apply(filter_valid_entities, axis=1)
    df['num_valid_entities'] = df['entities'].apply(len)
    
    return df





def extract_mention_with_context(text, start, end, context_window=50):
    """
    Extract a mention and its surrounding context from text.
    
    Args:
        text: Full document text
        start: Start position of mention in text
        end: End position of mention in text
        context_window: Number of characters to include before/after mention (default: 50)
    
    Returns:
        Dictionary with 'mention', 'left_context', 'right_context', 'full_context',
        'mention_start', and 'mention_end'
    """
    mention = text[start:end]
    
    # Get left and right context
    left_start = max(0, start - context_window)
    right_end = min(len(text), end + context_window)
    
    left_context = text[left_start:start]
    right_context = text[end:right_end]
    full_context = text[left_start:right_end]
    
    return {
        'mention': mention,
        'left_context': left_context,
        'right_context': right_context,
        'full_context': full_context,
        'mention_start': start,
        'mention_end': end
    }

def add_context_to_entities(row, context_window=50):
    """
    Add context information to all entities in a row.
    
    Args:
        row: DataFrame row with 'text' and 'entities' columns
        context_window: Number of characters to include before/after each mention (default: 50)
    
    Returns:
        List of entities enriched with context information
    """
    text = row['text']
    entities = row.get('entities', [])
    
    if entities is None or len(entities) == 0:
        return []
    
    entities_with_context = []
    for entity in entities:
        entity_copy = entity.copy()
        context_info = extract_mention_with_context(
            text, 
            entity['start'], 
            entity['end'], 
            context_window
        )
        entity_copy.update(context_info)
        entities_with_context.append(entity_copy)
    
    return entities_with_context


def apply_add_context(df, context_window=50, inplace=False):
    """
    Apply context extraction to all entities in DataFrame.
    
    Args:
        df: DataFrame with 'text' and 'entities' columns
        context_window: Number of characters to include before/after mentions (default: 50)
        inplace: If True, modify df in place; if False, return a copy (default: False)
    
    Returns:
        DataFrame with entities enriched with context information
    """
    if inplace is False:
        df = df.copy()
    
    print(f"Adding context (window={context_window}) to entities...")
    df['entities'] = df.apply(
        lambda row: add_context_to_entities(row, context_window), 
        axis=1
    )
    
    return df





def normalize_mention(mention):
    """
    Normalize entity mention for consistent matching.
    
    Operations:
    - Convert to lowercase
    - Strip leading/trailing whitespace
    - Remove extra internal whitespace
    - Strip leading/trailing punctuation
    """
    if not mention:
        return ""
    
    # Lowercase
    normalized = mention.lower()
    
    # Strip and collapse whitespace
    normalized = ' '.join(normalized.split())
    
    # Strip common punctuation from edges (but keep internal periods like "U.S.")
    normalized = normalized.strip('.,;:!?\'"()[]{}')
    
    return normalized


def add_normalized_mentions(row):
    """Add normalized_mention field to each entity."""
    for entity in row['entities']:
        entity['normalized_mention'] = normalize_mention(entity.get('mention', ''))
    return row


def apply_normalize_mentions(df, inplace=False):
    """Apply mention normalization to dataframe."""
    if not inplace:
        df = df.copy()
    
    print(f"Normalizing mentions in {len(df)} documents...")
    df = df.apply(add_normalized_mentions, axis=1)
    
    return df






def remove_overlapping_entities(row):
    """
    Remove overlapping entities, keeping longer mentions when overlaps occur.
    
    Args:
        row: DataFrame row with 'entities' column
    
    Returns:
        Tuple of (non-overlapping entities list, count of removed entities)
    """
    entities = row.get('entities', [])
    if not entities:
        return [], 0
    
    # Sort by start position, then by length (descending)
    sorted_entities = sorted(
        entities, 
        key=lambda e: (e['start'], -(e['end'] - e['start']))
    )
    
    non_overlapping = []
    last_end = -1
    
    for entity in sorted_entities:
        if entity['start'] >= last_end:
            non_overlapping.append(entity)
            last_end = entity['end']
    
    num_removed = len(entities) - len(non_overlapping)
    
    return non_overlapping, num_removed

def apply_remove_overlaps(df, inplace=False):
    """
    Apply overlap removal to all entities in DataFrame.
    
    Args:
        df: DataFrame with 'entities' column
        inplace: If True, modify df in place; if False, return a copy (default: False)
    
    Returns:
        DataFrame with overlapping entities removed and 'removed_overlaps' count column added
    """
    if not inplace:
        df = df.copy()
    
    print("Removing overlapping entities...")
    result = df.apply(remove_overlapping_entities, axis=1)
    df['entities'] = result.apply(lambda x: x[0])
    df['removed_overlaps'] = result.apply(lambda x: x[1])
    
    total_removed = df['removed_overlaps'].sum()
    print(f"  Removed {total_removed} overlapping entities")
    
    return df





def create_mention_candidate_pairs(row, max_candidates=10):
    """
    Create mention-candidate pairs for entity linking from entity annotations.
    
    Args:
        row: DataFrame row with 'entities' column
        max_candidates: Maximum number of candidate entities per mention (default: 10)
    
    Returns:
        List of mention-candidate pair dictionaries with mention, context, type, and true entity info
    """
    
    pairs = []
    
    for entity in row.get('entities', []):
        pair = {
            'mention': entity.get('mention', ''),
            'context': entity.get('full_context', ''),
            'entity_type': entity.get('tag', ''),
            'true_qid': entity.get('qid'),
            'true_pageid': entity.get('pageid'),
            'true_title': entity.get('title'),
            # we'd generate candidates here
            'candidates': []  #for candidate entities
        }
        pairs.append(pair)
    
    return pairs


def apply_create_candidate_pairs(df, max_candidates=10, inplace=False):
    """
    Apply mention-candidate pair creation to entire DataFrame.
    
    Args:
        df: DataFrame with 'entities' column
        max_candidates: Maximum number of candidates per mention (default: 10)
        inplace: If True, modify df in place; if False, return a copy (default: False)
    
    Returns:
        DataFrame with 'mention_candidate_pairs' column added
    """
    if not inplace:
        df = df.copy()
    
    print("Creating mention-candidate pairs...")
    df['mention_candidate_pairs'] = df.apply(
        lambda row: create_mention_candidate_pairs(row, max_candidates),
        axis=1
    )
    
    return df






def create_nil_detection_examples(row):
    """
    Separate entities into NIL (no KB link) and linked categories.
    
    Args:
        row: DataFrame row with 'entities' column
    
    Returns:
        Tuple of (nil_examples list, linked_examples list)
    """
    nil_examples = []
    linked_examples = []
    
    for entity in row.get('entities', []):
        entity_example = {
            'mention': entity.get('mention', ''),
            'context': entity.get('full_context', ''),
            'entity_type': entity.get('tag', ''),
            'is_nil': entity.get('qid') is None and entity.get('pageid') is None
        }
        
        if entity_example['is_nil']:
            nil_examples.append(entity_example)
        else:
            linked_examples.append(entity_example)
    
    return nil_examples, linked_examples


def apply_create_nil_examples(df, inplace=False):
    """
    Create NIL detection examples for entire DataFrame.
    """
    if not inplace:
        df = df.copy()
    
    print("Creating NIL detection examples...")
    result = df.apply(create_nil_detection_examples, axis=1)
    df['nil_examples'] = result.apply(lambda x: x[0])
    df['linked_examples'] = result.apply(lambda x: x[1])
    
    total_nil = df['nil_examples'].apply(len).sum()
    total_linked = df['linked_examples'].apply(len).sum()
    print(f"  NIL entities: {total_nil}, Linked entities: {total_linked}")
    
    return df





def split_long_documents(row, max_length=512, overlap=50):
    """
    Split long documents into overlapping chunks for model processing.
    
    Args:
        row: DataFrame row with 'text' and 'entities' columns
        max_length: Maximum character length per chunk (default: 512)
        overlap: Number of overlapping characters between chunks (default: 50)
    
    Returns:
        List of document chunks with adjusted entity positions
    """
    text = row['text']
    entities = row.get('entities', [])
    
    if len(text) <= max_length:
        return [row.to_dict()]  # No splitting needed
    
    chunks = []
    start_pos = 0
    chunk_id = 0
    
    while start_pos < len(text):
        end_pos = min(start_pos + max_length, len(text))
        chunk_text = text[start_pos:end_pos]
        
        # Find entities in this chunk
        chunk_entities = []
        for entity in entities:
            if entity['start'] >= start_pos and entity['end'] <= end_pos:
                # Adjust entity positions relative to chunk
                adjusted_entity = entity.copy()
                adjusted_entity['start'] = entity['start'] - start_pos
                adjusted_entity['end'] = entity['end'] - start_pos
                chunk_entities.append(adjusted_entity)
        
        chunk = {
            'text': chunk_text,
            'entities': chunk_entities,
            'chunk_start': start_pos,
            'chunk_end': end_pos,
            'chunk_id': chunk_id,
            'is_chunk': True
        }
        chunks.append(chunk)
        
        chunk_id += 1
        # Move to next chunk with overlap
        start_pos = end_pos - overlap
        if start_pos >= len(text) - overlap:
            break
    
    return chunks


def apply_split_long_documents(df, max_length=512, overlap=50):
    """
    Apply document splitting to entire DataFrame.
    
    Args:
        df: DataFrame with 'text' and 'entities' columns
        max_length: Maximum character length per chunk (default: 512)
        overlap: Number of overlapping characters between chunks (default: 50)
    
    Returns:
        New DataFrame with split document chunks
    """
    all_chunks = []
    for idx, row in df.iterrows():
        chunks = split_long_documents(row, max_length, overlap)
        for chunk in chunks:
            # Preserve original index info
            chunk['original_index'] = idx
            all_chunks.append(chunk)
    
    df_chunks = pd.DataFrame(all_chunks)
    
    print(f"  Original docs: {len(df)}, After splitting: {len(df_chunks)}")
    
    return df_chunks


# medmentions
def parse_pubtator_file(file_path, max_docs=None):
    """
    Parse PubTator format file and return list of mention dictionaries.
    
    Args:
        file_path: Path to corpus_pubtator.txt
        max_docs: Optional limit on number of documents
    
    Returns:
        List of mention dictionaries with metadata
    """
    mentions = []
    current_pmid = None
    current_title = ""
    current_abstract = ""
    doc_count = 0
    
    print(f"Parsing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip('\n')
            
            if not line:  # Empty line = document separator
                current_pmid = None
                current_title = ""
                current_abstract = ""
                continue
            
            parts = line.split('|')
            
            if len(parts) == 3:  # Title or abstract line
                pmid, line_type, text = parts
                if line_type == 't':
                    current_pmid = pmid
                    current_title = text
                    doc_count += 1
                    if max_docs and doc_count > max_docs:
                        break
                elif line_type == 'a':
                    current_abstract = text
            
            elif '\t' in line:  
                fields = line.split('\t')
                if len(fields) >= 6:
                    pmid = fields[0]
                    start_offset = int(fields[1])
                    end_offset = int(fields[2])
                    mention_text = fields[3]
                    entity_type = fields[4]
                    entity_id = fields[5]
                    
                    
                    full_text = current_title + " " + current_abstract
                    
                    # Extract context (200 chars before and after)
                    context_window = 200
                    context_start = max(0, start_offset - context_window)
                    context_end = min(len(full_text), end_offset + context_window)
                    
                    context_left = full_text[context_start:start_offset]
                    context_right = full_text[end_offset:context_end]
                    
                    mentions.append({
                        'pmid': pmid,
                        'mention': mention_text,
                        'entity_id': entity_id,  
                        'entity_type': entity_type,
                        'start': start_offset,
                        'end': end_offset,
                        'context_left': context_left,
                        'context_right': context_right,
                        'full_context': context_left + " " + mention_text + " " + context_right,
                        'title': current_title,
                        'abstract': current_abstract,
                        'text': full_text
                    })
    
    print(f"Parsed {len(mentions)} mentions from {doc_count} documents")
    return mentions


def filter_valid_entities(df):
    """
    Keep only mentions with valid UMLS CUIDs (for MedMentions dataset).
    
    Args:
        df: DataFrame with 'entity_id' column containing UMLS CUIDs
    
    Returns:
        Filtered DataFrame with only valid entity_id values (not null or '-')
    """
    original_count = len(df)
    
    df_filtered = df[(df['entity_id'].notna()) & (df['entity_id'] != '-')].copy()
    
    filtered_count = len(df_filtered)
    removed = original_count - filtered_count
    
    print(f"Original: {original_count:,} | Valid: {filtered_count:,} | Removed: {removed:,} ({removed/original_count*100:.2f}%)")
    
    return df_filtered




def extract_biomedical_features(df):
    """
    Extract biomedical-specific features for MedMentions dataset.
    
    Args:
        df: DataFrame with 'mention', 'entity_type', and 'full_context' columns
    
    Returns:
        DataFrame with added feature columns:
        - is_abbreviation: Boolean for all-caps mentions ≤6 chars
        - mention_word_count: Number of words in mention
        - semantic_type_main: Primary semantic type from entity_type
        - mention_length: Character length of mention
        - context_length: Word count of full context
    """
    df['is_abbreviation'] = df['mention'].apply(
        lambda x: x.isupper() and len(x) <= 6
    )
    
    df['mention_word_count'] = df['mention'].str.split().str.len()
    
    df['semantic_type_main'] = df['entity_type'].str.split(',').str[0]
    
    df['mention_length'] = df['mention'].str.len()
    
    df['context_length'] = df['full_context'].str.split().str.len()
    
    return df


def create_mention_record(row):
    """
    Convert DataFrame row to standardized mention record for MedMentions.
    
    Args:
        row: DataFrame row with MedMentions columns (pmid, mention, entity_id, etc.)
    
    Returns:
        Dictionary with standardized fields including mention text, context, entity info,
        features, and placeholder for candidate entities
    """
    record = {
        # Core fields
        'pmid': row['pmid'],
        'mention': row['mention'],
        'normalized_mention': row['normalized_mention'],
        
        # Context
        'context_left': row['context_left'],
        'context_right': row['context_right'],
        'full_context': row['full_context'],
        
        # Entity information
        'label_id': row['entity_id'], 
        'entity_type': row['entity_type'],
        'semantic_type_main': row['semantic_type_main'],
        
        # Position
        'start': row['start'],
        'end': row['end'],
        
        # Features
        'is_abbreviation': row['is_abbreviation'],
        'mention_length': row['mention_length'],
        'mention_word_count': row['mention_word_count'],
        'context_length': row['context_length'],
        
        # Metadata
        'title': row['title'],
        'abstract': row['abstract'],
        
        # Placeholder for candidates (to be populated later)
        'candidates': []
    }
    
    return record



# EXPERIMENTS FUNCTIONS

CONFIG = {
    'model_name': 'meta-llama/Llama-3.2-1B',  
    'batch_size': 32,  # Increase for faster processing on GPU
    'max_new_tokens': 50,
    'temperature': 0.3,
    'context_window_size': 100,
    'data_dir': 'data/processed/aida',
    'output_dir': 'data/experiments',
    'checkpoint_interval': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


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









def create_prompt(mention, context_left, context_right):
    """
    Create a prompt for entity clarification generation using context.
    
    Truncates context to specified window size and formats a prompt asking for
    a brief, factual description of the entity mention.
    
    Args:
        mention: Entity text to be clarified (e.g., "Jordan")
        context_left: Text appearing before the mention
        context_right: Text appearing after the mention
    
    Returns:
        String containing formatted prompt with mention and surrounding context
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
    print(f"\n🔍 Collecting unique mentions from {split_name}...")

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





# ============================================================================
# Cell 6: Main Generation Function (FIXED)
# ============================================================================

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
    estimated_time = num_batches * 0.5 / 60  # ~0.5s per batch
    print(f"\n Batched generation:")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Total batches: {num_batches}")
    print(f"   Estimated time: {estimated_time:.1f} minutes")

    # Generate in batches
    global_clarifications = {}

    for i in tqdm(range(0, len(batch_data), CONFIG['batch_size']), desc="Generating batches"):
        batch = batch_data[i:i + CONFIG['batch_size']]

        clarifications = generate_clarifications_batch(model, tokenizer, batch)

        # Store results
        for (mention, _, _, normalized), clarification in zip(batch, clarifications):
            global_clarifications[normalized] = clarification

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





def create_training_samples(doc, use_clarifications=False):
    """
    Create T5 training samples from a document with entity linking annotations.
    
    Generates individual training samples for each entity, with optional clarification
    augmentation. Creates properly formatted input/target pairs for T5 fine-tuning.
    
    Args:
        doc: Dictionary containing:
            - text (str): Full document text
            - entities (list): Entity dictionaries with 'mention', 'qid', 'start', 'end'
            - clarifications (dict, optional): Mapping of mention -> clarification text
        use_clarifications: Boolean flag to include clarifications in input text.
                           If True, appends "[CLARIFY: text]" after entity markers.
    
    Returns:
        List of training sample dictionaries, each containing:
        - input_text: Formatted text with "link entity:" prefix, marked mentions,
                     optional clarifications, and surrounding context
        - target_text: Wikidata QID in format "Q{id}" (e.g., "Q170566")
        
        Skips entities with 'NIL' or None QIDs.
        Truncates long inputs to 512 characters.
    """
    """
    Create individual training samples for each entity in the document.

    Key additions:
    - Task prefix: "link entity:"
    - Q prefix for Wikidata format
    - Better context windowing
    """
    text = doc['text']
    entities = doc['entities']
    clarifications = doc.get('clarifications', {})

    samples = []

    for entity in entities:
        mention = entity.get('mention', '')
        qid = entity.get('qid', 'NIL')
        start = entity.get('start', 0)
        end = entity.get('end', 0)

        # Skip if no valid QID
        if qid == 'NIL' or qid is None:
            continue

        # Clean QID (remove .0 decimal if present)
        qid_clean = str(qid).replace('.0', '')

        # Get context around entity (better than full text)
        context_window = 250
        context_left = text[max(0, start - context_window):start]
        context_right = text[end:min(len(text), end + context_window)]

        # Build marked entity
        if use_clarifications and mention in clarifications:
            clarification = clarifications[mention]
            marked_entity = f"[START_ENT] {mention} [END_ENT] [CLARIFY: {clarification}]"
        else:
            marked_entity = f"[START_ENT] {mention} [END_ENT]"

        # ✅ ADD TASK PREFIX: This tells T5 what to do
        input_text = f"link entity: {context_left}{marked_entity}{context_right}"

        # Truncate if too long
        if len(input_text) > 512:
            input_text = input_text[:512]

        # ✅ TARGET FORMAT: Q + QID (standard Wikidata format)
        target_text = f"Q{qid_clean}"

        samples.append({
            'input_text': input_text,
            'target_text': target_text
        })

    return samples


def process_split_for_training(clarifications_data, split_name):
    """
    Convert clarification data to T5 training format for both baseline and clarified models.
    
    Creates two parallel datasets: one with plain entity markers (baseline) and one with
    clarifications appended (clarify-and-link). Saves both versions to JSONL files.
    
    Args:
        clarifications_data: List of document dictionaries from generate_clarifications_for_split,
                            each containing 'text', 'entities', and 'clarifications'
        split_name: String identifier ('train', 'val', or 'test') for output file naming
    
    Returns:
        Tuple of (baseline_samples, clarified_samples):
        - baseline_samples: List of dicts with input_text (no clarifications) and target_text
        - clarified_samples: List of dicts with input_text (with clarifications) and target_text
        
        Both lists have same length and parallel ordering for comparison.
    """
    """
    Convert clarification data to training format.

    Creates TWO datasets:
    1. Baseline: link entity: [START_ENT]mention[END_ENT] → Q12345
    2. Clarified: link entity: [START_ENT]mention[END_ENT][CLARIFY:...] → Q12345
    """
    print(f"\n🔧 Processing {split_name} split for training...")

    baseline_samples = []
    clarified_samples = []

    for doc in tqdm(clarifications_data, desc=f"Creating {split_name} samples"):
        # Create baseline samples (without clarifications)
        baseline_samples.extend(create_training_samples(doc, use_clarifications=False))

        # Create clarified samples (with clarifications)
        clarified_samples.extend(create_training_samples(doc, use_clarifications=True))

    print(f"✓ Created {len(baseline_samples)} baseline samples")
    print(f"✓ Created {len(clarified_samples)} clarified samples")

    return baseline_samples, clarified_samples





def load_samples(filename):
    """
    Load training samples from JSONL file.
    
    Reads line-delimited JSON file containing T5 training samples with
    input_text and target_text fields.
    
    Args:
        filename: Path to JSONL file (string)
    
    Returns:
        List of sample dictionaries, each containing:
        - input_text: Formatted input for T5 model
        - target_text: Expected output (entity QID)
    """
    """Load JSONL samples."""
    samples = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples