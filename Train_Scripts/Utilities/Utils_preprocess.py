import pandas as pd
import numpy as np
from tqdm import tqdm


# Step 1: Add context to entities

def extract_mention_with_context(text, start, end, context_window=200):
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



def apply_add_context(df, context_window=200, inplace=False):
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



# Step 2: Normalize Mentions

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




# Step 3: Remove Overlapping Entities

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
