import pandas as pd
import json
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)

CONFIG = {
    'clarifications_dir': os.path.join(WORKSPACE_ROOT, 'AIDA', 'clarifications'),
    'output_dir': os.path.join(WORKSPACE_ROOT, 'AIDA', 'experiments', 't5_models'),
    'predictions_dir': os.path.join(WORKSPACE_ROOT, 'AIDA', 'experiments', 't5_models', 'predictions'),
    'model_name': 't5-small',
    'max_length': 512,
    'context_window': 250,
    'train_epochs': 10,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'eval_batch_size': 16,
    'device': 'cpu'  # Default to CPU, can be overridden
}

def create_training_samples(doc, use_clarifications=False):
    """
    Create T5 training samples from a document with entity linking annotations.
    
    Args:
        doc: Dictionary with 'text', 'entities', and 'clarifications'
        use_clarifications: Whether to include clarifications in input
    
    Returns:
        List of training sample dictionaries with 'input_text' and 'target_text'
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
        
        # Clean QID
        qid_clean = str(qid).replace('.0', '')
        
        # Get context around entity
        context_window = CONFIG['context_window']
        context_left = text[max(0, start - context_window):start]
        context_right = text[end:min(len(text), end + context_window)]
        
        # Build marked entity
        if use_clarifications and mention in clarifications:
            clarification = clarifications[mention]
            marked_entity = f"[START_ENT] {mention} [END_ENT] [CLARIFY: {clarification}]"
        else:
            marked_entity = f"[START_ENT] {mention} [END_ENT]"
        
        # Create input with task prefix
        input_text = f"link entity: {context_left}{marked_entity}{context_right}"
        
        # Truncate if too long
        if len(input_text) > CONFIG['max_length']:
            input_text = input_text[:CONFIG['max_length']]
        
        # Target format: Q + QID
        target_text = f"Q{qid_clean}"
        
        samples.append({
            'input_text': input_text,
            'target_text': target_text
        })
    
    return samples


def process_split_for_training(clarifications_data, split_name):
    """
    Convert clarification data to T5 training format.
    
    Args:
        clarifications_data: List of document dictionaries
        split_name: String identifier ('train', 'val', or 'test')
    
    Returns:
        Tuple of (baseline_samples, clarified_samples)
    """
    print(f"\n[*] Processing {split_name} split for training...")
    
    baseline_samples = []
    clarified_samples = []
    
    for doc in tqdm(clarifications_data, desc=f"Creating {split_name} samples"):
        baseline_samples.extend(create_training_samples(doc, use_clarifications=False))
        clarified_samples.extend(create_training_samples(doc, use_clarifications=True))
    
    print(f"[OK] Created {len(baseline_samples)} baseline samples")
    print(f"[OK] Created {len(clarified_samples)} clarified samples")
    
    return baseline_samples, clarified_samples


def save_samples(samples, filename):
    """Save training samples to JSONL file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"[OK] Saved: {filename}")


def load_samples(filename):
    """Load training samples from JSONL file."""
    samples = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples



# DATASET CLASS

class EntityLinkingDataset(Dataset):
    """PyTorch Dataset for entity linking with T5."""
    
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample['input_text'],
            text_target=sample['target_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {key: val.squeeze() for key, val in encoding.items()}


# TRAINING FUNCTIONS
def train_model(model, tokenizer, train_dataset, val_dataset, output_path, model_type='baseline'):
    """
    Train a T5 model for entity linking.
    
    Args:
        model: T5ForConditionalGeneration model
        tokenizer: T5Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_path: Path to save model
        model_type: 'baseline' or 'clarified'
    
    Returns:
        Trained Trainer object
    """
    print(f"\n{'='*70}")
    print(f"TRAINING {model_type.upper()} MODEL")
    print(f"{'='*70}")
    
    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        num_train_epochs=CONFIG['train_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=2,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        report_to="none",
        fp16=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n[OK] {model_type.capitalize()} model saved to: {output_path}")
    
    return trainer


def plot_training_metrics(trainer, output_path, model_type='baseline'):
    """Plot and save training metrics."""
    log_history = trainer.state.log_history
    logs_df = pd.DataFrame(log_history)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle(f'{model_type.capitalize()} Model Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    if 'loss' in logs_df.columns:
        train_logs = logs_df[logs_df['loss'].notna()]
        axes[0].plot(train_logs['step'], train_logs['loss'], 'b-', linewidth=2, label='Training Loss')
        axes[0].set_xlabel('Steps', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    
    # Plot 2: Learning Rate
    if 'learning_rate' in logs_df.columns:
        lr_logs = logs_df[logs_df['learning_rate'].notna()]
        axes[1].plot(lr_logs['step'], lr_logs['learning_rate'], 'g-', linewidth=2, label='Learning Rate')
        axes[1].set_xlabel('Steps', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_path, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Training plot saved to: {plot_path}")


# EVALUATION FUNCTIONS

def generate_predictions(model, samples, tokenizer, batch_size=16):
    """Generate predictions for test samples."""
    predictions = []
    ground_truths = []
    
    dataset = EntityLinkingDataset(samples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device(CONFIG['device'])
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=20,
                num_beams=5,
                early_stopping=True
            )
            
            pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            gt_texts = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            
            predictions.extend(pred_texts)
            ground_truths.extend(gt_texts)
    
    return predictions, ground_truths


def calculate_accuracy(predictions, ground_truths):
    """Calculate exact match accuracy."""
    correct = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truths):
        pred_norm = pred.strip().replace(" ", "")
        gt_norm = gt.strip().replace(" ", "")
        
        if pred_norm == gt_norm:
            correct += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, correct, total


def perform_error_analysis(baseline_preds, clarified_preds, ground_truths, test_samples):
    """Perform detailed error analysis."""
    both_correct = 0
    both_incorrect = 0
    only_baseline_correct = 0
    only_clarified_correct = 0
    
    clarified_improvements = []
    baseline_advantages = []
    
    for i, (b_pred, c_pred, gt) in enumerate(zip(baseline_preds, clarified_preds, ground_truths)):
        b_pred_norm = b_pred.strip().replace(" ", "")
        c_pred_norm = c_pred.strip().replace(" ", "")
        gt_norm = gt.strip().replace(" ", "")
        
        b_correct = (b_pred_norm == gt_norm)
        c_correct = (c_pred_norm == gt_norm)
        
        if b_correct and c_correct:
            both_correct += 1
        elif not b_correct and not c_correct:
            both_incorrect += 1
        elif b_correct and not c_correct:
            only_baseline_correct += 1
            baseline_advantages.append({
                'index': i,
                'input': test_samples[i]['input_text'][:200] + '...',
                'baseline_pred': b_pred,
                'clarified_pred': c_pred,
                'ground_truth': gt
            })
        else:
            only_clarified_correct += 1
            clarified_improvements.append({
                'index': i,
                'input': test_samples[i]['input_text'][:200] + '...',
                'baseline_pred': b_pred,
                'clarified_pred': c_pred,
                'ground_truth': gt
            })
    
    return {
        'both_correct': both_correct,
        'both_incorrect': both_incorrect,
        'only_baseline_correct': only_baseline_correct,
        'only_clarified_correct': only_clarified_correct,
        'clarified_improvements': clarified_improvements,
        'baseline_advantages': baseline_advantages
    }


def plot_evaluation_results(baseline_acc, clarified_acc, baseline_correct, clarified_correct, baseline_total, output_dir):
    """Create visualization of evaluation results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Test Set Evaluation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy Comparison
    models = ['Baseline', 'Clarified']
    accuracies = [baseline_acc, clarified_acc]
    colors = ['#3498db', '#e74c3c']
    
    bars = axes[0].bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Correct vs Incorrect
    categories = ['Baseline', 'Clarified']
    correct_counts = [baseline_correct, clarified_correct]
    incorrect_counts = [baseline_total - baseline_correct, baseline_total - clarified_correct]
    
    x = range(len(categories))
    width = 0.35
    
    bars1 = axes[1].bar([i - width/2 for i in x], correct_counts, width,
                        label='Correct', color='#2ecc71', alpha=0.7, edgecolor='black')
    bars2 = axes[1].bar([i + width/2 for i in x], incorrect_counts, width,
                        label='Incorrect', color='#e67e22', alpha=0.7, edgecolor='black')
    
    axes[1].set_ylabel('Number of Predictions', fontsize=12)
    axes[1].set_title('Correct vs Incorrect Predictions', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'test_evaluation_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Evaluation plot saved to: {plot_path}")
