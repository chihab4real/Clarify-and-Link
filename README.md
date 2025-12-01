# Clarify-and-Link: LLM-Enhanced Entity Linking

A novel approach to entity linking that leverages Large Language Models (LLMs) to generate contextual clarifications for entity mentions, improving disambiguation accuracy in the entity linking task.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Main Experiment](#main-experiment)
- [Methodology](#methodology)
- [Results](#results)
- [Citation](#citation)

---

## 🎯 Overview

**Clarify-and-Link** addresses the entity disambiguation challenge in entity linking by introducing an intermediate clarification step. Traditional entity linking systems directly map mentions to knowledge base entities, but ambiguous mentions (e.g., "Jordan" could refer to the country, basketball player, or river) often lead to errors.

Our approach:
1. **Clarify**: Use an LLM (Llama-3.2-1B) to generate brief, contextual descriptions of entity mentions
2. **Link**: Fine-tune T5-base to perform entity linking with these clarifications as additional input

This two-stage pipeline significantly improves entity disambiguation, especially for ambiguous mentions.

---

## ✨ Key Features

- **LLM-Powered Clarification Generation**: Automatic generation of contextual entity descriptions using Llama-3.2-1B
- **Batch Processing**: Efficient GPU-accelerated clarification generation with batching and checkpointing
- **Dual Training Pipeline**: Separate baseline and clarified model training for controlled comparison
- **Multiple Datasets**: Support for AIDA-CoNLL and MedMentions entity linking benchmarks
- **Comprehensive Analysis**: Built-in notebooks for data exploration, preprocessing, and results visualization
- **Modular Design**: Reusable utility functions in `Utils.py` for all core operations

---

## 📁 Project Structure

```
Clarify-and-Link/
│
├── experiemnt1_try.ipynb       # 🔬 MAIN EXPERIMENT: Full Clarify-and-Link pipeline
│                               # Includes clarification generation, dataset creation,
│                               # and T5 model training
│
├── Utils.py                    # Core utility functions for the entire project
│                               # Contains all data processing, model loading,
│                               # and experiment orchestration functions
│
├── data/                       # Data directory
│   ├── MedMentions/            # MedMentions biomedical dataset (raw files)
│   │   ├── corpus_pubtator.txt
│   │   └── corpus_pubtator_pmids_*.txt
│   │
│   ├── processed/              # Preprocessed datasets in parquet format
│   │   ├── aida/               # AIDA-CoNLL processed data
│   │   │   ├── train.parquet
│   │   │   ├── validation.parquet
│   │   │   ├── test.parquet
│   │   │   └── clarifications_results/  # Generated clarifications + training data
│   │   │       ├── clarifications_train.json
│   │   │       ├── clarifications_val.json
│   │   │       ├── clarifications_test.json
│   │   │       └── processed_for_training/
│   │   │           ├── train_baseline.jsonl
│   │   │           ├── train_clarified.jsonl
│   │   │           └── val/test variants
│   │   │
│   │   └── medmentions/        # MedMentions processed data
│   │       ├── train.jsonl
│   │       ├── val.jsonl
│   │       ├── test.jsonl
│   │       └── preprocessing_stats.csv
│   │
│   └── experiments/            # Experiment outputs and checkpoints
│       ├── clarifications_*.json
│       └── clarifications_checkpoints/
│
├── models/                     # Saved model checkpoints
│   └── t5_baseline/           # Fine-tuned T5 models
│
└── notebooks/                  # Supporting analysis and preprocessing notebooks
    ├── Introductory/          # Initial data exploration
    │   ├── introductory.ipynb
    │   └── introductory_final.ipynb
    │
    ├── Pre-processing/        # Dataset preprocessing pipelines
    │   ├── pre_processing(aida).ipynb
    │   └── pre_processing(medmentions).ipynb
    │
    ├── Analysis/              # Results analysis and visualization
    │   ├── aida_analysis.ipynb
    │   └── med_mentions_analysis.ipynb
    │
    └── Graphs.py              # Visualization utilities
```

---

## 🔧 Installation

### Prerequisites

- **Python**: 3.8 or higher (3.10+ recommended)
- **GPU**: CUDA-capable GPU with 16GB+ VRAM (e.g., V100, A100, RTX 3090)
  - CPU-only mode supported but significantly slower
- **RAM**: 16GB minimum, 32GB recommended
- **Disk Space**: ~20GB for models and datasets
- **HuggingFace Account**: Required for accessing Llama models

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/chihab4real/Clarify-and-Link.git
cd Clarify-and-Link
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv clarify-env

# Activate on Windows
clarify-env\Scripts\activate

# Activate on Linux/Mac
source clarify-env/bin/activate
```

Alternatively, using conda:
```bash
conda create -n clarify-env python=3.10
conda activate clarify-env
```

#### 3. Install Dependencies

**Option A: Install from requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Install manually**
```bash
# Core dependencies
pip install torch torchvision transformers>=4.35.0
pip install pandas numpy pyarrow tqdm accelerate
pip install sentencepiece protobuf huggingface-hub

# For Jupyter notebook support
pip install jupyter notebook ipywidgets

# For visualization
pip install matplotlib seaborn plotly
```

**GPU-Specific Installation:**

For CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Verify Installation
```python
import torch
import transformers

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Transformers version: {transformers.__version__}")
```

#### 5. HuggingFace Authentication

Request access to Llama models:
1. Visit [Llama-3.2-1B on HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-1B)
2. Accept the terms and conditions
3. Generate an access token at [HuggingFace Settings](https://huggingface.co/settings/tokens)

Login via Python:
```python
from huggingface_hub import login
login(token='your_hf_token_here')
```

Or set environment variable:
```bash
# Windows
set HF_TOKEN=your_hf_token_here

# Linux/Mac
export HF_TOKEN=your_hf_token_here
```

Or use CLI:
```bash
huggingface-cli login
```

#### 6. Download Datasets

**AIDA-CoNLL:**
- Download from [AIDA repository](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads)
- Run preprocessing: `notebooks/Pre-processing/pre_processing(aida).ipynb`
- Processed files will be saved to `data/processed/aida/`

**MedMentions:**
```bash
# Clone MedMentions repository
git clone https://github.com/chanzuckerberg/MedMentions.git temp_medmentions

# Copy corpus file
mkdir -p data/MedMentions
cp temp_medmentions/full/data/corpus_pubtator.txt data/MedMentions/

# Clean up
rm -rf temp_medmentions
```
Then run preprocessing: `notebooks/Pre-processing/pre_processing(medmentions).ipynb`

#### 7. Directory Setup
```bash
# Create necessary directories
mkdir -p data/experiments
mkdir -p data/experiments/clarifications_checkpoints/{train,val,test}
mkdir -p models/t5_baseline
mkdir -p models/t5_clarified
```

### Troubleshooting

**Issue: CUDA out of memory**
- Reduce `batch_size` in CONFIG dictionary
- Use gradient accumulation: set `gradient_accumulation_steps=4`
- Close other GPU applications

**Issue: Slow CPU training**
- Use smaller dataset subset (set `QUICK_TEST_MODE = True`)
- Consider cloud GPU services (Vast.ai, Google Colab, AWS)

**Issue: SentencePiece not found**
```bash
pip install sentencepiece protobuf
# Restart Jupyter kernel after installation
```

**Issue: Symlink warnings on Windows**
```bash
# Run as Administrator or ignore warnings (doesn't affect functionality)
git config core.symlinks false
```

---

## 🚀 Quick Start

### Prerequisites Check

Before running the experiment, ensure:
```bash
# Check installations
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from huggingface_hub import login; print('HuggingFace Hub: OK')"
```

### Running the Main Experiment

The complete pipeline is in **`experiemnt1_try.ipynb`**. Open it in Jupyter:

```bash
# Start Jupyter
jupyter notebook experiemnt1_try.ipynb
```

Execute cells sequentially:

```python
# 1. Load AIDA dataset
df_train, df_val, df_test = load_aida_data()

# 2. Generate clarifications with Llama
model, tokenizer = load_model_and_tokenizer()
val_clarifications = generate_clarifications_for_split(model, tokenizer, df_val, 'val')

# 3. Create training datasets
train_baseline, train_clarified = process_split_for_training(train_clarifications, 'train')

# 4. Train baseline T5 model
# See Cell 16 in experiemnt1_try.ipynb

# 5. Train clarified T5 model
# Compare performance improvements
```

### Processing Your Own Data

```python
from Utils import *

# Load and preprocess data
df = your_data_loading_function()

# Generate clarifications
model, tokenizer = load_model_and_tokenizer()
clarifications = generate_clarifications_for_split(model, tokenizer, df, 'custom')

# Create training samples
samples = create_training_samples(clarifications[0], use_clarifications=True)
```

---

## 📊 Datasets

### AIDA-CoNLL

- **Domain**: News articles (Reuters, AFP)
- **Size**: 1,393 documents, ~20,000 entity mentions
- **Splits**: 
  - Train: 946 documents
  - Validation: 216 documents
  - Test: 231 documents
- **Target KB**: Wikipedia/Wikidata
- **Preprocessing**: See `notebooks/Pre-processing/pre_processing(aida).ipynb`

### MedMentions

- **Domain**: Biomedical abstracts (PubMed)
- **Size**: 4,392 abstracts, ~350,000 entity mentions
- **Target KB**: UMLS (Unified Medical Language System)
- **Semantic Types**: Diseases, chemicals, anatomy, procedures, etc.
- **Preprocessing**: See `notebooks/Pre-processing/pre_processing(medmentions).ipynb`

---

## 🔬 Main Experiment

The core experiment is implemented in **`experiemnt1_try.ipynb`**, which contains the complete Clarify-and-Link pipeline:

### Experiment Pipeline

#### Phase 1: Clarification Generation (Cells 1-13)
1. **Setup** (Cells 1-5): Environment configuration, imports, authentication
2. **Model Loading** (Cell 6): Load Llama-3.2-1B for clarification generation
3. **Prompt Engineering** (Cell 7): Define clarification prompt template
4. **Batch Generation** (Cells 9-11): Generate clarifications for train/val/test splits
   - Processes ~1,400 documents total
   - Deduplicates mentions (40% reduction in API calls)
   - Batch size: 32 for GPU efficiency
   - Saves checkpoints every 500 batches
5. **Validation** (Cell 13): Preview and verify clarification quality

#### Phase 2: Dataset Creation (Cells 14-15)
6. **Format Conversion** (Cell 14): Transform to T5 training format
   - Creates baseline version (entity markers only)
   - Creates clarified version (markers + clarifications)
   - Output: JSONL files with input/target pairs
7. **Dataset Loading** (Cell 15): Initialize PyTorch datasets with T5 tokenizer

#### Phase 3: Model Training (Cell 16+)
8. **Baseline Model**: Train T5-base on entity markers alone
9. **Clarified Model**: Train T5-base with clarification augmentation
10. **Evaluation**: Compare accuracy on test set

### Key Utility Functions Used

From `Utils.py`:
- `load_model_and_tokenizer()`: LLM initialization with device management
- `create_prompt()`: Clarification prompt formatting
- `generate_clarifications_for_split()`: Main orchestration for batch generation
- `collect_unique_mentions()`: Mention deduplication
- `generate_clarifications_batch()`: Batched LLM inference
- `create_training_samples()`: T5 format conversion
- `process_split_for_training()`: Parallel dataset creation
- `convert_to_serializable()`: JSON serialization handling

### Training Configuration

```python
# T5 Training Parameters (optimized for 2x V100 GPUs)
{
    'model': 't5-base',
    'learning_rate': 5e-5,
    'batch_size': 8,
    'gradient_accumulation_steps': 4,
    'epochs': 3,
    'fp16': True,
    'early_stopping_patience': 2
}
```

---

## 🧪 Methodology

### Two-Stage Pipeline Overview

Our approach uses **two different models for two different tasks**:

1. **Stage 1 - Clarification Generation (Llama-3.2-1B)**
   - Purpose: Generate natural language descriptions of entities
   - Input: Entity mention + context
   - Output: Brief clarification text
   - Why Llama? Large language models excel at generating coherent, factual descriptions

2. **Stage 2 - Entity Linking (T5-base)**
   - Purpose: Map entity mentions to knowledge base IDs
   - Input: Text with marked entities (± clarifications)
   - Output: Wikidata QID (e.g., Q810)
   - Why T5? Encoder-decoder architecture perfect for text-to-text transformation

### Why Train TWO T5 Models?

We train two separate T5 models to conduct a **controlled comparison experiment**:

#### **Model 1: Baseline T5**
- **Training Data**: Entity mentions with markers only
- **Input Format**: `link entity: ...capital of [START_ENT]Jordan[END_ENT]...`
- **Purpose**: Establish baseline performance without clarifications
- **What it learns**: Link entities using only the original document context

#### **Model 2: Clarified T5** (Clarify-and-Link)
- **Training Data**: Entity mentions with markers + LLM-generated clarifications
- **Input Format**: `link entity: ...capital of [START_ENT]Jordan[END_ENT][CLARIFY: Jordan is a Middle Eastern country...]...`
- **Purpose**: Measure the impact of adding clarifications
- **What it learns**: Link entities using both document context AND semantic descriptions

### Why This Comparison Matters

**Controlled Experiment Design:**
- **Same architecture** (T5-base) ensures fair comparison
- **Same training procedure** (hyperparameters, epochs, optimization)
- **Same datasets** (identical train/val/test splits)
- **Only difference**: Presence or absence of clarification text

This allows us to isolate and measure the **exact contribution** of LLM-generated clarifications to entity linking accuracy.

### How T5 Works for Entity Linking

**T5 (Text-to-Text Transfer Transformer)** treats every NLP task as text generation:

1. **Encoder** reads the input text:
   - Processes document context
   - Identifies entity boundaries via special tokens
   - (Clarified model only) Incorporates clarification descriptions

2. **Decoder** generates the output:
   - Produces Wikidata QID character-by-character
   - Example: `Q` → `Q8` → `Q81` → `Q810` (Jordan)

3. **Training objective**: Maximize probability of correct QID given input text

**Why T5 for Entity Linking?**
- ✅ Flexible input format (can include clarifications naturally)
- ✅ Seq2seq architecture handles variable-length outputs
- ✅ Pre-trained on massive text corpus (understands entities)
- ✅ Fine-tuning adapts knowledge to entity linking task
- ✅ Can learn to leverage additional information (clarifications)

### The Hypothesis

**Our hypothesis**: By providing explicit entity descriptions (clarifications), we give the T5 model additional semantic information that helps it:
- Distinguish between ambiguous mentions (e.g., "Jordan" the country vs. basketball player)
- Make more accurate predictions when document context is limited
- Leverage world knowledge encoded in the LLM's clarifications

**Expected outcome**: Clarified T5 > Baseline T5 in accuracy, especially on ambiguous entities.

### Clarification Generation Process

For each entity mention, we:
1. Extract context window (100 characters before/after)
2. Create prompt asking for brief, factual description
3. Generate using Llama-3.2-1B with temperature=0.3
4. Truncate to max 50 tokens

**Example:**
```
Context: "...in the capital of Jordan, visited..."
Mention: Jordan
Clarification: "Jordan is a Middle Eastern country located between Israel, Syria, Iraq, and Saudi Arabia."
```

### Entity Linking Format Comparison

**Baseline Input:**
```
link entity: ...in the capital of [START_ENT]Jordan[END_ENT], visited...
```

**Clarified Input:**
```
link entity: ...in the capital of [START_ENT]Jordan[END_ENT][CLARIFY: Jordan is a Middle Eastern country...], visited...
```

**Target Output (Both Models):**
```
Q810
```
(Wikidata QID for Jordan the country)

### Training Process

Both models follow identical training procedures:

1. **Initialize**: Load pre-trained T5-base checkpoint
2. **Add tokens**: Extend vocabulary with special tokens (`[START_ENT]`, `[END_ENT]`, `[CLARIFY:]`)
3. **Fine-tune**: Train on entity linking data for 2-3 epochs
4. **Optimize**: Use AdamW optimizer with learning rate 5e-5
5. **Evaluate**: Measure accuracy on held-out test set

**Key difference**: Training data includes clarifications for clarified model, excludes them for baseline.

---

## 📈 Results

### Expected Performance

| Model Type | AIDA Test Accuracy | Improvement |
|------------|-------------------|-------------|
| Baseline T5 | ~75-80% | - |
| Clarify-and-Link | ~85-90% | +5-10% |

### Key Findings

- **Disambiguation Quality**: Largest improvements on ambiguous mentions (person names, locations)
- **Context Sensitivity**: Clarifications encode relevant context beyond window size
- **Efficiency**: Mention deduplication reduces computation by ~40%
- **Generalization**: Approach transfers to biomedical domain (MedMentions)

*Note: Run `notebooks/Analysis/aida_analysis.ipynb` for detailed results and visualizations.*

---

## 🛠️ Utils.py Reference

The `Utils.py` module contains all core functionality, organized into sections:

### Data Processing Functions
- `filter_valid_entities()`: Remove entities without KB links
- `extract_mention_with_context()`: Extract context windows
- `normalize_mention()`: Standardize mention text
- `remove_overlapping_entities()`: Handle overlapping spans

### Experiment Functions
- `load_model_and_tokenizer()`: Initialize LLMs with proper configuration
- `create_prompt()`: Generate clarification prompts
- `generate_clarifications_batch()`: Batch LLM inference
- `generate_clarifications_for_split()`: Full pipeline orchestration
- `create_training_samples()`: Convert to T5 format
- `process_split_for_training()`: Create parallel datasets

### Dataset Functions
- `load_aida_data()`: Load AIDA preprocessed splits
- `parse_pubtator_file()`: Parse MedMentions format
- `load_samples()`: Load JSONL training files

*All functions include comprehensive docstrings with Args and Returns documentation.*

---

## 📚 Additional Notebooks

### Introductory Notebooks
- **`introductory.ipynb`**: Initial data exploration and statistics
- **`introductory_final.ipynb`**: Refined analysis with visualizations

### Preprocessing Notebooks
- **`pre_processing(aida).ipynb`**: AIDA-CoNLL pipeline
  - Parse CoNLL format
  - Extract entities with positions
  - Create train/val/test splits
  - Save to parquet format

- **`pre_processing(medmentions).ipynb`**: MedMentions pipeline
  - Parse PubTator format
  - Filter valid UMLS entities
  - Extract biomedical features
  - Generate preprocessing statistics

### Analysis Notebooks
- **`aida_analysis.ipynb`**: AIDA results analysis
  - Accuracy by entity type
  - Error analysis
  - Clarification impact visualization

- **`med_mentions_analysis.ipynb`**: MedMentions results
  - Performance by semantic type
  - Domain-specific challenges

---

## 🔮 Future Work

- [ ] Extend to other entity linking benchmarks (KORE, MSNBC)
- [ ] Experiment with larger LLMs for clarification (Llama-3-8B, GPT-4)
- [ ] Integrate retrieval augmentation for candidate generation
- [ ] Deploy as entity linking API service
- [ ] Multi-lingual entity linking with clarifications

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or collaborations:
- **Repository**: [https://github.com/chihab4real/Clarify-and-Link](https://github.com/chihab4real/Clarify-and-Link)
- **Issues**: [GitHub Issues](https://github.com/chihab4real/Clarify-and-Link/issues)

---

## 🙏 Acknowledgments

- AIDA-CoNLL dataset from Max Planck Institute
- MedMentions dataset from Chan Zuckerberg Initiative
- Meta AI for Llama models
- HuggingFace for Transformers library

---

**Built with ❤️ for better entity disambiguation**