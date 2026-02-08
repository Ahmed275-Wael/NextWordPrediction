# ProjectNextWord: Word-Level Shakespeare Text Generation

A Transformer-based language model for generating Shakespeare-style text using word-level tokenization with fine-tuned FastText embeddings and semantic anchor initialization.

## 🎯 Research Question

> *"Can we effectively combine pre-trained word embeddings with semantic anchor initialization to enable word-level language modeling on Early Modern English text, and how does this compare to character-level approaches?"*

## 📁 Project Structure

```
ProjectNextWord/
├── config.py           # Configuration settings and hyperparameters
├── utils.py            # Utility functions (checkpointing, early stopping)
├── data_loader.py      # Data loading, tokenization, FastText alignment
├── embeddings.py       # Embedding module with semantic anchors
├── transformer.py      # Multi-head attention and Transformer decoder
├── model.py            # Complete model combining embeddings + transformer
├── train.py            # Training pipeline with validation
├── main.py             # Entry point - run training/generation/evaluation
├── requirements.txt    # Python dependencies
├── data/               # Downloaded data and FastText model
├── models/             # Saved model checkpoints
└── logs/               # Training logs and plots
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ProjectNextWord
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python main.py --mode train --epochs 50
```

### 3. Generate Text

```bash
python main.py --mode generate --prompt "to be or not to be" --temperature 0.8
```

### 4. Evaluate

```bash
python main.py --mode evaluate
```

### 5. Run Everything

```bash
python main.py --mode all
```

## 🏗️ Architecture

### Embedding Layer
- **FastText-initialized**: Pre-trained 300d embeddings
- **Semantic Anchors**: Archaic words mapped to modern equivalents
  - `thou` → `you`, `hath` → `has`, `wherefore` → `why`, etc.
- **Differential Learning Rates**: Lower LR for embeddings, higher for rest

### Transformer Decoder
- **Layers**: 6 decoder blocks
- **Attention**: 6 heads, causal (autoregressive) masking
- **FFN**: 1200 hidden dimension with GELU activation
- **Normalization**: Pre-LayerNorm for stable training

### Training Features
- Label smoothing (0.1)
- Semantic anchor preservation loss
- Cosine learning rate schedule with warmup
- Gradient clipping
- Early stopping

## 📊 Configuration

Key settings in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EMBEDDING_DIM` | 300 | FastText embedding dimension |
| `NUM_LAYERS` | 6 | Transformer decoder layers |
| `NUM_HEADS` | 6 | Attention heads |
| `FFN_HIDDEN_DIM` | 1200 | Feed-forward hidden size |
| `MAX_SEQ_LENGTH` | 64 | Maximum sequence length (words) |
| `BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 1e-3 | Main learning rate |
| `EMBEDDING_LR` | 1e-4 | Embedding learning rate |

## 🔬 Key Innovation: Semantic Anchors

The challenge with word-level Shakespeare modeling is vocabulary mismatch:
- Pre-trained embeddings don't know archaic words
- Random initialization requires extensive training

**Solution**: Initialize archaic words near their modern equivalents:

```python
SEMANTIC_ANCHORS = {
    'thou': 'you',      # Pronouns
    'hath': 'has',      # Verbs
    'wherefore': 'why', # Adverbs
    "'tis": 'it_is',    # Contractions
    ...
}
```

This provides:
1. Meaningful starting point for OOV words
2. Preserved semantic relationships
3. Faster convergence

## 📈 Expected Results

| Metric | Expected Value |
|--------|----------------|
| Test Perplexity | 80-150 |
| Test Accuracy | 25-35% |
| Training Time | 2-4 hours (GPU) |

## 🎭 Sample Generations

```
Prompt: "to be or not to be"
Generated: "to be or not to be, that is the question:
whether 'tis nobler in the mind to suffer
the slings and arrows of outrageous fortune..."

Prompt: "the king"
Generated: "the king hath sent for you, and you must
go with all convenient speed to court..."
```

## 📚 File Descriptions

### `config.py`
Central configuration file with all hyperparameters, paths, and semantic anchor mappings.

### `data_loader.py`
- Downloads Shakespeare text
- Word-level tokenization (handles contractions like `'tis`, `o'er`)
- Vocabulary building with frequency filtering
- FastText embedding alignment
- PyTorch DataLoader creation

### `embeddings.py`
- `ShakespeareEmbedding`: Embedding with positional encoding
- `SemanticAnchorLoss`: Auxiliary loss for preserving archaic-modern relationships
- `EmbeddingWithTiedWeights`: Weight tying between input and output

### `transformer.py`
- `MultiHeadSelfAttention`: Scaled dot-product attention with multiple heads
- `FeedForwardNetwork`: Position-wise FFN with GELU
- `TransformerDecoderBlock`: Single block (attention + FFN + residuals)
- `TransformerDecoder`: Full stack of N decoder blocks

### `model.py`
- `ShakespeareTransformer`: Complete model combining all components
- `TextGenerator`: Generation utilities (temperature, top-k, top-p sampling)

### `train.py`
- `Trainer`: Training loop with validation, checkpointing, early stopping
- Differential learning rates for embeddings vs. rest
- Learning rate scheduling with warmup

### `main.py`
Entry point with CLI interface for training, generation, and evaluation.

## 🔧 Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --mode {train,generate,evaluate,all}
                        Mode to run (default: all)
  --seed INT            Random seed (default: 42)
  --epochs INT          Number of training epochs (default: 50)
  --checkpoint PATH     Path to checkpoint file
  --prompt TEXT         Prompt for generation (default: "to be or not to be")
  --temperature FLOAT   Sampling temperature (default: 0.8)
  --max_length INT      Maximum generation length (default: 100)
```

## 📝 For Research Paper

This implementation supports the paper:

> **"Bridging Centuries: Fine-Tuning Word Embeddings for Early Modern English Text Generation"**

Key contributions:
1. Semantic anchor initialization for OOV archaic vocabulary
2. Differential learning rates preserving pre-trained knowledge
3. Comparative analysis vs. character-level approaches

## 🙏 Acknowledgments

- FastText pre-trained embeddings by Facebook Research
- Shakespeare text from [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017)
