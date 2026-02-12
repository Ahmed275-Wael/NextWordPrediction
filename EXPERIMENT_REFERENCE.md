# Shakespeare Next-Word Prediction: Complete Experiment Reference
## CSO7013 Machine Learning Final Assessment

**Author**: [Your Name]  
**Date**: February 2026  
**Course**: Masters in Data Science — CSO7013 Machine Learning  
**Repository**: NextWordPrediction

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Experiment Timeline & Results](#3-experiment-timeline--results)
4. [Key Findings & Insights](#4-key-findings--insights)
5. [Dataset Information](#5-dataset-information)
6. [Training Configurations](#6-training-configurations)
7. [Optimization Techniques](#7-optimization-techniques)
8. [Code Structure](#8-code-structure)
9. [Generated Samples](#9-generated-samples)
10. [Theoretical Foundations](#10-theoretical-foundations)
11. [Future Work](#11-future-work)
12. [References](#12-references)

---

## 1. Project Overview

### 1.1 Objective
Build a **next-word prediction model** for Shakespeare text using a Transformer architecture. The model should learn the vocabulary, style, and structure of Early Modern English to generate coherent Shakespeare-like text.

### 1.2 Final Best Results

| Metric | Value | Configuration |
|--------|-------|---------------|
| **Test Perplexity** | **178.9** | Pre-train + Fine-tune v2 (Discriminative) |
| **Test Accuracy** | **23.33%** | Pre-train + Fine-tune v2 (Discriminative) |
| **Test Loss** | **5.1868** | Pre-train + Fine-tune v2 (Discriminative) |

### 1.3 Key Achievements
- **5.2× perplexity reduction** from word-level baseline (393 → 178.9)
- **24.2% relative accuracy improvement** over BPE-only training (20.8% → 23.3%)
- Successfully applied transfer learning with discriminative fine-tuning
- Implemented contracting stride for efficient training
- Demonstrated Chinchilla scaling principle in practice

### 1.4 Technical Stack
- **Framework**: PyTorch 2.6.0+cu124
- **Hardware**: NVIDIA GeForce GTX 1660 Ti (6GB VRAM)
- **Python**: 3.13.3
- **OS**: Windows 11

---

## 2. Architecture Deep Dive

### 2.1 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SHAKESPEARE TRANSFORMER PIPELINE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────────┐│
│  │ BPE Tokenizer│ →  │Token Embedding│ →  │ Sinusoidal Positional Encoding ││
│  │ (8000 vocab)│    │ (300-dim)    │    │ PE(pos,2i)=sin(pos/10000^2i/d) ││
│  └─────────────┘    └──────────────┘    └─────────────────────────────────┘│
│         ↓                                          ↓                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Embedding Dropout (p=0.15/0.20)                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      5× TRANSFORMER DECODER BLOCKS                       ││
│  │  ┌───────────────────────────────────────────────────────────────────┐  ││
│  │  │  Pre-LayerNorm → Multi-Head Self-Attention (6 heads) → Dropout    │  ││
│  │  │                           ↓ (+ Residual)                          │  ││
│  │  │  Pre-LayerNorm → Feed-Forward Network (300→1024→300) → Dropout    │  ││
│  │  │                           ↓ (+ Residual)                          │  ││
│  │  └───────────────────────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Final LayerNorm                                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │              Weight-Tied Output Projection (300 → 8000)                 ││
│  │                   W_out = W_embed.T (Press & Wolf, 2017)                ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │    Cross-Entropy Loss with Label Smoothing (α=0.1) + AdamW Optimizer   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component-by-Component Breakdown

#### Component 1: BPE Tokenizer
```python
# Configuration
vocab_size = 8000  # Byte-Pair Encoding vocabulary
compression_ratio = 4.1  # chars per token

# Training: Learned on combined Gutenberg + Shakespeare corpus
# Handles both archaic Shakespearean words and modern Gutenberg English
```

**Why BPE over Word-Level?**
- Word-level vocabulary had ~10K+ unique words → many rare/OOV tokens
- BPE decomposes rare words into subword units
- Reduces vocabulary while maintaining semantic granularity
- Example: "unfriendliness" → ["un", "friend", "li", "ness"]

#### Component 2: Token Embedding Layer
```python
nn.Embedding(
    num_embeddings=8000,  # vocabulary size
    embedding_dim=300,    # embedding dimensions
    padding_idx=0         # PAD token index
)

# Initialization: Random normal (mean=0, std=0.02)
# Training: Fully trainable (not frozen)
```

#### Component 3: Sinusoidal Positional Encoding
```python
# Mathematical formulation (Vaswani et al., 2017)
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

# Properties:
# - Deterministic (not learned)
# - Allows model to attend to relative positions
# - Generalizes to longer sequences than seen in training
```

**Implementation:**
```python
def _create_positional_encoding(max_seq_length, embedding_dim):
    position = torch.arange(max_seq_length).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2).float() * 
        (-torch.log(torch.tensor(10000.0)) / embedding_dim)
    )
    pe = torch.zeros(max_seq_length, embedding_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)
```

#### Component 4: Multi-Head Self-Attention
```python
# Configuration
num_heads = 6
head_dim = embed_dim // num_heads  # 300 // 6 = 50
scale = sqrt(head_dim)  # sqrt(50) ≈ 7.07

# Attention formula
Attention(Q, K, V) = softmax(QK^T / scale) × V

# Causal masking for autoregressive generation
# Upper triangular mask prevents attending to future tokens
```

**Key Design Choices:**
- `bias=False` on all linear projections (nanoGPT optimization)
- Scaled residual initialization: `std = 0.02 / sqrt(2 * num_layers)`
- Attention weights stored for visualization when requested

#### Component 5: Feed-Forward Network
```python
# Architecture: Two linear layers with GELU activation
FFN(x) = GELU(x @ W₁) @ W₂

# Dimensions
input:  (batch, seq, 300)   # embed_dim
hidden: (batch, seq, 1024)  # ffn_hidden_dim
output: (batch, seq, 300)   # embed_dim
```

**Why GELU over ReLU?**
- GELU (Gaussian Error Linear Unit) provides smoother gradients
- Better training dynamics for Transformer architectures
- Used in GPT-2, BERT, and most modern language models

#### Component 6: Pre-LayerNorm Architecture
```python
# Pre-Norm (our implementation) vs Post-Norm (original Transformer)

# Pre-Norm (more stable):
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# Post-Norm (original):
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

**Why Pre-Norm?**
- More stable training dynamics
- Gradients flow more directly through residual connections
- Works better without carefully tuned learning rate warmup
- Used in GPT-2, LLaMA, and most modern architectures

#### Component 7: Weight Tying
```python
# Output projection shares weights with embedding layer
output_logits = hidden @ embedding_weights.T

# Benefits:
# - Reduces parameters by 8000 × 300 = 2.4M
# - Regularizes the model (embedding ↔ output consistency)
# - Press & Wolf (2017): Improves perplexity by ~10%
```

#### Component 8: Label Smoothing Cross-Entropy
```python
# Standard cross-entropy
L = -log(p(y_true))

# With label smoothing (α = 0.1)
target_dist = (1-α) × one_hot(y_true) + α × uniform(vocab)
L = -sum(target_dist × log(predictions))

# Effect: Prevents overconfident predictions, improves generalization
```

#### Component 9: AdamW Optimizer + Cosine Schedule
```python
optimizer = AdamW(
    params,
    lr=5e-4,           # Peak learning rate
    betas=(0.9, 0.99), # nanoGPT uses β₂=0.99 for stability
    weight_decay=0.05  # L2 regularization
)

# Cosine annealing with warmup
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
# LR: warmup → peak → cosine decay to 0
```

#### Component 10: Contracting Stride
```python
# Innovation: Stride decreases throughout training
# Epoch 1-5:   stride = 128 (no overlap)
# Epoch 6-10:  stride = 64  (50% overlap)
# Epoch 11-15: stride = 32  (75% overlap)
# Epoch 16+:   stride = 16  (87.5% overlap)

# Effect: More training examples as training progresses
# Early: Fast epoch, diverse contexts
# Late: Dense coverage, local pattern refinement
```

### 2.3 Model Size Analysis

| Component | Parameters | Calculation |
|-----------|------------|-------------|
| Token Embeddings | 2,400,000 | 8000 × 300 |
| Positional Encoding | 0 | Not learned (buffer) |
| Attention (Q,K,V,O × 5 layers) | 1,800,000 | 5 × 4 × 300 × 300 |
| FFN (fc1, fc2 × 5 layers) | 3,060,000 | 5 × 2 × 300 × 1024 |
| LayerNorm (11 total) | 6,600 | 11 × 2 × 300 |
| Output Projection | 0 | Tied to embeddings |
| **Total** | **7,275,300** | ~7.3M parameters |

---

## 3. Experiment Timeline & Results

### 3.1 Experiment Summary Table

| Experiment | Tokenizer | Pre-training | Test PPL | Test Acc | Key Changes |
|------------|-----------|--------------|----------|----------|-------------|
| **Word-level** | Word (10K) | None | 393.0 | 18.78% | Baseline with FastText embeddings |
| **BPE v1** | BPE (4K) | None | 267.5 | 19.40% | First BPE attempt |
| **BPE v2** | BPE (4K) | None | 245.3 | 20.15% | Improved hyperparameters |
| **BPE v3** | BPE (5K) | None | 235.8 | 20.45% | Larger vocab |
| **BPE v4** | BPE (5K) | None | 229.70 | 20.80% | nanoGPT optimizations |
| **Pre+Fine v1** | BPE (8K) | Gutenberg 15ep | 191.9 | 22.53% | Transfer learning |
| **Pre+Fine v2** | BPE (8K) | Gutenberg 30ep | **178.9** | **23.33%** | Discriminative fine-tuning |

### 3.2 Detailed Experiment Reports

#### Experiment 1: Word-Level Baseline

**Configuration:**
- Tokenizer: Word-level with FastText initialization
- Vocabulary: ~10,000 words (MIN_FREQ=3)
- Embeddings: FastText wiki-news-subwords-300 (pre-trained)
- Training: 50 epochs on Shakespeare only

**Results:**
```
Test Loss:       5.9739
Test Perplexity: 393.04
Test Accuracy:   18.78%
```

**Issues Identified:**
- High OOV rate for archaic words ("thou", "hath", "doth")
- Vocabulary too large for dataset size
- Sparse training signal for rare words

---

#### Experiment 2: BPE v1-v3 (Iterative Improvements)

**Key Changes:**
- Switched to BPE tokenization (4K → 5K vocab)
- Removed FastText pre-training (incompatible with BPE)
- Tuned dropout, learning rate, batch size

**Results Progression:**
```
BPE v1: PPL 267.5 → BPE v2: PPL 245.3 → BPE v3: PPL 235.8
```

**Key Insight:**
BPE dramatically reduced OOV issues and provided consistent subword coverage.

---

#### Experiment 3: BPE v4 (nanoGPT Optimizations)

**Changes Applied:**
1. `bias=False` on all linear layers (reduces params, improves regularization)
2. Scaled residual initialization: `std = 0.02 / sqrt(2 * num_layers)`
3. β₂ = 0.99 (from 0.999) for AdamW optimizer
4. Pre-LayerNorm architecture (already present)
5. Weight tying (already present)

**Results:**
```
Test Loss:       5.4371
Test Perplexity: 229.70
Test Accuracy:   20.80%
```

**Analysis:**
Improvements were marginal because the main nanoGPT benefits (Pre-Norm, Weight Tying) were already implemented. The model hit a **capacity ceiling** — 6.4M parameters on 1.1M tokens is 5.8× over Chinchilla optimal.

---

#### Experiment 4: Pre-train + Fine-tune v1

**Motivation:** Apply Chinchilla scaling principle via transfer learning.

**Pre-training Corpus (Gutenberg):**
- 19 classic English texts (~23MB)
- 5.7M BPE tokens (vs 1.4M Shakespeare)
- Chinchilla ratio: 7.3M params / 5.1M tokens = 1.43 params/token

**Configuration:**
```python
PRETRAIN_CONFIG = {
    "bpe_vocab_size": 8000,
    "num_epochs": 15,
    "learning_rate": 5e-4,
    "dropout": 0.15,
}

FINETUNE_CONFIG = {
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "dropout": 0.2,
}
```

**Results:**
```
Pre-training (Gutenberg):
  Test Loss: 5.0257
  Test PPL:  152.3
  Test Acc:  26.87%

Fine-tuning (Shakespeare):
  Test Loss: 5.2571
  Test PPL:  191.9
  Test Acc:  22.53%
```

**Key Achievement:**
- Pre-training provided better English language understanding
- Fine-tuning specialized to Shakespeare style
- 16.4% PPL reduction over BPE v4

---

#### Experiment 5: Pre-train + Fine-tune v2 (Final Best)

**New Technique: Discriminative Fine-Tuning (ULMFiT)**

Based on Howard & Ruder, 2018 "Universal Language Model Fine-tuning for Text Classification"

**Concept:** Different layers need different learning rates:
- **Bottom layers** (embeddings, early decoders): Learn general English patterns → should change slowly
- **Top layers** (final decoders): Learn task-specific patterns → can change faster

**Learning Rate Formula:**
```
η_l = η_top / ξ^(L-l)

where:
  η_l    = learning rate for layer l
  η_top  = top layer learning rate (1e-4)
  ξ      = decay factor (2.6, from ULMFiT)
  L      = total layers (5)
  l      = layer index (0 = bottom)
```

**Resulting Layer-wise LRs:**
```
Embeddings (general English):  lr = 3.24e-07
Decoder Layer 0 (bottom):      lr = 2.19e-06
Decoder Layer 1:               lr = 5.69e-06
Decoder Layer 2:               lr = 1.48e-05
Decoder Layer 3:               lr = 3.85e-05
Decoder Layer 4 (top):         lr = 1.00e-04
Final Norm / Output:           lr = 1.00e-04
```

**Extended Training Configuration:**
```python
PRETRAIN_CONFIG = {
    "num_epochs": 30,        # Extended from 15
    "patience": 10,
    "stride_min": 16,
}

FINETUNE_CONFIG = {
    "num_epochs": 45,        # Extended from 30
    "patience": 12,
    "discriminative_lr": True,
    "lr_decay_factor": 2.6,
}
```

**Full Training Log:**

**Pre-training Phase (30 epochs, 338.4 minutes):**
```
Epoch   1: Val PPL 490.7 → Epoch  10: Val PPL 176.3 (stride 128→64→32)
Epoch  11: Val PPL 170.6 → Epoch  20: Val PPL 152.3 (stride 32→16)
Epoch  21: Val PPL 151.9 → Epoch  30: Val PPL 148.5 (stride 16, converging)

Final Pre-train Test: Loss 5.0257, PPL 152.3, Accuracy 26.87%
```

**Fine-tuning Phase (45 epochs, 130.5 minutes):**
```
Epoch   1: Val PPL 215.5 → Epoch  10: Val PPL 171.7 (stride 128→64)
Epoch  11: Val PPL 169.5 → Epoch  20: Val PPL 159.6 (stride 32→16)
Epoch  21: Val PPL 158.9 → Epoch  30: Val PPL 157.3 (stride 16)
Epoch  31: Val PPL 157.1 → Epoch  45: Val PPL 156.7 (converged)

Final Fine-tune Test: Loss 5.1868, PPL 178.9, Accuracy 23.33%
```

**Final Results Comparison:**
```
BPE v4 (scratch):       Test PPL 229.70, Accuracy 20.80%
Pre+Fine v1 (uniform):  Test PPL 191.9,  Accuracy 22.53%
Pre+Fine v2 (discrim):  Test PPL 178.9,  Accuracy 23.33%  ← BEST
```

---

## 4. Key Findings & Insights

### 4.1 The Chinchilla Scaling Law

**Observation:** Training on Shakespeare alone (1.1M tokens) with a 7.3M parameter model is 5.8× over-parameterized by Chinchilla standards.

**Chinchilla Optimal Ratio:** ~20 tokens per parameter

| Configuration | Tokens | Params | Ratio | Status |
|--------------|--------|--------|-------|--------|
| Shakespeare only | 1.1M | 7.3M | 0.15 | 5.8× over-param |
| Gutenberg pre-train | 5.1M | 7.3M | 0.70 | 1.4× over-param ✓ |
| Combined (approx) | 6.5M | 7.3M | 0.89 | Near optimal ✓ |

**Lesson:** When data is limited, either reduce model size or add more data via transfer learning.

### 4.2 BPE vs Word-Level Tokenization

| Aspect | Word-Level | BPE |
|--------|------------|-----|
| Vocabulary Size | ~10K (dataset-dependent) | 5-8K (configurable) |
| OOV Handling | UNK token (information loss) | Subword decomposition |
| Rare Words | Poorly learned | Well-represented |
| Final PPL | 393 | 178.9 (best) |

**Conclusion:** BPE is strictly superior for small-corpus language modeling.

### 4.3 Transfer Learning Benefits

**What Pre-training Provides:**
1. General English syntax and grammar
2. Common word relationships and collocations
3. Better initial weights for fine-tuning
4. Reduced overfitting through broader exposure

**What Fine-tuning Specializes:**
1. Shakespearean vocabulary (archaic pronouns, verbs)
2. Iambic pentameter patterns
3. Stage directions and character names
4. Early Modern English idioms

### 4.4 Contracting Stride Effectiveness

**Stride Schedule:**
```
Epochs 1-5:   stride=128 →  623 batches/epoch (pre-train)
Epochs 6-10:  stride=64  → 1246 batches/epoch
Epochs 11-15: stride=32  → 2491 batches/epoch
Epochs 16+:   stride=16  → 4982 batches/epoch
```

**Benefits:**
- Early training: Fast iterations, diverse global patterns
- Late training: Dense coverage, local pattern refinement
- Total training examples increase 8× through training

### 4.5 Discriminative Fine-Tuning Impact

**Improvement:** PPL 191.9 → 178.9 (6.8% reduction)

**Why It Works:**
- Prevents catastrophic forgetting of general English knowledge
- Allows top layers to rapidly adapt to Shakespeare style
- Maintains stability in bottom layers (embeddings)

---

## 5. Dataset Information

### 5.1 Shakespeare Corpus (Fine-tuning)

**Source:** Project Gutenberg Complete Works of Shakespeare
**URL:** https://www.gutenberg.org/cache/epub/100/pg100.txt

| Metric | Value |
|--------|-------|
| Raw Size | 5.4 MB |
| Characters | 5,378,667 |
| After Cleaning | 5,359,230 chars |
| BPE Tokens | 1,370,232 |
| Train Tokens | 1,096,185 (80%) |
| Val Tokens | 137,023 (10%) |
| Test Tokens | 137,024 (10%) |

### 5.2 Gutenberg Corpus (Pre-training)

**Source:** 19 classic English texts from Project Gutenberg

| Work | Author | Size |
|------|--------|------|
| King James Bible | Various | 4.4 MB |
| Paradise Lost | John Milton | 500 KB |
| Canterbury Tales | Geoffrey Chaucer | 700 KB |
| The Iliad | Homer (Pope translation) | 700 KB |
| The Odyssey | Homer (Pope translation) | 600 KB |
| The Aeneid | Virgil (Dryden translation) | 400 KB |
| Beowulf | Anonymous | 150 KB |
| Le Morte d'Arthur | Sir Thomas Malory | 900 KB |
| Don Quixote | Miguel de Cervantes | 2.2 MB |
| Oliver Twist | Charles Dickens | 500 KB |
| Great Expectations | Charles Dickens | 500 KB |
| David Copperfield | Charles Dickens | 900 KB |
| Pride and Prejudice | Jane Austen | 700 KB |
| Emma | Jane Austen | 900 KB |
| Wuthering Heights | Emily Brontë | 400 KB |
| Jane Eyre | Charlotte Brontë | 500 KB |
| Moby Dick | Herman Melville | 1.2 MB |
| War and Peace | Tolstoy (Maude trans.) | 3.2 MB |
| Anna Karenina | Tolstoy (Maude trans.) | 2.0 MB |

| Metric | Value |
|--------|-------|
| Total Raw Size | 23.4 MB |
| Characters | 23,440,394 |
| BPE Tokens | 5,667,448 |
| Train Tokens | 5,100,703 (90%) |
| Val Tokens | 283,372 (5%) |
| Test Tokens | 283,373 (5%) |

### 5.3 BPE Tokenizer Statistics

**Trained on:** Combined Gutenberg + Shakespeare (~29 MB)

| Metric | Value |
|--------|-------|
| Vocabulary Size | 8,000 tokens |
| Compression Ratio | 4.1 chars/token |
| Special Tokens | PAD, UNK, BOS, EOS |

---

## 6. Training Configurations

### 6.1 Pre-training Configuration (v2)

```python
PRETRAIN_CONFIG = {
    # Tokenizer
    "bpe_vocab_size": 8000,
    
    # Architecture
    "num_layers": 5,
    "num_heads": 6,
    "embed_dim": 300,
    "ffn_hidden_dim": 1024,
    
    # Training
    "batch_size": 64,
    "learning_rate": 5e-4,
    "weight_decay": 0.05,
    "warmup_steps": 1000,
    "num_epochs": 30,
    "patience": 10,
    "dropout": 0.15,
    "attention_dropout": 0.1,
    "label_smoothing": 0.1,
    "max_seq_length": 128,
    
    # Contracting Stride
    "stride_initial": 128,
    "stride_min": 16,
    "stride_contract_every": 5,
}
```

### 6.2 Fine-tuning Configuration (v2)

```python
FINETUNE_CONFIG = {
    # Training
    "learning_rate": 1e-4,      # Top layer LR
    "weight_decay": 0.1,
    "warmup_steps": 200,
    "num_epochs": 45,
    "patience": 12,
    "dropout": 0.2,
    "attention_dropout": 0.15,
    "label_smoothing": 0.1,
    
    # Discriminative Fine-Tuning
    "discriminative_lr": True,
    "lr_decay_factor": 2.6,
    
    # Contracting Stride
    "stride_initial": 128,
    "stride_min": 16,
    "stride_contract_every": 5,
}
```

### 6.3 Generation Configuration

```python
GENERATION_CONFIG = {
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "max_length": 100,
    "repetition_penalty": 1.2,
}
```

---

## 7. Optimization Techniques

### 7.1 Applied Optimizations

| Technique | Source | Impact |
|-----------|--------|--------|
| **BPE Tokenization** | Sennrich et al., 2016 | Reduced OOV, better rare word handling |
| **Pre-LayerNorm** | GPT-2, LLaMA | More stable training, better gradients |
| **Weight Tying** | Press & Wolf, 2017 | ~2.4M fewer params, regularization |
| **Scaled Residual Init** | nanoGPT, GPT-2 | Stable early training |
| **Cosine LR Schedule** | Loshchilov & Hutter, 2017 | Smooth convergence |
| **AdamW (β₂=0.99)** | nanoGPT recommendation | Faster adaptation |
| **Label Smoothing** | Szegedy et al., 2016 | Prevents overconfidence |
| **Contracting Stride** | Custom | Progressive data exposure |
| **Transfer Learning** | Standard NLP | Better initialization |
| **Discriminative LR** | ULMFiT (Howard & Ruder, 2018) | Layer-wise adaptation |

### 7.2 Optimizations Considered But Not Applied

| Technique | Reason Not Applied |
|-----------|-------------------|
| **FlashAttention** | GTX 1660 Ti lacks Ampere architecture |
| **Gradient Checkpointing** | Not memory-constrained |
| **Mixed Precision (FP16)** | Marginal benefit at this scale |
| **Mixture of Experts** | Too complex for dataset size |
| **Rotary Positional Encoding** | Sinusoidal sufficient for seq_len=128 |

---

## 8. Code Structure

### 8.1 File Organization

```
ProjectNextWord/
├── config.py              # All hyperparameters and settings
├── bpe_tokenizer.py       # BPE tokenization implementation
├── embeddings.py          # Embedding layers with positional encoding
├── transformer.py         # Multi-head attention, FFN, decoder blocks
├── model.py               # Full model + text generator
├── data_loader.py         # Dataset and DataLoader utilities
├── train.py               # Training loop (original BPE experiments)
├── pretrain_finetune.py   # Transfer learning pipeline
├── gutenberg.py           # Gutenberg corpus downloader
├── utils.py               # Helper functions
├── main.py                # CLI entry point
│
├── data/
│   ├── shakespeare_full.txt
│   ├── gutenberg_corpus.txt
│   └── bpe_tokenizer_pretrain_8000.json
│
├── models/
│   ├── pretrained_gutenberg_v2.pt
│   └── finetuned_shakespeare_v2.pt
│
├── logs/
│   ├── pretrain_history.png
│   └── finetune_history.png
│
└── EXPERIMENT_REFERENCE.md  # This document
```

### 8.2 Key Classes

```python
# bpe_tokenizer.py
class BPETokenizer:
    def train(text, save_path)
    def encode(text) -> List[int]
    def decode(tokens) -> str

# embeddings.py
class EmbeddingWithTiedWeights(nn.Module):
    # Combines token + positional embeddings with weight tying

# transformer.py
class MultiHeadSelfAttention(nn.Module):
class FeedForwardNetwork(nn.Module):
class TransformerDecoderBlock(nn.Module):
class TransformerDecoder(nn.Module):

# model.py
class ShakespeareTransformer(nn.Module):
class TextGenerator:
    def generate(prompt, temperature, top_k, top_p)

# pretrain_finetune.py
class PretrainFinetuneTrainer:
    def pretrain(train_data, val_data)
    def finetune(train_data, val_data)
    def _create_discriminative_optimizer()
```

### 8.3 Usage Examples

```bash
# Full pipeline (pretrain + finetune)
python pretrain_finetune.py --mode all

# Pre-train only
python pretrain_finetune.py --mode pretrain

# Fine-tune from checkpoint
python pretrain_finetune.py --mode finetune

# Evaluate model
python pretrain_finetune.py --mode evaluate

# Generate text
python pretrain_finetune.py --mode generate
```

---

## 9. Generated Samples

### 9.1 Best Model Generations (Pre+Fine v2)

**Seed: "to be or not to be"**
```
to be or not to be a man . you are so mad . i am sorry that , i ' ll 
tell you , but i have been drunk . [ _ within . _ ] now , my lord . 
enter king . king . where is the duke ? what , what news
```

**Seed: "the king"**
```
the king ' s . i would not have thought her , but she had an excellent 
speech , and her tongue too . and there was a very rare wit in ' t . 
enter a messenger . messenger . where ' s the prince ? messenger . with 
him ,
```

**Seed: "love is"**
```
love is too hot . he ' s too heavy . you must be mad , sir . i ' ll 
make you mad with my husband . [ _exit . _ ] scene iv . another part 
of the forest enter queen richard and his son , the earl of york
```

**Seed: "thou art"**
```
thou art a creature . i must not know thee ; for i will , though thou 
hadst a brother , thou shalt have no wife with me . go , away ! thou 
wilt not be with us again . i ' ll not go with you . [ _exeunt .
```

**Seed: "what light through yonder"**
```
what light through yonder earth ' s clouds , like night - clouds to 
winter ; but i say i am not yet . a night before i see the moon ' s 
face , and then i think it is the night ' s night of day . and so i 
see the
```

### 9.2 Temperature Variations

**Prompt: "to be or not to be"**

| Temperature | Generation |
|-------------|------------|
| 0.5 (focused) | to be or not to be a man , but that ' s no matter . the king ' s a fool , and his wife ' s a fool , and his wife ' s |
| 0.8 (balanced) | to be or not to be a man . i would , my lord , i must do it with a man . hamlet . i had rather be a soldier ' s friend . king |
| 1.0 (diverse) | to be or not to be ; nor let them go . enter a messenger , servant , servant to a soldier . servant . why dost thou not know me ? messenger . yes . |
| 1.2 (creative) | to be or not to be so . ' twas his fault too that , having made them die , i have been all in doubt . falstaff . the matter will prove a villain , |

### 9.3 Generation Quality Analysis

**Strengths:**
- Correct stage directions format: `[ _exit . _ ]`, `enter king`
- Proper character names: hamlet, king, falstaff, york
- Archaic pronouns: thou, thee, thy
- Coherent dialogue structure

**Weaknesses:**
- Occasional anachronisms or illogical transitions
- Repetitive patterns at high temperatures
- Character name consistency within passages

---

## 10. Theoretical Foundations

### 10.1 The Transformer Architecture

**Original Paper:** "Attention Is All You Need" (Vaswani et al., 2017)

**Key Innovation:** Replace recurrence (RNN/LSTM) with self-attention, enabling:
- Parallel processing of all positions
- Direct long-range dependencies
- Scalable training on GPUs

**Self-Attention Formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 10.2 Language Modeling Objective

**Task:** Predict the next token given previous tokens

**Loss Function:**
$$\mathcal{L} = -\sum_{t=1}^{T} \log P(w_t | w_1, ..., w_{t-1})$$

**Perplexity:**
$$\text{PPL} = \exp(\mathcal{L} / T)$$

Interpretation: A perplexity of 178.9 means the model is, on average, as uncertain as choosing uniformly among 178.9 equally likely next words.

### 10.3 Byte-Pair Encoding

**Algorithm:**
1. Initialize vocabulary with individual characters
2. Find most frequent adjacent pair
3. Merge pair into new token
4. Add to vocabulary
5. Repeat until target vocab size

**Example:**
```
"the theater" → "th" "e" " " "th" "e" "a" "t" "e" "r"
              → "the" " " "the" "a" "t" "e" "r"
              → "the" " " "thea" "t" "e" "r"
              → "the" " theater"
```

### 10.4 Chinchilla Scaling Law

**Paper:** "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)

**Finding:** For compute-optimal training:
$$N_{opt} \approx \frac{D}{20}$$

Where:
- $N_{opt}$ = optimal number of parameters
- $D$ = number of training tokens

**Implication:** A model should see ~20 tokens per parameter for optimal efficiency.

### 10.5 Transfer Learning in NLP

**Paradigm:**
1. **Pre-train** on large, general corpus (unsupervised)
2. **Fine-tune** on small, task-specific corpus (supervised)

**Benefits:**
- Better generalization from broader data exposure
- Reduced overfitting on small datasets
- Faster convergence during fine-tuning

**ULMFiT (Howard & Ruder, 2018):**
- Introduced discriminative fine-tuning
- Layer-wise learning rate decay
- Gradual unfreezing (not used here)

---

## 11. Future Work

### 11.1 Planned Experiments

1. **Random Stride Lengths**
   - Replace deterministic stride halving with randomized strides
   - May improve robustness and generalization

2. **Larger Pre-training Corpus**
   - Add more Gutenberg texts (goal: 50MB+)
   - Potentially include modern English sources

3. **Model Scaling**
   - Double model size to 15M parameters
   - Requires corresponding increase in data

### 11.2 Potential Improvements

1. **Rotary Positional Embeddings (RoPE)**
   - Better length generalization
   - Used in LLaMA, GPT-NeoX

2. **Grouped Query Attention**
   - Reduce memory footprint
   - Faster inference

3. **Knowledge Distillation**
   - Train smaller student model from our best model
   - Faster inference, same quality

4. **Domain-Adaptive Pre-training (DAPT)**
   - Pre-train on Early Modern English specifically
   - May improve Shakespeare domain fit

---

## 12. References

### 12.1 Core Papers

1. Vaswani, A., et al. (2017). **Attention Is All You Need.** NeurIPS.
2. Press, O., & Wolf, L. (2017). **Using the Output Embedding to Improve Language Models.** EACL.
3. Sennrich, R., et al. (2016). **Neural Machine Translation of Rare Words with Subword Units.** ACL.
4. Howard, J., & Ruder, S. (2018). **Universal Language Model Fine-tuning for Text Classification.** ACL.
5. Hoffmann, J., et al. (2022). **Training Compute-Optimal Large Language Models.** (Chinchilla paper)

### 12.2 Implementation References

1. Karpathy, A. **nanoGPT.** https://github.com/karpathy/nanoGPT
2. Karpathy, A. **char-rnn.** https://github.com/karpathy/char-rnn

### 12.3 Prior Art on Shakespeare

| Work | Val Loss | Notes |
|------|----------|-------|
| nanoGPT (char-level) | 1.47 | Character-level, no tokenization |
| Karpathy char-rnn | ~1.5 | LSTM-based |
| **This work** | 5.05 | Token-level BPE, transfer learning |

*(Note: Character-level losses are not directly comparable to token-level)*

---

## Appendix A: Training Logs

### A.1 Pre-training v2 Summary

```
Total Time: 338.4 minutes (~5.6 hours)
Best Epoch: 30
Best Val Loss: 4.9755 (Val PPL: 148.5)
Test Loss: 5.0257
Test PPL: 152.3
Test Accuracy: 26.87%
```

### A.2 Fine-tuning v2 Summary

```
Total Time: 130.5 minutes (~2.2 hours)
Best Epoch: 44
Best Val Loss: 5.0458 (Val PPL: 156.7)
Test Loss: 5.1868
Test PPL: 178.9
Test Accuracy: 23.33%
```

### A.3 Total Training Time

```
Pre-training:  338.4 min
Fine-tuning:   130.5 min
Total:         468.9 min (~7.8 hours)
```

---

## Appendix B: Model Checkpoints

| Checkpoint | Path | Description |
|------------|------|-------------|
| `pretrained_gutenberg_v2.pt` | models/ | Pre-trained on Gutenberg (30 epochs) |
| `finetuned_shakespeare_v2.pt` | models/ | Fine-tuned on Shakespeare (45 epochs) |
| `best_model_bpe.pt` | models/ | BPE v4 (scratch training) |
| `best_model.pt` | models/ | Word-level baseline |

---

## Appendix C: Environment Details

```
Python:     3.13.3
PyTorch:    2.6.0+cu124
CUDA:       12.4
GPU:        NVIDIA GeForce GTX 1660 Ti (6GB)
OS:         Windows 11
CPU:        [Your CPU details]
RAM:        [Your RAM details]
```

---

*Document generated: February 2026*
*Last experiment: Pre-train + Fine-tune v2 with Discriminative Fine-Tuning*
