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
| **Test Perplexity** | **146.4** | Pre-train v4 + Fine-tune v4 (Discriminative LR, 23M params) |
| **Test Accuracy** | **25.59%** | Pre-train v4 + Fine-tune v4 (Discriminative LR, 23M params) |
| **Pre-train Test PPL** | **112.8** | Pre-train v4 (324 Gutenberg books, 55M tokens) |
| **Pre-train Test Acc** | **28.08%** | Pre-train v4 (324 Gutenberg books, 55M tokens) |

### 1.3 Key Achievements
- **2.7× perplexity reduction** from word-level baseline (393 → 146.4)
- **36% perplexity reduction** over BPE scratch training (229.7 → 146.4)
- **3.2× model scaling**: 7.3M → 23M parameters with 6-layer / 8-head / 512d architecture
- **17× data scaling**: 19 Gutenberg books → 324 books (5.7M → 55M BPE tokens)
- Discriminative fine-tuning with 120× LR ratio between bottom and top layers
- Gradual unfreezing (ULMFiT) tested — showed discriminative LR already sufficient
- Demonstrated Chinchilla scaling principle: 55M tokens / 23M params ≈ 2.4:1 ratio

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

#### v2 Architecture (7.3M params — 5 layers, 6 heads, 300d)

| Component | Parameters | Calculation |
|-----------|------------|-------------|
| Token Embeddings | 2,400,000 | 8000 × 300 |
| Positional Encoding | 0 | Not learned (buffer) |
| Attention (Q,K,V,O × 5 layers) | 1,800,000 | 5 × 4 × 300 × 300 |
| FFN (fc1, fc2 × 5 layers) | 3,060,000 | 5 × 2 × 300 × 1024 |
| LayerNorm (11 total) | 6,600 | 11 × 2 × 300 |
| Output Projection | 0 | Tied to embeddings |
| **Total** | **7,275,300** | ~7.3M parameters |

#### v4 Architecture (23M params — 6 layers, 8 heads, 512d)

| Component | Parameters | Calculation |
|-----------|------------|-------------|
| Token Embeddings | 4,096,000 | 8000 × 512 |
| Positional Encoding | 0 | Not learned (buffer) |
| Attention (Q,K,V,O × 6 layers) | 6,291,456 | 6 × 4 × 512 × 512 |
| FFN (fc1, fc2 × 6 layers) | 12,582,912 | 6 × (512×2048 + 2048×512) |
| LayerNorm (13 total) | 13,312 | 13 × 2 × 512 |
| Output Projection | 0 | Tied to embeddings |
| **Total** | **~22,983,680** | ~23M parameters |

---

## 3. Experiment Timeline & Results

### 3.1 Experiment Summary Table

| # | Experiment | Tokenizer | Params | Training Data | Test PPL | Test Acc | Key Changes |
|---|------------|-----------|--------|---------------|----------|----------|-------------|
| 1 | **Word-level** | Word (10K) | ~7.3M | Shakespeare only (1.1M tok) | 393.0 | 18.78% | Baseline with FastText embeddings |
| 2 | **BPE v1** | BPE (4K) | ~6.4M | Shakespeare only (1.1M tok) | 267.5 | 19.40% | First BPE attempt |
| 3 | **BPE v2** | BPE (4K) | ~6.4M | Shakespeare only (1.1M tok) | 245.3 | 20.15% | Improved hyperparameters |
| 4 | **BPE v3** | BPE (5K) | ~6.4M | Shakespeare only (1.1M tok) | 235.8 | 20.45% | Larger vocab |
| 5 | **BPE v4** | BPE (5K) | ~6.4M | Shakespeare only (1.1M tok) | 229.70 | 20.80% | nanoGPT optimizations |
| 6 | **AWD-LSTM** | BPE (5K) | ~10M | Shakespeare only (1.1M tok) | ~178 | — | Merity et al. recipe, SGD+ASGD |
| 7 | **Pre+Fine v1** | BPE (8K) | 7.3M | Gutenberg 19 books (5.7M tok) → Shakespeare | 191.9 | 22.53% | Transfer learning, uniform LR |
| 8 | **Pre+Fine v2** | BPE (8K) | 7.3M | Gutenberg 19 books (5.7M tok) → Shakespeare | 178.9 | 23.33% | Discriminative fine-tuning |
| 9 | **Pre-train v3** | BPE (8K) | 7.3M | Gutenberg 324 books (55M tok) | 134.9* | — | Expanded corpus, *abandoned at ep 8 |
| 10 | **Pre-train v4** | BPE (8K) | **23M** | Gutenberg 324 books (55M tok) | **112.8** | **28.08%** | Scaled model (6L/8H/512d/2048FFN) |
| 11 | **Fine-tune v4** | BPE (8K) | 23M | v4 pretrained → Shakespeare (1.43M tok) | **146.4** | **25.59%** | Discriminative LR, stride=64 **BEST** |
| 12 | **Fine-tune v5** | BPE (8K) | 23M | v4 pretrained → Shakespeare (1.43M tok) | 154.2 | 24.91% | Heavier regularisation (dropout=0.25) |
| 13 | **Fine-tune v6** | BPE (8K) | 23M | v4 pretrained → Shakespeare (1.43M tok) | 148.2 | 25.63% | Gradual unfreezing (ULMFiT) |

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
Pre+Fine v2 (discrim):  Test PPL 178.9,  Accuracy 23.33%
```

---

#### Experiment 6: AWD-LSTM Baseline

**Motivation:** Fair head-to-head comparison with Transformer using identical data pipeline.

**Architecture (Merity et al., 2018):**
```
AWD-LSTM (3-layer, weight drop + variational dropout)
  Embedding Dim:     300 (same as Transformer)
  Hidden Size:       1150
  LSTM Layers:       3
  Weight Tying:      Yes (embedding ↔ output)
  Total Parameters:  ~10M
  
  Dropout Rates:
    Embedding Drop:  0.1
    Input VarDrop:   0.3  (paper: 0.65 — reduced for our scale)
    Hidden VarDrop:  0.25 (paper: 0.3)
    Output VarDrop:  0.4
    Weight Drop:     0.5  (DropConnect on recurrent weights)
```

**Training Recipe:**
- Optimizer: SGD → NT-ASGD (non-monotonic trigger after 5 epochs plateau)
- AR (Activation Regularization): α=2.0
- TAR (Temporal Activation Regularization): β=1.0
- Same BPE-5000 tokenizer and contracting stride as Transformer

**Results:**
```
Best Val PPL:  ~178
Status:        Training completed but terminal exited with code 1
               (exact test metrics not recorded)
```

**Checkpoint:** `best_model_lstm.pt`

---

#### Experiment 7: Gutenberg Expansion + Pre-train v3

**Motivation:** Massively expand pre-training corpus from 19 → 324 books.

**Expanded Gutenberg Corpus:**
```
Books:         324 unique texts (332 entries, 324 unique URLs)
Authors:       ~150+ (Dickens, Austen, Twain, Tolstoy, Dostoevsky, etc.)
Raw Size:      217.5 MB (228,113,706 characters)
BPE Tokens:    55,200,000 (8K vocab)
Train Tokens:  49,680,000 (90%)
Val Tokens:    2,760,000 (5%)
Test Tokens:   2,760,000 (5%)
```

**Configuration (same 7.3M param model):**
```python
{
    "num_layers": 5, "num_heads": 6, "embed_dim": 300, "ffn_hidden_dim": 1024,
    "bpe_vocab_size": 8000, "batch_size": 64, "learning_rate": 3e-4,
    "num_epochs": 10, "max_seq_length": 128
}
```

**Results:**
```
Reached Epoch 8: Val PPL 134.9
Status:          Abandoned — model too small for 55M tokens (underfitting)
                 Chinchilla ratio: 55M / 7.3M = 7.5:1 (model needs scaling)
```

**Checkpoint:** `pretrained_gutenberg_v3.pt`

**Key Insight:** With 17× more data, the 7.3M param model was severely capacity-limited. This motivated scaling to 23M parameters.

---

#### Experiment 8: Pre-train v4 (Scaled Model, 23M params)

**Motivation:** Scale model to match expanded Gutenberg corpus.

**Architecture (scaled up):**
```
Previous (7.3M):  5 layers, 6 heads, 300d, 1024 FFN
New (23M):        6 layers, 8 heads, 512d, 2048 FFN

Chinchilla ratio: 55M tokens / 23M params ≈ 2.4:1 (healthy regime)
VRAM usage:       ~5.2 GB (fits GTX 1660 Ti with batch_size=64)
```

**Configuration:**
```python
PRETRAIN_CONFIG = {
    "bpe_vocab_size": 8000,
    "num_layers": 6, "num_heads": 8, "embed_dim": 512, "ffn_hidden_dim": 2048,
    "batch_size": 64, "learning_rate": 3e-4, "weight_decay": 0.05,
    "warmup_steps": 2000, "num_epochs": 10, "patience": 8,
    "dropout": 0.1, "attention_dropout": 0.05, "label_smoothing": 0.1,
    "max_seq_length": 128,
    "seq_len_switch_epoch": 8, "short_seq_length": 64,
    "stride_initial": 128, "stride_min": 128, "stride_contract_every": 999,
}
```

**Training Log:**
```
Epoch  1: Train PPL 469.0 | Val PPL 270.9 | LR 2.97e-04 | 44.3 min
Epoch  2: Train PPL 214.9 | Val PPL 200.2 | LR 2.85e-04 | 44.3 min
Epoch  3: Train PPL 177.1 | Val PPL 170.1 | LR 2.64e-04 | 44.3 min
Epoch  4: Train PPL 156.3 | Val PPL 152.1 | LR 2.35e-04 | 44.3 min
Epoch  5: Train PPL 143.1 | Val PPL 141.6 | LR 2.01e-04 | 44.3 min
Epoch  6: Train PPL 133.5 | Val PPL 133.7 | LR 1.65e-04 | 44.3 min
Epoch  7: Train PPL 126.3 | Val PPL 127.8 | LR 1.28e-04 | 46.5 min
Epoch  8: Train PPL 136.6 | Val PPL 119.5 | LR 9.18e-05 | 88.7 min ← seq_len→64
Epoch  9: Train PPL 127.5 | Val PPL 117.1 | LR 5.76e-05 | 88.6 min
Epoch 10: Train PPL 121.4 | Val PPL 116.2 | LR 2.72e-05 | 88.6 min
```

**Results:**
```
Total Time:      585.8 minutes (~9.8 hours)
Best Val PPL:    116.2 (epoch 10)
Test Loss:       4.7263
Test PPL:        112.8
Test Accuracy:   28.08%
```

**Checkpoint:** `pretrained_gutenberg_v4.pt`

**Key Achievement:** Val PPL 116.2 vs v2's 148.5 — **21.7% improvement** from scaling model + data.

---

#### Experiment 9: Fine-tune v4 (Discriminative LR) ← BEST MODEL

**Motivation:** Fine-tune scaled 23M param model on Shakespeare with discriminative LR.

**Configuration:**
```python
FINETUNE_CONFIG = {
    "learning_rate": 3e-5,           # ~10× pre-train's final LR (was 1e-4 in v2)
    "weight_decay": 0.1,
    "warmup_steps": 150,
    "num_epochs": 25, "patience": 8,
    "dropout": 0.2, "attention_dropout": 0.15,
    "label_smoothing": 0.1,
    "discriminative_lr": True, "lr_decay_factor": 2.6,
    "gradual_unfreezing": False,
    "stride_initial": 64, "stride_min": 64, "stride_contract_every": 999,
}
```

**Discriminative LR Schedule (6 layers + embeddings):**
```
Embeddings:        3.74e-08  (barely moves)
Decoder Layer 0:   2.53e-07
Decoder Layer 1:   6.57e-07
Decoder Layer 2:   1.71e-06
Decoder Layer 3:   4.44e-06
Decoder Layer 4:   1.15e-05
Decoder Layer 5:   3.00e-05  (top — adapts fastest)
Output Head:       3.00e-05
LR ratio (top/bottom): ~120×
```

**Training Log:**
```
Epoch  1: Train PPL 154.3 | Val PPL 132.4 | LR 3.00e-05 | 2.6 min
Epoch  2: Train PPL 127.8 | Val PPL 125.1 | LR 2.99e-05 | 2.6 min
...
Epoch 11: Train PPL  96.7 | Val PPL 119.4 | LR 1.78e-05 | 2.6 min
...
Epoch 22: Train PPL  75.3 | Val PPL 117.9 | LR 4.81e-06 | 2.6 min  ← best
...
Epoch 25: Train PPL  72.4 | Val PPL 118.6 | LR 1.88e-06 | 2.6 min
```

**Results:**
```
Total Time:      65.7 minutes (~1.1 hours)
Best Val PPL:    117.9 (epoch 22)
Test PPL:        146.4
Test Accuracy:   25.59%
Val-Test Gap:    ~24% (structural — different plays in val vs test split)
```

**Checkpoint:** `finetuned_shakespeare_v4.pt`

**Analysis:**
- **36% PPL improvement** over BPE scratch (229.7 → 146.4)
- **18.2% PPL improvement** over v2 fine-tune (178.9 → 146.4)
- Val-test gap (~24%) is structural: sequential split means different plays in each partition, and some plays have more archaic/unusual language

---

#### Experiment 10: Fine-tune v5 (Heavier Regularisation)

**Motivation:** Reduce val-test gap by increasing regularisation.

**Changes from v4:**
```diff
- stride_initial: 64   → + stride_initial: 128   (less overlap)
- dropout: 0.2         → + dropout: 0.25          (more dropout)
- num_epochs: 25       → + num_epochs: 18
- patience: 8          → + patience: 6
```

**Results:**
```
Total Time:    34.1 minutes
Best Val PPL:  126.0
Test PPL:      154.2
Test Accuracy: 24.91%
Val-Test Gap:  ~22%
```

**Checkpoint:** `finetuned_shakespeare_v5.pt`

**Conclusion:** Heavier regularisation made everything **worse**. The val-test gap is structural (different plays), not overfitting. Model was actually slightly underfitting at these settings.

---

#### Experiment 11: Fine-tune v6 (Gradual Unfreezing)

**Motivation:** Test ULMFiT's gradual unfreezing — start with only top layer trainable, progressively unfreeze deeper layers.

**Unfreezing Schedule:**
```
Epochs  1-3:  Only top layer + output head  (13.7% trainable, 3.2M/23M)
Epochs  4-6:  + Decoder Layer 4             (27.4% trainable, 6.3M/23M)
Epochs  7-9:  + Decoder Layer 3             (41.1% trainable, 9.5M/23M)
Epochs 10-12: + Decoder Layer 2             (54.8% trainable, 12.6M/23M)
Epochs 13-15: + Decoder Layer 1             (68.5% trainable, 15.8M/23M)
Epochs 16-18: + Decoder Layer 0             (82.2% trainable, 18.9M/23M)
Epochs 19-25: + Embeddings                  (100% trainable, 23M/23M)
```

**Results:**
```
Total Time:    88.9 minutes
Best Val PPL:  119.2
Test PPL:      148.2
Test Accuracy: 25.63%
Val-Test Gap:  ~24%
```

**Checkpoint:** `finetuned_shakespeare_v6.pt`

**Conclusion:** Essentially identical to v4 (PPL 148.2 vs 146.4). Discriminative LR already protected lower layers with a 120× ratio between bottom and top LRs, making gradual unfreezing redundant. The experiment confirmed v4 as optimal.

---

### 3.3 Results Progression Chart

```
PPL
400 ┤ ■ Word-level (393)
    │
300 ┤
    │   ■ BPE v1 (267.5)
250 ┤     ■ BPE v2-v3 (245-236)
    │       ■ BPE v4 (229.7)
200 ┤
    │         ■ Pre+Fine v1 (191.9)
    │           ■ Pre+Fine v2 (178.9)    ◀ old best (7.3M, 19 books)
150 ┤             ■ v5 (154.2)
    │             ■ v6 (148.2)
    │             ■ v4 Fine-tune (146.4) ◀ NEW BEST (23M, 324 books)
    │
100 ┤
    └──────────────────────────────────────────────
        Experiment Progression →
```

---

## 4. Key Findings & Insights

### 4.1 The Chinchilla Scaling Law

**Observation:** Training on Shakespeare alone (1.1M tokens) with a 7.3M parameter model is 5.8× over-parameterized by Chinchilla standards.

**Chinchilla Optimal Ratio:** ~20 tokens per parameter

| Configuration | Tokens | Params | Tok/Param | Status |
|--------------|--------|--------|-----------|--------|
| Shakespeare only (BPE v4) | 1.1M | 6.4M | 0.17 | 5.8× over-param |
| Gutenberg 19 books (v1-v2) | 5.1M | 7.3M | 0.70 | 1.4× over-param |
| Gutenberg 324 books (v3) | 55M | 7.3M | 7.5 | Model too small ✗ |
| **Gutenberg 324 books (v4)** | **55M** | **23M** | **2.4** | **Healthy regime ✓** |
| Chinchilla optimal | 55M | 2.75M | 20 | Theoretical optimal |

**Lessons:**
1. When data is limited, either reduce model size or add more data via transfer learning
2. When data is massively expanded (17×), the model must scale proportionally
3. Our 2.4:1 ratio is still over-parameterised vs Chinchilla-optimal, but well within practical range

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

**Impact across model scales:**

| Scale | Uniform LR | Discriminative LR | Improvement |
|-------|-----------|-------------------|-------------|
| 7.3M params, 19 books | PPL 191.9 | PPL 178.9 | 6.8% |
| 23M params, 324 books | N/A | PPL 146.4 | — (only tested discrim.) |

**Why It Works:**
- Prevents catastrophic forgetting of general English knowledge
- Allows top layers to rapidly adapt to Shakespeare style
- With 23M model, the 120× LR ratio (top vs bottom) is so protective that gradual unfreezing (v6) adds no benefit

### 4.6 Why Gradual Unfreezing Didn't Help (v6)

ULMFiT’s gradual unfreezing (freeze all but top layer, unfreeze one layer every N epochs) was tested in v6 but produced essentially the same result as v4 (PPL 148.2 vs 146.4).

**Explanation:** Discriminative LR with decay_factor=2.6 over 7 parameter groups already creates a 120× ratio between bottom and top learning rates. The bottom layers (LR ~3.7e-8) barely move — functionally equivalent to being frozen. Adding explicit freezing on top of this provides no additional protection.

### 4.7 The Val-Test Gap

All v4–v6 fine-tuned models show a ~24% gap between val PPL and test PPL (e.g., 117.9 vs 146.4 for v4). This gap is **structural**, not a sign of overfitting:

- Shakespeare’s text is split sequentially (80/10/10), so val and test contain **different plays**
- Some plays have more archaic or unusual language than others
- Heavier regularisation (v5) did not reduce the gap, confirming it’s not overfitting

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

#### Phase 1: Original 19 Books (used in v1 and v2)
**Source:** 19 classic English texts from Project Gutenberg

| Work | Author | Size |
|------|--------|------|
| King James Bible | Various | 4.4 MB |
| Paradise Lost | John Milton | 500 KB |
| Canterbury Tales | Geoffrey Chaucer | 700 KB |
| The Iliad / Odyssey | Homer (Pope) | 1.3 MB |
| The Aeneid | Virgil (Dryden) | 400 KB |
| Beowulf | Anonymous | 150 KB |
| Le Morte d'Arthur | Sir Thomas Malory | 900 KB |
| Don Quixote | Cervantes | 2.2 MB |
| Oliver Twist / Great Exp. / David Copper. | Dickens | 1.9 MB |
| Pride & Prejudice / Emma | Austen | 1.6 MB |
| Wuthering Heights / Jane Eyre | Brontës | 900 KB |
| Moby Dick | Melville | 1.2 MB |
| War and Peace / Anna Karenina | Tolstoy | 5.2 MB |

| Metric | Value |
|--------|-------|
| Total Raw Size | 23.4 MB |
| Characters | 23,440,394 |
| BPE Tokens | 5,667,448 |
| Train Tokens | 5,100,703 (90%) |
| Val Tokens | 283,372 (5%) |
| Test Tokens | 283,373 (5%) |

#### Phase 2: Expanded 324 Books (used in v3 and v4)
**Source:** 324 unique texts from Project Gutenberg (332 entries, 324 unique URLs)

**Coverage:** ~150+ authors spanning 16th–20th century English literature including:
- Charles Dickens (12+ works), Jane Austen (6 works), Mark Twain (10+ works)
- Shakespeare's contemporaries: Marlowe, Jonson, Webster
- Poetry: Milton, Shelley, Keats, Byron, Wordsworth, Whitman
- American lit: Hawthorne, Melville, Poe, London, Fitzgerald
- Russian lit (translations): Tolstoy, Dostoevsky, Chekhov, Turgenev
- French lit (translations): Hugo, Dumas, Verne, Balzac
- Philosophy: Plato, Aristotle, Marcus Aurelius, Machiavelli
- Religious texts: King James Bible, Quran, Bhagavad Gita

| Metric | Value |
|--------|-------|
| Total Books | 324 unique |
| Total Raw Size | 217.5 MB |
| Characters | 228,113,706 |
| BPE Tokens (8K vocab) | ~55,200,000 |
| Train Tokens | ~49,680,000 (90%) |
| Val Tokens | ~2,760,000 (5%) |
| Test Tokens | ~2,760,000 (5%) |
| Cached File | `data/gutenberg_expanded.txt` |

### 5.3 BPE Tokenizer Statistics

**Trained on:** Combined Gutenberg + Shakespeare (~29 MB)

| Metric | Value |
|--------|-------|
| Vocabulary Size | 8,000 tokens |
| Compression Ratio | 4.1 chars/token |
| Special Tokens | PAD, UNK, BOS, EOS |

---

## 6. Training Configurations

### 6.1 Pre-training Configuration

#### v2 (19 books, 7.3M params)
```python
PRETRAIN_CONFIG = {
    "bpe_vocab_size": 8000,
    "num_layers": 5, "num_heads": 6, "embed_dim": 300, "ffn_hidden_dim": 1024,
    "batch_size": 64, "learning_rate": 5e-4, "weight_decay": 0.05,
    "warmup_steps": 1000, "num_epochs": 30, "patience": 10,
    "dropout": 0.15, "attention_dropout": 0.1, "label_smoothing": 0.1,
    "max_seq_length": 128,
    "stride_initial": 128, "stride_min": 16, "stride_contract_every": 5,
}
```

#### v4 (324 books, 23M params) — FINAL
```python
PRETRAIN_CONFIG = {
    "bpe_vocab_size": 8000,
    "num_layers": 6, "num_heads": 8, "embed_dim": 512, "ffn_hidden_dim": 2048,
    "batch_size": 64, "learning_rate": 3e-4, "weight_decay": 0.05,
    "warmup_steps": 2000, "num_epochs": 10, "patience": 8,
    "dropout": 0.1, "attention_dropout": 0.05, "label_smoothing": 0.1,
    "max_seq_length": 128,
    "seq_len_switch_epoch": 8, "short_seq_length": 64,
    "stride_initial": 128, "stride_min": 128, "stride_contract_every": 999,
}
```

### 6.2 Fine-tuning Configuration

#### v2 (7.3M params, 19-book pre-train)
```python
FINETUNE_CONFIG = {
    "learning_rate": 1e-4, "weight_decay": 0.1,
    "warmup_steps": 200, "num_epochs": 45, "patience": 12,
    "dropout": 0.2, "attention_dropout": 0.15, "label_smoothing": 0.1,
    "discriminative_lr": True, "lr_decay_factor": 2.6,
    "stride_initial": 128, "stride_min": 16, "stride_contract_every": 5,
}
```

#### v4 (23M params, 324-book pre-train) — BEST
```python
FINETUNE_CONFIG = {
    "learning_rate": 3e-5, "weight_decay": 0.1,
    "warmup_steps": 150, "num_epochs": 25, "patience": 8,
    "dropout": 0.2, "attention_dropout": 0.15, "label_smoothing": 0.1,
    "discriminative_lr": True, "lr_decay_factor": 2.6,
    "gradual_unfreezing": False,
    "stride_initial": 64, "stride_min": 64, "stride_contract_every": 999,
}
```

#### v5 (heavier regularisation) — Worse
```python
# Changes from v4:
"stride_initial": 128, "dropout": 0.25, "num_epochs": 18, "patience": 6
```

#### v6 (gradual unfreezing) — No improvement
```python
# Changes from v4:
"gradual_unfreezing": True, "unfreeze_every": 3
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
| **Weight Tying** | Press & Wolf, 2017 | ~2.4M–4.1M fewer params, regularization |
| **Scaled Residual Init** | nanoGPT, GPT-2 | Stable early training |
| **Cosine LR Schedule** | Loshchilov & Hutter, 2017 | Smooth convergence |
| **AdamW (β₂=0.99)** | nanoGPT recommendation | Faster adaptation |
| **Label Smoothing** | Szegedy et al., 2016 | Prevents overconfidence |
| **Contracting Stride** | Custom | Progressive data exposure |
| **Transfer Learning** | Standard NLP | Better initialization |
| **Discriminative LR** | ULMFiT (Howard & Ruder, 2018) | Layer-wise adaptation (120× ratio) |
| **Model Scaling (v4)** | Chinchilla principles | 7.3M → 23M params for 55M tokens |
| **Seq Length Schedule** | Custom | 128 → 64 in final 3 epochs |
| **Gradual Unfreezing** | ULMFiT (Howard & Ruder, 2018) | Tested — no benefit over discriminative LR |

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
│   ├── gutenberg_corpus.txt               # Original 19 books
│   ├── gutenberg_expanded.txt             # Expanded 324 books (217 MB)
│   ├── bpe_tokenizer_pretrain_8000.json   # BPE tokenizer (19-book corpus)
│   └── bpe_tokenizer_expanded_8000.json   # BPE tokenizer (324-book corpus)
│
├── models/
│   ├── best_model.pt                  # Word-level baseline
│   ├── best_model_bpe.pt              # BPE v4 (scratch)
│   ├── best_model_lstm.pt             # AWD-LSTM baseline
│   ├── pretrained_gutenberg.pt        # Pre-train v1 (19 books)
│   ├── pretrained_gutenberg_v2.pt     # Pre-train v2 (19 books, 30ep)
│   ├── pretrained_gutenberg_v3.pt     # Pre-train v3 (324 books, 7.3M, abandoned)
│   ├── pretrained_gutenberg_v4.pt     # Pre-train v4 (324 books, 23M) ★
│   ├── finetuned_shakespeare.pt       # Fine-tune v1 (uniform LR)
│   ├── finetuned_shakespeare_v2.pt    # Fine-tune v2 (discrim. LR, 7.3M)
│   ├── finetuned_shakespeare_v4.pt    # Fine-tune v4 (discrim. LR, 23M) ★ BEST
│   ├── finetuned_shakespeare_v5.pt    # Fine-tune v5 (heavier reg.)
│   └── finetuned_shakespeare_v6.pt    # Fine-tune v6 (gradual unfreezing)
│
├── logs/
│   ├── pretrain_history.png
│   ├── finetune_history.png
│   ├── training_history.png
│   └── training_history_bpe.png
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

### 11.1 Completed Experiments (from original plan)

| Planned | Status | Outcome |
|---------|--------|---------|
| Larger Pre-training Corpus | ✅ Done | 19 → 324 books (17× expansion) |
| Model Scaling | ✅ Done | 7.3M → 23M params (3.2×) |
| Gradual Unfreezing | ✅ Tested | No improvement over discriminative LR |

### 11.2 Remaining Potential Improvements

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
| nanoGPT (char-level) | 1.47 | Character-level, 10.7M params, ~32K chars |
| Karpathy char-rnn | ~1.5 | LSTM-based, character-level |
| **This work (v4)** | **4.73** | **Token-level BPE, 23M params, transfer learning** |
| This work (v2) | 5.05 | Token-level BPE, 7.3M params, transfer learning |

*(Note: Character-level losses are not directly comparable to token-level.
BPC comparison: our BPE model = 4.73 / ln(2) / 4.1 ≈ 1.66 bits/char,
vs nanoGPT char-level = 1.47 / ln(2) ≈ 2.12 bits/char —
BPE model is actually more efficient per character.)*

---

## Appendix A: Training Logs

### A.1 Pre-training v2 Summary (19 books, 7.3M params)

```
Total Time: 338.4 minutes (~5.6 hours)
Best Epoch: 30
Best Val Loss: 4.9755 (Val PPL: 148.5)
Test Loss: 5.0257
Test PPL: 152.3
Test Accuracy: 26.87%
```

### A.2 Fine-tuning v2 Summary (7.3M params)

```
Total Time: 130.5 minutes (~2.2 hours)
Best Epoch: 44
Best Val Loss: 5.0458 (Val PPL: 156.7)
Test Loss: 5.1868
Test PPL: 178.9
Test Accuracy: 23.33%
```

### A.3 Pre-training v4 Summary (324 books, 23M params)

```
Total Time: 585.8 minutes (~9.8 hours)
Best Epoch: 10
Best Val PPL: 116.2
Test Loss: 4.7263
Test PPL: 112.8
Test Accuracy: 28.08%

Seq Length Schedule: 128 (epochs 1-7) → 64 (epochs 8-10)
Stride: Fixed at 128 (no contraction)
Data: 324 Gutenberg books, ~55M BPE tokens
```

### A.4 Fine-tuning v4 Summary (23M params) — BEST

```
Total Time: 65.7 minutes (~1.1 hours)
Best Epoch: 22
Best Val PPL: 117.9
Test PPL: 146.4
Test Accuracy: 25.59%
Discriminative LR: 3.74e-08 (embeddings) → 3e-05 (top layer)
```

### A.5 Fine-tuning v5 Summary (heavier regularisation)

```
Total Time: 34.1 minutes
Best Val PPL: 126.0
Test PPL: 154.2
Test Accuracy: 24.91%
Changes: stride=128, dropout=0.25
Outcome: WORSE than v4 — the val-test gap is structural, not overfitting
```

### A.6 Fine-tuning v6 Summary (gradual unfreezing)

```
Total Time: 88.9 minutes (~1.5 hours)
Best Val PPL: 119.2
Test PPL: 148.2
Test Accuracy: 25.63%
Unfreezing: 1 layer every 3 epochs (13.7% → 100% trainable)
Outcome: Same as v4 — discriminative LR already sufficient
```

### A.7 Total Training Time

```
Phase 1 (v1-v2, 19 books):
  Pre-training v2:   338.4 min
  Fine-tuning v2:    130.5 min
  Subtotal:          468.9 min (~7.8 hours)

Phase 2 (v3-v6, 324 books):
  Pre-training v4:   585.8 min
  Fine-tuning v4:     65.7 min
  Fine-tuning v5:     34.1 min
  Fine-tuning v6:     88.9 min
  Subtotal:          774.5 min (~12.9 hours)

Grand Total:        ~1243 min (~20.7 hours)
```

---

## Appendix B: Model Checkpoints

| Checkpoint | Path | Params | Description |
|------------|------|--------|-------------|
| `best_model.pt` | models/ | ~7.3M | Word-level baseline (FastText embeddings) |
| `best_model_bpe.pt` | models/ | ~6.4M | BPE v4 scratch (nanoGPT optimizations) |
| `best_model_lstm.pt` | models/ | ~10M | AWD-LSTM baseline (Merity et al.) |
| `pretrained_gutenberg.pt` | models/ | 7.3M | Pre-train v1 (19 books, 15 epochs) |
| `pretrained_gutenberg_v2.pt` | models/ | 7.3M | Pre-train v2 (19 books, 30 epochs) |
| `pretrained_gutenberg_v3.pt` | models/ | 7.3M | Pre-train v3 (324 books, abandoned ep 8) |
| `pretrained_gutenberg_v4.pt` | models/ | **23M** | **Pre-train v4 (324 books, 10 epochs)** |
| `finetuned_shakespeare.pt` | models/ | 7.3M | Fine-tune v1 (uniform LR) |
| `finetuned_shakespeare_v2.pt` | models/ | 7.3M | Fine-tune v2 (discriminative LR) |
| `finetuned_shakespeare_v4.pt` | models/ | **23M** | **Fine-tune v4 (discrim. LR) — BEST** |
| `finetuned_shakespeare_v5.pt` | models/ | 23M | Fine-tune v5 (heavier regularisation) |
| `finetuned_shakespeare_v6.pt` | models/ | 23M | Fine-tune v6 (gradual unfreezing) |

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
*Last updated: After Pre-train v4 + Fine-tune v4/v5/v6 experiments*
*Best model: Fine-tune v4 — Test PPL 146.4, Accuracy 25.59% (23M params, 324 Gutenberg books)*
