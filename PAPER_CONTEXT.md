# Paper Context Brief — CSO7013 Machine Learning Final Assessment
## Shakespeare Next-Word Prediction with Transformer Architecture

**Last Updated:** February 21, 2026

---

## 1. Project Identity

- **Course:** CSO7013 Machine Learning (Masters in Data Science)
- **Task:** Next-word prediction on Shakespeare's Complete Works
- **Architecture:** Decoder-only Transformer (GPT-style)
- **Framework:** PyTorch 2.6.0+cu124
- **Hardware:** NVIDIA GeForce GTX 1660 Ti (6GB VRAM), Windows 11
- **Python:** 3.13.3
- **Repository:** Ahmed275-Wael/NextWordPrediction

---

## 2. Best Results

| Metric | Value | Model |
|--------|-------|-------|
| **Test Perplexity** | **146.4** | Fine-tune v4 (23M params, discriminative LR) |
| **Test Accuracy** | **25.59%** | Fine-tune v4 |
| **Pre-train Test PPL** | **112.8** | Pre-train v4 (324 Gutenberg books, 55M tokens) |
| **Pre-train Test Acc** | **28.08%** | Pre-train v4 |
| **Bits/Char** | **~1.66** | Competitive with nanoGPT char-level (~2.12 BPC) |

---

## 3. Key Achievements

- **2.7× perplexity reduction** from word-level baseline (393 → 146.4)
- **36% perplexity reduction** over BPE scratch training (229.7 → 146.4)
- **3.2× model scaling**: 7.3M → 23M parameters (6L/8H/512d/2048FFN)
- **17× data scaling**: 19 → 324 Gutenberg books (5.7M → 55M BPE tokens)
- **Discriminative fine-tuning** with 120× LR ratio (ULMFiT-inspired)
- **Chinchilla scaling demonstration**: 55M tokens / 23M params ≈ 2.4:1
- **~20.7 hours total training** across 13 experiments

---

## 4. Complete Experiment Timeline

| # | Experiment | Params | Data | Test PPL | Test Acc | Key Innovation |
|---|-----------|--------|------|----------|----------|----------------|
| 1 | Word-level baseline | ~7.3M | Shakespeare (1.1M tok) | 393.0 | 18.78% | FastText embeddings |
| 2 | BPE v1 | ~6.4M | Shakespeare (1.1M tok) | 267.5 | 19.40% | Switched to BPE (4K vocab) |
| 3 | BPE v2 | ~6.4M | Shakespeare (1.1M tok) | 245.3 | 20.15% | Hyperparameter tuning |
| 4 | BPE v3 | ~6.4M | Shakespeare (1.1M tok) | 235.8 | 20.45% | Larger BPE vocab (5K) |
| 5 | BPE v4 | ~6.4M | Shakespeare (1.1M tok) | 229.7 | 20.80% | nanoGPT optimizations |
| 6 | AWD-LSTM | ~10M | Shakespeare (1.1M tok) | ~178 | — | Merity et al. recipe |
| 7 | Pre+Fine v1 | 7.3M | 19 books (5.7M) → Shakes | 191.9 | 22.53% | Transfer learning |
| 8 | Pre+Fine v2 | 7.3M | 19 books (5.7M) → Shakes | 178.9 | 23.33% | Discriminative LR |
| 9 | Pre-train v3 | 7.3M | 324 books (55M tok) | 134.9* | — | *Abandoned — underfitting |
| 10 | Pre-train v4 | **23M** | 324 books (55M tok) | **112.8** | **28.08%** | Scaled model |
| 11 | **Fine-tune v4** | **23M** | v4 → Shakespeare | **146.4** | **25.59%** | **BEST — discrim. LR** |
| 12 | Fine-tune v5 | 23M | v4 → Shakespeare | 154.2 | 24.91% | Heavier reg. (worse) |
| 13 | Fine-tune v6 | 23M | v4 → Shakespeare | 148.2 | 25.63% | Gradual unfreezing (no gain) |

---

## 5. Architecture Details

### 5.1 Final Model (v4 — 23M params)

```
Decoder-only Transformer:
  Layers:           6
  Heads:            8
  Embedding Dim:    512
  FFN Hidden:       2048
  Activation:       GELU
  Normalization:    Pre-LayerNorm (bias=False)
  Positional Enc:   Sinusoidal (fixed, not learned)
  Weight Tying:     Yes (embedding ↔ output projection)
  Residual Init:    Scaled (0.02 / √(2 × num_layers))
  Tokenizer:        BPE (8,000 vocab)
  Total Params:     ~22,983,680
```

### 5.2 Original Model (v1–v3 — 7.3M params)

```
  Layers: 5, Heads: 6, Embed: 300, FFN: 1024
  BPE Vocab: 5,000 (Shakespeare-only experiments) / 8,000 (transfer learning)
```

### 5.3 Pipeline Flow

```
Input Text → BPE Tokenizer (8K vocab) → Token Embedding (512d) 
→ Sinusoidal Positional Encoding → Embedding Dropout
→ 6× [Pre-LN → Multi-Head Attention → Residual → Pre-LN → FFN → Residual]
→ Final LayerNorm → Weight-Tied Output Projection → Softmax
→ Cross-Entropy with Label Smoothing (α=0.1)
```

---

## 6. Datasets

### 6.1 Shakespeare (Fine-tuning Target)

| Metric | Value |
|--------|-------|
| Source | Project Gutenberg Complete Works |
| Raw Size | 5.4 MB |
| BPE Tokens | 1,370,232 |
| Train/Val/Test | 80% / 10% / 10% (sequential split) |

### 6.2 Gutenberg Corpus (Pre-training)

| Phase | Books | Raw Size | BPE Tokens |
|-------|-------|----------|------------|
| Phase 1 (v1–v2) | 19 books | 23.4 MB | 5.7M |
| Phase 2 (v3–v4) | 324 books | 217.5 MB | 55.2M |

Phase 2 coverage: ~150+ authors, 16th–20th century English lit (Dickens, Austen, Twain, Tolstoy, Dostoevsky, Milton, Shelley, Poe, Hugo, Dumas, Plato, etc.)

---

## 7. Training Configurations

### 7.1 Pre-training v4 (324 books, 23M params)
```
LR: 3e-4, Weight Decay: 0.05, Warmup: 2000 steps
Epochs: 10, Batch: 64, Seq Len: 128 → 64 (epoch 8)
Dropout: 0.1, Attn Dropout: 0.05, Label Smoothing: 0.1
Stride: Fixed 128 (no contraction)
Optimizer: AdamW (β₁=0.9, β₂=0.99)
Time: 585.8 min (~9.8 hours)
```

### 7.2 Fine-tuning v4 — BEST
```
LR: 3e-5, Weight Decay: 0.1, Warmup: 150 steps
Epochs: 25, Batch: 64, Seq Len: 128
Dropout: 0.2, Attn Dropout: 0.15, Label Smoothing: 0.1
Stride: Fixed 64
Discriminative LR: decay_factor=2.6 → 120× ratio (top/bottom)
  Embeddings: 3.74e-08 → Top Layer: 3.00e-05
Time: 65.7 min (~1.1 hours)
```

### 7.3 Contracting Stride (used in BPE scratch & v1–v2 transfer)
```
Epochs 1-5: stride=128 (no overlap)
Epochs 6-10: stride=64 (50% overlap)
Epochs 11-15: stride=32 (75% overlap)
Epochs 16+: stride=16 (87.5% overlap)
```

---

## 8. Optimization Techniques Applied

| Technique | Citation | Impact |
|-----------|----------|--------|
| BPE Tokenization | Sennrich et al., 2016 | Eliminated OOV, 40% PPL reduction vs word-level |
| Pre-LayerNorm | Xiong et al., 2020; GPT-2 | Stable training, better gradients |
| Weight Tying | Press & Wolf, 2017 | −2.4M–4.1M params, regularization |
| Scaled Residual Init | GPT-2; nanoGPT | Stable early training |
| Label Smoothing (α=0.1) | Szegedy et al., 2016 | Prevents overconfidence |
| Cosine LR + Warmup | Loshchilov & Hutter, 2017 | Smooth convergence |
| AdamW (β₂=0.99) | Loshchilov & Hutter, 2019; nanoGPT | Faster adaptation |
| Contracting Stride | Custom | Progressive data exposure, 8× more examples |
| Transfer Learning | Standard NLP paradigm | Better init from 324-book corpus |
| Discriminative LR | Howard & Ruder, 2018 (ULMFiT) | 120× layer-wise LR ratio |
| Model Scaling | Hoffmann et al., 2022 (Chinchilla) | 7.3M → 23M for 55M tokens |
| Seq Length Schedule | Custom | 128 → 64 in final pre-train epochs |
| GELU Activation | Hendrycks & Gimpel, 2016 | Smoother gradients than ReLU |

---

## 9. Key Findings for Paper

### 9.1 Chinchilla Scaling
- Shakespeare alone: 1.1M tokens / 6.4M params = 0.17 tok/param (5.8× over-parameterized)
- 324-book pre-train: 55M tokens / 23M params = 2.4 tok/param (healthy regime)
- Scaling both data (17×) and model (3.2×) together was critical

### 9.2 BPE vs Word-Level
- Word-level PPL 393 → BPE PPL 229.7 (42% reduction)
- BPE handles archaic Shakespeare vocabulary via subword decomposition

### 9.3 Transfer Learning Progression
- No pre-train: PPL 229.7 → 19-book pre-train: PPL 178.9 (22% ↓)  
- 19-book pre-train: PPL 178.9 → 324-book pre-train: PPL 146.4 (18% ↓)

### 9.4 Discriminative LR > Gradual Unfreezing
- Discriminative LR (v4): PPL 146.4
- Gradual Unfreezing (v6): PPL 148.2  
- With 120× LR ratio, bottom layers are effectively frozen already

### 9.5 Val-Test Gap is Structural (~24%)
- Sequential split puts different plays in val vs test
- Heavier regularisation (v5) didn't help → not overfitting
- Different plays have different archaic language density

### 9.6 AWD-LSTM Comparison
- LSTM (10M params, same data): Val PPL ~178
- Transformer (6.4M params, same data): PPL 229.7
- Transformer + transfer (23M, 55M tokens): PPL 146.4
- Transformer benefits more from scaling and pre-training

---

## 10. Generated Text Samples (Best Model)

**"to be or not to be"** → to be or not to be a man . you are so mad . i am sorry that , i ' ll tell you , but i have been drunk . [ _ within . _ ] now , my lord . enter king . king . where is the duke ? what , what news

**"thou art"** → thou art a creature . i must not know thee ; for i will , though thou hadst a brother , thou shalt have no wife with me . go , away ! thou wilt not be with us again . i ' ll not go with you . [ _exeunt .

**Quality:** Correct stage directions, proper character names (hamlet, falstaff, york), archaic pronouns (thou, thee, thy), coherent dialogue structure.

---

## 11. Model Checkpoints

| File | Params | Description |
|------|--------|-------------|
| `best_model.pt` | ~7.3M | Word-level baseline |
| `best_model_bpe.pt` | ~6.4M | BPE v4 scratch |
| `best_model_lstm.pt` | ~10M | AWD-LSTM baseline |
| `pretrained_gutenberg.pt` | 7.3M | Pre-train v1 (19 books) |
| `pretrained_gutenberg_v2.pt` | 7.3M | Pre-train v2 (19 books, 30ep) |
| `pretrained_gutenberg_v3.pt` | 7.3M | Pre-train v3 (324 books, abandoned) |
| `pretrained_gutenberg_v4.pt` | **23M** | Pre-train v4 (324 books, 10ep) |
| `finetuned_shakespeare.pt` | 7.3M | Fine-tune v1 (uniform LR) |
| `finetuned_shakespeare_v2.pt` | 7.3M | Fine-tune v2 (discrim. LR) |
| `finetuned_shakespeare_v4.pt` | **23M** | **Fine-tune v4 — BEST** |
| `finetuned_shakespeare_v5.pt` | 23M | Fine-tune v5 (heavier reg.) |
| `finetuned_shakespeare_v6.pt` | 23M | Fine-tune v6 (gradual unfreezing) |

---

## 12. References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Press, O., & Wolf, L. (2017). "Using the Output Embedding to Improve Language Models." EACL.
3. Sennrich, R., et al. (2016). "Neural Machine Translation of Rare Words with Subword Units." ACL.
4. Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." ACL.
5. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." (Chinchilla)
6. Merity, S., et al. (2018). "Regularizing and Optimizing LSTM Language Models." ICLR.
7. Karpathy, A. nanoGPT. https://github.com/karpathy/nanoGPT
8. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
9. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." ICLR.
10. Hendrycks, D., & Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)."
11. Szegedy, C., et al. (2016). "Rethinking the Inception Architecture for Computer Vision." CVPR.

---

## 13. Code Structure

```
ProjectNextWord/
├── config.py              # Hyperparameters
├── bpe_tokenizer.py       # BPE tokenization (HuggingFace tokenizers)
├── embeddings.py          # Token + positional embeddings, weight tying
├── transformer.py         # Attention, FFN, decoder blocks
├── model.py               # Full model + text generator
├── data_loader.py         # Dataset, DataLoader, preprocessing
├── train.py               # Training loop (scratch experiments)
├── pretrain_finetune.py   # Transfer learning pipeline
├── gutenberg.py           # Gutenberg corpus downloader (324 books)
├── utils.py               # Helpers (seed, checkpoints, early stopping)
├── main.py                # CLI entry point
├── EXPERIMENT_REFERENCE.md # Full experiment documentation
├── PAPER_CONTEXT.md        # This file (paper-writing context)
├── data/                   # Corpora, tokenizers, caches
├── models/                 # 12 model checkpoints
└── logs/                   # Training history plots
```

---

*Attach this file at the start of any paper-writing session for full context.*
