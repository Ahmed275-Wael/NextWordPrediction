# Shakespeare Next-Word Prediction

> **CSO7013 Machine Learning — Masters in Data Science**
>
> A decoder-only Transformer trained to predict the next word in Shakespeare's Complete Works.
> 13 experiments, 12 checkpoints, ~20.7 hours of training on a single GTX 1660 Ti.

---

## Quick Start — View Results (No Data Required)

The model checkpoints and data files are too large to include. To view the pre-computed test results immediately:

```bash
# No dependencies needed — just Python
python view_results.py
```

This reads `test_results.json` and prints a ranked summary of all 12 models.

---

## Setup — Virtual Environment

```bash
# 1. Create the virtual environment
python -m venv .venv

# 2. Activate it
#    Windows (PowerShell):
.venv\Scripts\Activate.ps1
#    Windows (cmd):
.venv\Scripts\activate.bat
#    Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> All commands below assume the `.venv` is activated.

---

## Run the Full Tester (Requires Checkpoints + Data)

If you have the model checkpoints in `models/` and the data files in `data/`, you can
re-run the full evaluation with `tester.py`:

```bash
# Evaluate all 12 models (~2 minutes on GPU)
python tester.py

# Export results to CSV and JSON
python tester.py --export test_results.csv --export-json test_results.json
```

### Tester Options

```
python tester.py [OPTIONS]

  --model NAME [NAME ...]   Test specific checkpoint(s) only
  --generate                Generate text samples from each model
  --seeds "prompt" ...      Custom seed prompts (used with --generate)
  --export FILE.csv         Export results to CSV
  --export-json FILE.json   Export results to JSON
  --batch-size N            Batch size for evaluation (default: 64)
  --label-smoothing F       Label smoothing (default: 0.0 for fair eval)
  --seed N                  Random seed (default: 42)
```

#### Examples

```bash
# Test a single model
python tester.py --model finetuned_shakespeare_v4.pt

# Test all models and generate text
python tester.py --generate

# Generate with custom prompts
python tester.py --generate --seeds "to be or not" "the king hath"

# Full evaluation with exports
python tester.py --export test_results.csv --export-json test_results.json
```

> **Note:** Model checkpoints (`.pt` files) are not tracked in git due to their size.
> Place them in the `models/` directory before running the tester.

---

## Project Overview

This project explores how far a small Transformer can go on Shakespeare next-word prediction when you systematically improve tokenization, scale data and model size, and apply transfer learning.

### The Journey (13 Experiments)

| # | Experiment | Params | Training Data | Test PPL | Test Acc | What Changed |
|---|-----------|--------|---------------|----------|----------|--------------|
| 1 | Word-level baseline | 8.6M | Shakespeare (1.2M tokens) | 155.5 | 18.64% | FastText embeddings, word-level vocab |
| 2-4 | BPE iterations | 6.4M | Shakespeare (1.4M tokens) | 229.7 - 113.8 | 20.25% | Switched to BPE subwords (5K vocab) |
| 5 | BPE v4 (Scratch) | 6.4M | Shakespeare | 113.8 | 20.25% | nanoGPT-style init, Pre-LN, bias=False |
| 6 | AWD-LSTM baseline | 20.5M | Shakespeare (BPE 5K) | 134.0 | 19.68% | Merity et al. recipe (3-layer LSTM) |
| 7 | Pre-train + Fine-tune v1 | 7.3M | 19 Gutenberg books - Shakespeare | 97.6 | 22.53% | Transfer learning, uniform LR |
| 8 | Pre-train + Fine-tune v2 | 7.3M | 19 books - Shakespeare | 90.5 | 23.33% | Discriminative LR (ULMFiT) |
| 9 | Pre-train v3 (324 books) | 7.3M | 324 books (55M tokens) | - | - | Model too small for data; abandoned |
| 10 | Pre-train v4 (324 books) | 23M | 324 books (55M tokens) | 89.7 | 23.18% | Scaled to 6L/8H/512d/2048FFN |
| 11 | **Fine-tune v4** | **23M** | **v4 - Shakespeare** | **72.1** | **25.59%** | **Best — discriminative LR** |
| 12 | Fine-tune v5 | 23M | v4 - Shakespeare | 76.1 | 24.91% | Heavier regularisation (no gain) |
| 13 | Fine-tune v6 | 23M | v4 - Shakespeare | 73.0 | 25.63% | Gradual unfreezing (marginal) |

**Best model:** Fine-tune v4 — **PPL 72.1**, **Accuracy 25.59%**, **Top-5 Accuracy 47.37%**

### Key Findings

1. **BPE tokenization matters.** Switching from word-level to BPE subwords cut perplexity from 155.5 to 113.8 — the single most impactful change.
2. **Transfer learning from Gutenberg works.** Pre-training on 324 classic English books then fine-tuning on Shakespeare reduced perplexity from 113.8 to 72.1 (37% reduction).
3. **Scaling data and model together is critical.** The 7.3M model couldn't absorb 55M tokens (Experiment #9 stalled), but the 23M model could — validating Chinchilla-style scaling (55M tokens / 23M params = 2.4:1).
4. **Discriminative LR beats uniform LR.** A 120x learning-rate ratio between bottom and top layers (ULMFiT-style) outperformed both uniform LR and gradual unfreezing.
5. **Transformers beat LSTMs at all scales.** The AWD-LSTM (20.5M params, PPL 134.0) loses to a Transformer one-third its size (6.4M params, PPL 113.8).

---

## Architecture

### Best Model (23M params — Experiments #10-13)

```
Input Text
  --> BPE Tokenizer (8,000 vocab, trained on 324 Gutenberg books)
  --> Token Embedding (512d) + Sinusoidal Positional Encoding
  --> Embedding Dropout (p=0.20)
  --> 6x Transformer Decoder Block:
        Pre-LayerNorm --> Multi-Head Self-Attention (8 heads) --> Residual
        Pre-LayerNorm --> FFN (512 --> 2048 --> 512, GELU)    --> Residual
  --> Final LayerNorm
  --> Weight-Tied Output Projection (512 --> 8,000)
  --> Cross-Entropy with Label Smoothing (alpha=0.1)
```

### Small Model (7.3M params — Experiments #1-9)

```
5 layers, 6 heads, 300d embedding, 1024 FFN hidden
BPE vocab: 5,000 (scratch) or 8,000 (transfer)
```

### Design Choices

| Choice | Motivation |
|--------|-----------|
| Pre-LayerNorm | More stable training than Post-LN (Xiong et al., 2020) |
| Weight Tying | Saves 4M params, acts as regulariser (Press & Wolf, 2017) |
| Scaled Residual Init | `std = 0.02 / sqrt(2 * num_layers)` prevents residual stream growth (nanoGPT) |
| bias=False everywhere | Fewer params, cleaner gradients (GPT-2 / PaLM convention) |
| GELU activation | Smoother than ReLU (Hendrycks & Gimpel, 2016) |
| AdamW (beta2=0.99) | Faster adaptation for language modelling (nanoGPT default) |
| Cosine LR + linear warmup | Smooth convergence (Loshchilov & Hutter, 2017) |
| Label Smoothing (0.1) | Prevents overconfident predictions (Szegedy et al., 2016) |

---

## Datasets

### Shakespeare (Fine-tuning Target)

| Property | Value |
|----------|-------|
| Source | Project Gutenberg — Complete Works of William Shakespeare |
| Raw Size | 5.4 MB (~5.36M characters) |
| BPE Tokens | ~1.37M |
| Word Tokens | ~1.22M |
| Split | 80% train / 10% val / 10% test (sequential) |

### Gutenberg Corpus (Pre-training)

| Phase | Books | Raw Size | BPE Tokens | Used In |
|-------|-------|----------|------------|---------|
| Phase 1 | 19 books | 23.4 MB | 5.7M | Experiments #7-8 |
| Phase 2 | 324 books | 217.5 MB | 55.2M | Experiments #9-13 |

Phase 2 covers 150+ authors across 16th-20th century English literature (Dickens, Austen, Twain, Milton, Shelley, Poe, Tolstoy, and many more).

---

## Test Results (All 12 Checkpoints)

Evaluated with `tester.py` using `label_smoothing=0.0` and `batch_size=64`:

| Rank | Model | Params | PPL | Acc% | Top-5% | Tokens/s |
|------|-------|--------|-----|------|--------|----------|
| 1 | Fine-tune v4 (Discrim. LR, 23M) | 23.0M | **72.1** | **25.59** | **47.37** | 42,411 |
| 2 | Fine-tune v6 (Gradual Unfreezing) | 23.0M | 73.0 | 25.63 | 47.34 | 41,978 |
| 3 | Fine-tune v5 (Heavier Reg.) | 23.0M | 76.1 | 24.91 | 46.50 | 42,169 |
| 4 | Pre-train v4 (324 books, 23M) | 23.0M | 89.7 | 23.18 | 44.63 | 43,120 |
| 5 | Fine-tune v2 (Discrim. LR, 7.3M) | 7.3M | 90.5 | 23.33 | 44.71 | 105,691 |
| 6 | Fine-tune v1 (Uniform LR) | 7.3M | 97.6 | 22.53 | 43.56 | 105,918 |
| 7 | Pre-train v3 (324 books, 7.3M) | 7.3M | 103.6 | 21.66 | 42.65 | 102,998 |
| 8 | BPE v4 Scratch | 6.4M | 113.8 | 20.25 | 40.86 | 116,321 |
| 9 | AWD-LSTM Baseline | 20.5M | 134.0 | 19.68 | 38.74 | 47,283 |
| 10 | Pre-train v2 (19 books, 30ep) | 7.3M | 143.9 | 18.67 | 39.48 | 107,685 |
| 11 | Pre-train v1 (19 books) | 7.3M | 151.6 | 17.88 | 38.44 | 103,603 |
| 12 | Word-Level Baseline | 8.6M | 155.5 | 18.64 | 39.27 | 39,093 |

---

## Generated Text Samples (Best Model)

```
Prompt: "to be or not to be"
--> to be or not to be a man . you are so mad . i am sorry
    that , i 'll tell you , but i have been drunk .
    [ _ within . _ ] now , my lord . enter king . king .
    where is the duke ? what , what news

Prompt: "thou art"
--> thou art a creature . i must not know thee ; for i will ,
    though thou hadst a brother , thou shalt have no wife
    with me . go , away ! thou wilt not be with us again .
    i 'll not go with you . [ _exeunt .
```

---

## Project Structure

```
ProjectNextWord/
|-- config.py                 # All hyperparameters and paths
|-- bpe_tokenizer.py          # BPE tokenization (HuggingFace tokenizers)
|-- embeddings.py             # Token + positional embeddings, weight tying
|-- transformer.py            # Multi-head attention, FFN, decoder blocks
|-- model.py                  # Full ShakespeareTransformer + TextGenerator
|-- data_loader.py            # Dataset, DataLoader, word-level preprocessing
|-- train.py                  # Training loop for scratch experiments
|-- pretrain_finetune.py      # Transfer learning pipeline (pre-train + fine-tune)
|-- lstm_model.py             # AWD-LSTM baseline (Merity et al., 2018)
|-- run_lstm_baseline.py      # Entry point for LSTM training
|-- gutenberg.py              # Gutenberg corpus downloader (324 books)
|-- utils.py                  # Seed, checkpoints, early stopping, helpers
|-- main.py                   # CLI entry point (train / generate / evaluate)
|-- tester.py                 # Unified tester for all 12 checkpoints
|-- view_results.py           # View pre-computed results (no data needed)
|-- augmentation.py           # Data augmentation utilities
|-- requirements.txt          # Python dependencies
|-- EXPERIMENT_REFERENCE.md   # Detailed experiment documentation (1,400+ lines)
|-- PAPER_CONTEXT.md          # Paper-writing context brief
|-- MODEL_PIPELINE_EXPLAINED.md # Architecture walkthrough
|-- test_results.csv          # Tester output (CSV)
|-- test_results.json         # Tester output (JSON)
|-- data/                     # Corpora, tokenizers, embedding caches
|-- models/                   # 12 model checkpoints (.pt files)
+-- logs/                     # Training history plots
```

### File Guide

| File | Purpose |
|------|---------|
| `config.py` | Central configuration — hyperparameters, paths, device, semantic anchors |
| `bpe_tokenizer.py` | Trains / loads BPE tokenizers using HuggingFace `tokenizers` library |
| `embeddings.py` | Token embedding + sinusoidal positional encoding; weight tying; semantic anchor loss |
| `transformer.py` | Multi-head self-attention, feed-forward network, Pre-LN decoder blocks, full decoder |
| `model.py` | `ShakespeareTransformer` (embeddings + decoder + output); `TextGenerator` (sampling) |
| `data_loader.py` | Shakespeare tokenizer, `Vocabulary` class, `ShakespeareDataset`, FastText alignment |
| `train.py` | `Trainer` class — training loop with validation, checkpointing, early stopping, LR scheduling |
| `pretrain_finetune.py` | Two-phase pipeline: pre-train on Gutenberg, fine-tune on Shakespeare with discriminative LR |
| `lstm_model.py` | AWD-LSTM with weight dropout, variational dropout, embedding dropout (Merity et al.) |
| `gutenberg.py` | Downloads and cleans 324 Project Gutenberg books for pre-training |
| `tester.py` | Evaluates all 12 checkpoints: loss, perplexity, accuracy, top-5, generation, CSV/JSON export |
| `view_results.py` | Displays pre-computed results from `test_results.json` — no checkpoints or data needed |
| `main.py` | CLI entry point: `--mode train`, `--mode generate`, `--mode evaluate`, `--mode all` |

---

## Training Configurations

### Pre-training v4 (Best pre-train — 324 books, 23M params)

```
LR: 3e-4          Weight Decay: 0.05       Warmup: 2000 steps
Epochs: 10         Batch Size: 64           Seq Length: 128 -> 64
Dropout: 0.1       Attn Dropout: 0.05       Label Smoothing: 0.1
Optimizer: AdamW (beta1=0.9, beta2=0.99)
Time: ~9.8 hours
```

### Fine-tuning v4 (Best overall — discriminative LR)

```
LR: 3e-5          Weight Decay: 0.1        Warmup: 150 steps
Epochs: 25         Batch Size: 64           Seq Length: 128
Dropout: 0.2       Attn Dropout: 0.15       Label Smoothing: 0.1
Discriminative LR: decay_factor=2.6, 120x ratio (top/bottom)
  Embeddings: 3.74e-08  -->  Top Layer: 3.00e-05
Time: ~1.1 hours
```

---

## Optimisation Techniques

| Technique | Reference | Impact |
|-----------|-----------|--------|
| BPE Tokenization | Sennrich et al., 2016 | Eliminated OOV; biggest single PPL drop |
| Pre-LayerNorm | Xiong et al., 2020 | Stable training without careful LR tuning |
| Weight Tying | Press & Wolf, 2017 | Saved 2.4M-4.1M params; implicit regularisation |
| Scaled Residual Init | GPT-2 / nanoGPT | Prevents residual stream explosion at init |
| Label Smoothing | Szegedy et al., 2016 | Soft targets; prevents overconfidence |
| Cosine LR + Warmup | Loshchilov & Hutter, 2017 | Smooth convergence profile |
| AdamW (beta2=0.99) | Loshchilov & Hutter, 2019 | Faster second-moment adaptation |
| Contracting Stride | Custom | Progressive data exposure (stride 128 -> 16) |
| Transfer Learning | Standard NLP paradigm | Better init from 324-book corpus |
| Discriminative LR | Howard & Ruder, 2018 (ULMFiT) | 120x layer-wise LR ratio |
| Model Scaling | Hoffmann et al., 2022 (Chinchilla) | 7.3M -> 23M to match 55M tokens |
| Seq Length Schedule | Custom | 128 -> 64 in final pre-train epochs |

---

## How to Train from Scratch

> Make sure the `.venv` is activated before running any training commands.

### Train a BPE model on Shakespeare only

```bash
python main.py --mode train --epochs 50
```

### Pre-train on Gutenberg + fine-tune on Shakespeare

```bash
# Pre-train (adjust config.py for model size and data paths)
python pretrain_finetune.py --phase pretrain --epochs 10

# Fine-tune
python pretrain_finetune.py --phase finetune --checkpoint models/pretrained_gutenberg_v4.pt --epochs 25
```

### Train the AWD-LSTM baseline

```bash
python run_lstm_baseline.py
```

### Generate text from a trained model

```bash
python main.py --mode generate --prompt "to be or not to be" --temperature 0.8
```

---

## Technical Stack

| Component | Version |
|-----------|---------|
| Python | 3.13.3 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA GeForce GTX 1660 Ti (6 GB) |
| OS | Windows 11 |
| Tokenizer | HuggingFace `tokenizers` library |
| Embeddings | FastText (word-level baseline only) |

---

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Press, O., & Wolf, L. (2017). "Using the Output Embedding to Improve Language Models." *EACL*.
3. Sennrich, R., et al. (2016). "Neural Machine Translation of Rare Words with Subword Units." *ACL*.
4. Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." *ACL*.
5. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." *(Chinchilla)*
6. Merity, S., et al. (2018). "Regularizing and Optimizing LSTM Language Models." *ICLR*.
7. Karpathy, A. nanoGPT. https://github.com/karpathy/nanoGPT
8. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." *ICML*.
9. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR*.
10. Hendrycks, D., & Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)."
11. Szegedy, C., et al. (2016). "Rethinking the Inception Architecture for Computer Vision." *CVPR*.

## 🙏 Acknowledgments

- FastText pre-trained embeddings by Facebook Research
- Shakespeare text from [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017)
