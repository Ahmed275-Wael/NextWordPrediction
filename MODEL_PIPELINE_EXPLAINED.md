# Shakespeare Next-Word Prediction — Model Pipeline Explained

## The Big Picture

We feed Shakespeare text into a Transformer model that learns to predict the next word.  
Given **"The cat sat on the"**, the model learns to predict **"mat"** (or similar).

---

## Step-by-Step Pipeline

### 1. Raw Text → Tokens

Shakespeare's text is split into individual words (tokens):

```
"To be, or not to be" → ["to", "be", ",", "or", "not", "to", "be"]
```

Each word gets a unique ID number from our vocabulary (12,481 words):

```
"to" → 5,  "be" → 12,  "," → 3,  "or" → 47,  "not" → 31
```

### 2. Token IDs → Embeddings (300 dimensions)

Each ID maps to a 300-dimensional vector from FastText (pre-trained on Wikipedia):

```
"to"  (ID 5)  → [0.12, -0.34, 0.56, ..., 0.23]   (300 numbers)
"be"  (ID 12) → [0.45, 0.21, -0.18, ..., 0.67]    (300 numbers)
```

**Why 300?** FastText was trained with 300 dimensions. Words with similar meanings  
have similar vectors (e.g., "king" and "queen" are close in this 300-d space).

**Why FastText?** Shakespeare uses archaic words like "thou", "hath", "doth".  
FastText can handle these because it builds vectors from word **parts** (sub-words),  
so even rare words get meaningful representations.

### 3. Add Positional Encoding

Transformers process all words **simultaneously** (unlike RNNs which go word-by-word).  
This means they don't naturally know word ORDER. Positional encoding fixes this  
by adding a unique pattern to each position:

```
Position 0: PE = [sin(0), cos(0), sin(0), cos(0), ...]
Position 1: PE = [sin(1), cos(1), sin(1/10000^(2/300)), ...]
Position 2: PE = [sin(2), cos(2), sin(2/10000^(2/300)), ...]
```

The addition combines meaning + position:

```
x("be" at position 1) = embedding("be") + PE(position 1)
                       = [word meaning] + [position signal]
                       = [0.45+sin(1), 0.21+cos(1), ...]
```

Now the model knows WHAT the word is AND WHERE it is.

### 4. Create Q, K, V (Query, Key, Value)

From the same input x, three **different** linear projections create three vectors:

```
Q = x × W_Q    →  "What am I looking for?"
K = x × W_K    →  "What do I offer to others?"  
V = x × W_V    →  "What information do I carry?"
```

Each W is a learned 300×300 weight matrix. Same input, different weights → different outputs.

**Analogy**: Imagine a library:
- **Q** (Query) = the question you ask ("I need books about cats")
- **K** (Key) = the label on each book's spine ("Animals", "Cooking", "History")
- **V** (Value) = the actual content inside each book

### 5. Multi-Head Attention (6 heads × 50 dims)

Q, K, and V (each 300 dims) are **reshaped** into 6 heads of 50 dims:

```
Q [300] → Q₁[50], Q₂[50], Q₃[50], Q₄[50], Q₅[50], Q₆[50]
K [300] → K₁[50], K₂[50], K₃[50], K₄[50], K₅[50], K₆[50]
V [300] → V₁[50], V₂[50], V₃[50], V₄[50], V₅[50], V₆[50]
```

**Why split?** Each head applies softmax INDEPENDENTLY. This is the key insight.  
A single head can only produce ONE attention pattern (e.g., "sat" mostly looks  
at "The"). With 6 heads, the model can simultaneously attend to:

- Head 1: subject-verb links ("sat" → "cat")
- Head 2: article patterns ("sat" → "The")
- Head 3: self-reference ("sat" → "sat")
- Head 4: nearby words
- Head 5: punctuation patterns
- Head 6: long-range dependencies

**The math (per head)**:

```
scores = Q_h × K_h^T / √50          ← dot product, scaled
weights = softmax(scores)            ← convert to probabilities
output  = weights × V_h              ← weighted combination of values
```

**Why splitting ≠ full dot product**: The raw dot products sum to the same total,  
BUT softmax is non-linear: softmax(a+b) ≠ softmax(a) + softmax(b).  
Each head's softmax creates its OWN probability distribution independently.

### 6. Concatenate + Output Projection

All 6 head outputs (50 dims each) are concatenated back to 300 dims,  
then mixed through a learned weight matrix W_o:

```
concat = [head1_out | head2_out | ... | head6_out]   → 300 dims
output = concat × W_o                                 → 300 dims
```

W_o learns which combinations of head outputs are most useful.

### 7. Add & Normalize (Residual Connection)

The attention output is ADDED back to the original input (skip connection),  
then normalized:

```
x = LayerNorm(x + attention_output)
```

**Why?** The residual connection helps gradients flow during training  
(prevents vanishing gradients in deep networks). LayerNorm stabilizes values.

### 8. Feed-Forward Network (FFN)

A simple two-layer neural network processes each position independently:

```
FFN(x) = ReLU(x × W₁ + b₁) × W₂ + b₂
         300 → 1024 → 300
```

It expands to 1024 dims (more space to learn complex patterns),  
then compresses back to 300. Another residual + LayerNorm follows.

### 9. Repeat for All Layers

Steps 4–8 form ONE transformer layer. Our model stacks **4 layers**:

```
Layer 1: Basic patterns      (word associations, punctuation)
Layer 2: Grammar patterns    (subject-verb, tense agreement)
Layer 3: Semantic patterns   (meaning, context understanding)
Layer 4: High-level patterns (style, long-range coherence)
```

Each layer has its OWN set of weights — 4 layers × 6 heads = 24 attention patterns total.

### 10. Output Projection → Next Word

The final 300-dim vector is projected to vocabulary size (12,481):

```
logits = final_output × W_embed^T    → [12,481 scores]
probabilities = softmax(logits)       → [12,481 probabilities]
predicted_word = argmax(probabilities) → "mat" (highest probability)
```

**Weight tying**: The output matrix W_embed^T is the SAME as the embedding  
matrix (transposed). This means the model predicts by measuring how "close"  
its output is to each word's embedding — elegant and parameter-efficient.

---

## Model Architecture Summary

```
Input: "The cat sat on the"
  │
  ▼
┌─────────────────────────┐
│   Embedding Layer       │  word → 300-dim vector (from FastText)
│   + Positional Encoding │  + position information
└────────────┬────────────┘
             │
       ┌─────┴─────┐
       │  LAYER 1   │ ─── Multi-Head Attention (6 heads × 50d)
       │            │ ─── Add & LayerNorm
       │            │ ─── Feed-Forward (300 → 1024 → 300)
       │            │ ─── Add & LayerNorm
       ├────────────┤
       │  LAYER 2   │ ─── (same structure, different weights)
       ├────────────┤
       │  LAYER 3   │ ─── (same structure, different weights)
       ├────────────┤
       │  LAYER 4   │ ─── (same structure, different weights)
       └─────┬──────┘
             │
             ▼
┌─────────────────────────┐
│   Output Projection     │  300d → 12,481 vocabulary scores
│   (tied to embeddings)  │
└────────────┬────────────┘
             │
             ▼
        Prediction: "mat" (next word)
```

---

## Key Numbers

| Component              | Value                              |
|------------------------|------------------------------------|
| Vocabulary size        | 12,481 words                       |
| Embedding dimension    | 300 (from FastText)                |
| Positional encoding    | Sinusoidal (fixed, not learned)    |
| Transformer layers     | 4                                  |
| Attention heads        | 6 per layer (24 total)             |
| Head dimension         | 50 (300 ÷ 6)                       |
| FFN hidden dimension   | 1024                               |
| Dropout                | 0.3 (30% of neurons randomly off)  |
| Attention dropout      | 0.25                               |
| Total parameters       | ~7.66 million                      |
| Training tokens        | ~975,000 (Complete Shakespeare)    |
| Sequence length        | 64 words                           |
| Batch size             | 32 sequences                       |

---

## Training Process

### What Happens in ONE Epoch

One epoch = one complete pass through all 975,000 training tokens:

```
975,000 tokens ÷ 64 words per sequence ≈ 15,000 sequences
15,000 sequences ÷ 32 per batch ≈ 470 batches

Each batch:
  1. FORWARD  — Feed 32 sequences through the model → get predictions
  2. LOSS     — Compare predictions to actual next words (cross-entropy)
  3. BACKWARD — Calculate gradients (how much to adjust each weight)
  4. UPDATE   — Adjust weights slightly (AdamW optimizer step)
```

### Loss Function

**Cross-Entropy Loss** with label smoothing (0.15):

```
If true next word is "mat" (index 7421):
  Perfect prediction: [0, 0, ..., 1, ..., 0]     (1.0 at position 7421)
  Smoothed target:    [0.00001, ..., 0.85, ..., 0.00001]  (spread 15% across all words)
  Model output:       [0.01, 0.02, ..., 0.12, ..., 0.003] (learned probabilities)
  
  Loss = how different the model output is from the smoothed target
```

**Label smoothing** prevents the model from becoming overconfident about any  
single answer, which reduces overfitting.

### Learning Rate Schedule

```
LR
│     ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
│    ╱                        ╲
│   ╱   Cosine decay            ╲
│  ╱                              ╲
│ ╱                                ╲
│╱                                  ╲___
└──────────────────────────────────────── Steps
 ↑                                    ↑
 Warmup (2000 steps)                 End (very small LR)
```

- **Warmup**: Start with tiny LR (5e-5), gradually increase to full LR (5e-4)  
- **Cosine decay**: Smoothly decrease LR following a cosine curve  
- **Why?** Large LR early on can destroy pre-trained embeddings. Small LR at the  
  end allows fine-grained convergence.

### Early Stopping

If validation loss doesn't improve for 10 consecutive epochs → stop training  
and reload the best model weights:

```
Epoch 1: Val Loss = 5.63 ← Best! Save. Counter = 0
Epoch 2: Val Loss = 5.50 ← New best! Save. Counter = 0
Epoch 3: Val Loss = 5.55 ← Worse. Counter = 1
Epoch 4: Val Loss = 5.60 ← Worse. Counter = 2
...
Epoch 12: Val Loss = 5.80 ← Worse. Counter = 10 → STOP! Load Epoch 2 weights.
```

### Regularization Techniques (Preventing Overfitting)

| Technique          | What it does                                         |
|--------------------|------------------------------------------------------|
| Dropout (0.3)      | Randomly turns off 30% of neurons during training    |
| Attention dropout  | Randomly zeros 25% of attention weights              |
| Weight decay (0.05)| Penalizes large weights (L2 regularization)          |
| Label smoothing    | Softens target probabilities (prevents overconfidence)|
| Data augmentation  | Randomly swaps words in input (adds noise)           |
| Early stopping     | Stops training when model starts overfitting          |

---

## Text Generation (After Training)

Given a seed phrase, the model generates word by word:

```
Seed: "to be or"

Step 1: Feed "to be or" → model predicts probabilities for next word
        → sample from top words: "not" (chosen)

Step 2: Feed "to be or not" → predict next
        → "to" (chosen)

Step 3: Feed "to be or not to" → predict next
        → "be" (chosen)

Result: "to be or not to be"
```

**Temperature** controls randomness:
- Low (0.3): very predictable, repetitive
- Medium (0.8): balanced creativity (our setting)
- High (1.5): very random, may be incoherent

**Top-k sampling**: Only consider the top 50 most probable words  
**Top-p sampling**: Only consider words whose cumulative probability < 0.9  
**Repetition penalty**: Reduce probability of recently generated words

---

## Why This Architecture for Shakespeare?

1. **FastText embeddings**: Handle archaic words ("thou", "hath") via sub-word vectors
2. **Semantic anchors**: Map archaic → modern words to guide embedding fine-tuning
3. **Transformer (not RNN)**: Captures long-range dependencies in Shakespeare's  
   complex sentence structures (subordinate clauses, inversions)
4. **Multi-head attention**: Different heads can learn different linguistic patterns  
   (syntax, semantics, style) simultaneously
5. **Weight tying**: Reduces parameters and creates a natural "similarity search"  
   for word prediction
