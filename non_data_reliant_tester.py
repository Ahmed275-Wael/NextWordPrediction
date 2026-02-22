"""
Self-Contained Model Tester — Evaluate All Checkpoints Without External Data

This script evaluates all 12 model checkpoints using an embedded ~58 KB
Shakespeare text sample (compressed to ~30 KB via zlib+base64). It does NOT
require the large data files in /data — only the small BPE tokenizer JSON
files (~300-500 KB each) and the model checkpoints in /models.

Metrics produced: Loss, Perplexity, Accuracy, Top-5 Accuracy, tokens/sec.
Optionally generates sample text from configurable seed prompts.

Dependencies on /data (all small, <600 KB each):
  - data/bpe_tokenizer_5000.json        (used by: best_model_bpe, best_model_lstm)
  - data/bpe_tokenizer_pretrain_8000.json (used by: pretrained v1/v2, finetuned v1/v2)
  - data/bpe_tokenizer_expanded_8000.json (used by: pretrained v3/v4, finetuned v4/v5/v6)

NOT required:
  - data/shakespeare_full.txt (5.5 MB)
  - data/gutenberg_*.txt (23 MB - 1.8 GB)
  - data/embeddings_cache.pt (15 MB)
  - data/gutenberg_chinchilla.tokens.npy (1.7 GB)

Usage:
    python non_data_reliant_tester.py                     # Test all models
    python non_data_reliant_tester.py --model best_model_bpe.pt  # One model
    python non_data_reliant_tester.py --generate           # Include generation
    python non_data_reliant_tester.py --export results.csv # Export CSV
    python non_data_reliant_tester.py --export-json r.json # Export JSON
"""

import argparse
import base64
import csv
import json
import math
import time
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
from bpe_tokenizer import BPETokenizer, BPEVocabulary
from model import ShakespeareTransformer, TextGenerator, create_model


# =============================================================================
# EMBEDDED SHAKESPEARE TEST CORPUS  (~58 KB, zlib-compressed + base64)
#
# Source: last 10 % of the Gutenberg Complete Works of Shakespeare
#         (same region as the 80/10/10 test split used during training)
# =============================================================================
_SHAKESPEARE_B64 = (
    "eNqNvUuSI1eSJTrHKhBzhC+APnAJkpEMr2IwKAwnKSkpLRQDcB0wusEMNDM4iBzlImpSIlUreEt4gzd4O8kN9BZazzmq167BPbKr"
    "RboyCLfPtfvR79Gj//v//f/6y2p5v6wOy2rdpOXYLR/rplnW4/Jcj/vl4bIYU9UPt8v6cTnuk/3abpfn1Kfltju3uHXTnZrtctvX"
    "z4lXrLsq37wc6t1+uFksfnz3w8P9D59uFt90h7Syew5pWZ0re/eham/sKedqWA6pHTGCTWUjsEclu/H7dz988/5m8bnuV/rdbjzv"
    "7Q3jvjstt1X/z3/85zCWb/i15t321113lx/wa2oajPaMv+46u+Fvv73/M53a8ea3/7VYfP7m/Q/vl/e/3Cw/1g1G9G7Zd91hWbf8"
    "pm9PTwkvWh6rptqkxeJ9O6Z++UvV2JDr1r7oc90819Vq+bA/9XW3rGyWPh9T2tqLPt9//8v9O/uG1D9X7fhmsfjl3ffvMVob18d6"
    "GPs0DDbSzz++f/+t/VIN9mw8sY+nPfY218Oya5eX7nQzu/+dTeG6s/9TjxzgY9cvm+4ZU+fP+6Ebl93jyzs/PWKBDv7+Fb6zne6y"
    "hz1wmXddt8XNy6e22zyl7XJfH15+1IqXVHb9UG3n77lvtzYPWOhtdcASDCkdlgOW4OHDzz/df8JD7Bc8YMTKxpPabrybPelDdWwu"
    "9oDt7N7O/hubECvymOpxmL9ef9fXxz2/4jUcxr1eyQPw4n2/1kO6vqtuh7FqNwlTip2x6dqxr/rL/M6/dqfelqJpLsX972xP7Lvz"
    "8o9TNyZ+pc0/L5rffO9X1HgbruuXv6f+qW6LZ328+G/LelhW9oUnO73j/Mu15bGsNuu2nrbxdZ2eeT28D935zbSu+NiVbeMd5MO0"
    "Fe+W33Iyl5u9/Q2f39iz5uP/DpLAdsmySdVzygtvk8WxPkGC2OzZEw6pSV35WQ+YYvv/Jjs62wEHXAqRZBvId3+/XDfYkbZs7bLB"
    "m2KOqrq/uV4Ee9YztmRt+2/g3/NiXPjLiid127UQfd2Kx2fc2zjH+pCu9jFOwJmSkROL5y3xjqoxQTYskz0IBwazs047LlfM5juT"
    "qvaKZ5vydMHHn7t+a2duZ+KjSQesEMbxx6nePNkWH/Y8s4/zAeBI2sjq+Xk6J87FE3fjziakvylXscOkj1zMQUf15U4dUvNofz6n"
    "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"
    "PLJTGgj/BxZpD8Y="
)



def _decompress_shakespeare() -> str:
    """Decompress the embedded Shakespeare sample."""
    raw = base64.b64decode(_SHAKESPEARE_B64)
    return zlib.decompress(raw).decode("utf-8")


# =============================================================================
# LIGHTWEIGHT DATASET (reuses ShakespeareDataset interface)
# =============================================================================

class EmbeddedShakespeareDataset(Dataset):
    """
    PyTorch Dataset from pre-encoded token IDs.
    Uses stride=1 by default to maximise evaluation sequences from the
    limited embedded text.
    """

    def __init__(self, encoded: List[int], seq_length: int = 128, stride: int = 1):
        self.encoded = encoded
        self.seq_length = seq_length
        self.stride = stride
        self.starts = list(range(0, len(encoded) - seq_length, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.starts[idx]
        inp = self.encoded[s : s + self.seq_length]
        tgt = self.encoded[s + 1 : s + self.seq_length + 1]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# =============================================================================
# MODEL REGISTRY  (identical to tester.py)
# =============================================================================

MODEL_REGISTRY = OrderedDict({
    # -- Word-level baseline -------------------------------------------------
    "best_model.pt": {
        "display_name": "Word-Level Baseline",
        "experiment": "#1",
        "tokenizer": "word",
        "num_layers": 5,
        "num_heads": 6,
        "embed_dim": 300,
        "ffn_hidden_dim": 1024,
        "max_seq_length": 64,
        "bpe_vocab_size": None,
        "bpe_path": None,
        "use_bias": True,
        "description": "FastText-initialised word embeddings, 5L/6H/300d",
    },
    # -- BPE scratch ---------------------------------------------------------
    "best_model_bpe.pt": {
        "display_name": "BPE v4 (Scratch)",
        "experiment": "#5",
        "tokenizer": "bpe",
        "num_layers": 5,
        "num_heads": 6,
        "embed_dim": 300,
        "ffn_hidden_dim": 1024,
        "max_seq_length": 128,
        "bpe_vocab_size": 5000,
        "bpe_path": config.DATA_DIR / "bpe_tokenizer_5000.json",
        "description": "BPE-5K, nanoGPT optimisations, scratch training",
    },
    # -- AWD-LSTM ------------------------------------------------------------
    "best_model_lstm.pt": {
        "display_name": "AWD-LSTM Baseline",
        "experiment": "#6",
        "tokenizer": "lstm",
        "num_layers": None,
        "num_heads": None,
        "embed_dim": 300,
        "ffn_hidden_dim": None,
        "max_seq_length": 128,
        "bpe_vocab_size": 5000,
        "bpe_path": config.DATA_DIR / "bpe_tokenizer_5000.json",
        "description": "Merity et al. AWD-LSTM, 3-layer, weight drop",
    },
    # -- Pre-trained Gutenberg (19 books, 7.3M) ------------------------------
    "pretrained_gutenberg.pt": {
        "display_name": "Pre-train v1 (19 books)",
        "experiment": "#7-pt",
        "tokenizer": "bpe",
        "num_layers": 5,
        "num_heads": 6,
        "embed_dim": 300,
        "ffn_hidden_dim": 1024,
        "max_seq_length": 128,
        "bpe_vocab_size": 8000,
        "bpe_path": config.DATA_DIR / "bpe_tokenizer_pretrain_8000.json",
        "description": "Pre-trained on 19 Gutenberg books (5.7M tokens)",
    },
    "pretrained_gutenberg_v2.pt": {
        "display_name": "Pre-train v2 (19 books, 30ep)",
        "experiment": "#8-pt",
        "tokenizer": "bpe",
        "num_layers": 5,
        "num_heads": 6,
        "embed_dim": 300,
        "ffn_hidden_dim": 1024,
        "max_seq_length": 128,
        "bpe_vocab_size": 8000,
        "bpe_path": config.DATA_DIR / "bpe_tokenizer_pretrain_8000.json",
        "description": "Extended pre-training (30 epochs, contracting stride)",
    },
    # -- Pre-trained Gutenberg (324 books) -----------------------------------
    "pretrained_gutenberg_v3.pt": {
        "display_name": "Pre-train v3 (324 books, 7.3M)",
        "experiment": "#9",
        "tokenizer": "bpe",
        "num_layers": 5,
        "num_heads": 6,
        "embed_dim": 300,
        "ffn_hidden_dim": 1024,
        "max_seq_length": 128,
        "bpe_vocab_size": 8000,
        "bpe_path": config.DATA_DIR / "bpe_tokenizer_expanded_8000.json",
        "description": "324 books, 7.3M params - abandoned (model too small)",
    },
    "pretrained_gutenberg_v4.pt": {
        "display_name": "Pre-train v4 (324 books, 23M)",
        "experiment": "#10",
        "tokenizer": "bpe",
        "num_layers": 6,
        "num_heads": 8,
        "embed_dim": 512,
        "ffn_hidden_dim": 2048,
        "max_seq_length": 128,
        "bpe_vocab_size": 8000,
        "bpe_path": config.DATA_DIR / "bpe_tokenizer_expanded_8000.json",
        "description": "Scaled 23M model, 324 Gutenberg books (55M tokens)",
    },
    # -- Fine-tuned Shakespeare (7.3M, 19-book pretrain) ---------------------
    "finetuned_shakespeare.pt": {
        "display_name": "Fine-tune v1 (Uniform LR)",
        "experiment": "#7",
        "tokenizer": "bpe",
        "num_layers": 5,
        "num_heads": 6,
        "embed_dim": 300,
        "ffn_hidden_dim": 1024,
        "max_seq_length": 128,
        "bpe_vocab_size": 8000,
        "bpe_path": config.DATA_DIR / "bpe_tokenizer_pretrain_8000.json",
        "description": "Pre-train v1 -> Shakespeare, uniform fine-tuning LR",
    },
    "finetuned_shakespeare_v2.pt": {
        "display_name": "Fine-tune v2 (Discrim. LR, 7.3M)",
        "experiment": "#8",
        "tokenizer": "bpe",
        "num_layers": 5,
        "num_heads": 6,
        "embed_dim": 300,
        "ffn_hidden_dim": 1024,
        "max_seq_length": 128,
        "bpe_vocab_size": 8000,
        "bpe_path": config.DATA_DIR / "bpe_tokenizer_pretrain_8000.json",
        "description": "Pre-train v2 -> Shakespeare, discriminative LR (ULMFiT)",
    },
    # -- Fine-tuned Shakespeare (23M, 324-book pretrain) ---------------------
    "finetuned_shakespeare_v4.pt": {
        "display_name": "Fine-tune v4 (Discrim. LR, 23M) * BEST",
        "experiment": "#11",
        "tokenizer": "bpe",
        "num_layers": 6,
        "num_heads": 8,
        "embed_dim": 512,
        "ffn_hidden_dim": 2048,
        "max_seq_length": 128,
        "bpe_vocab_size": 8000,
        "bpe_path": config.DATA_DIR / "bpe_tokenizer_expanded_8000.json",
        "description": "BEST - 23M params, 324-book pre-train, discriminative LR",
    },
    "finetuned_shakespeare_v5.pt": {
        "display_name": "Fine-tune v5 (Heavier Reg.)",
        "experiment": "#12",
        "tokenizer": "bpe",
        "num_layers": 6,
        "num_heads": 8,
        "embed_dim": 512,
        "ffn_hidden_dim": 2048,
        "max_seq_length": 128,
        "bpe_vocab_size": 8000,
        "bpe_path": config.DATA_DIR / "bpe_tokenizer_expanded_8000.json",
        "description": "v4 + heavier dropout/stride - performed worse",
    },
    "finetuned_shakespeare_v6.pt": {
        "display_name": "Fine-tune v6 (Gradual Unfreezing)",
        "experiment": "#13",
        "tokenizer": "bpe",
        "num_layers": 6,
        "num_heads": 8,
        "embed_dim": 512,
        "ffn_hidden_dim": 2048,
        "max_seq_length": 128,
        "bpe_vocab_size": 8000,
        "bpe_path": config.DATA_DIR / "bpe_tokenizer_expanded_8000.json",
        "description": "v4 + gradual unfreezing (ULMFiT) - no improvement",
    },
})


# =============================================================================
# HELPER UTILITIES
# =============================================================================

class AverageMeter:
    """Running average tracker."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# SELF-CONTAINED TESTER
# =============================================================================

class SelfContainedTester:
    """
    Evaluates all model checkpoints using the embedded Shakespeare sample.

    Requires ONLY:
      - BPE tokenizer JSON files in data/ (~300-500 KB each)
      - Model checkpoints in models/

    Does NOT require:
      - shakespeare_full.txt
      - gutenberg_*.txt
      - embeddings_cache.pt
    """

    DEFAULT_SEEDS = [
        "to be or not to be",
        "the king",
        "love is",
        "thou art",
        "what light through yonder",
    ]

    def __init__(
        self,
        models_dir: Path = config.MODELS_DIR,
        device: torch.device = config.DEVICE,
        batch_size: int = 64,
        label_smoothing: float = 0.0,
        seed: int = 42,
        eval_stride: int = 32,
    ):
        self.models_dir = models_dir
        self.device = device
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.seed = seed
        self.eval_stride = eval_stride

        # Caches
        self._bpe_cache: Dict[str, BPETokenizer] = {}
        self._vocab_cache: Dict[str, BPEVocabulary] = {}
        self._loader_cache: Dict[str, DataLoader] = {}

        # Embedded text (decompressed once)
        self._text: Optional[str] = None

        # Results
        self.results: List[Dict] = []

    # -- text loading -------------------------------------------------------
    def _get_text(self) -> str:
        if self._text is None:
            print("Decompressing embedded Shakespeare sample...")
            self._text = _decompress_shakespeare()
            print(f"  {len(self._text):,} characters ready")
        return self._text

    # -- BPE loading --------------------------------------------------------
    def _get_bpe(self, bpe_path: Path, vocab_size: int) -> BPETokenizer:
        key = str(bpe_path)
        if key not in self._bpe_cache:
            bpe = BPETokenizer(vocab_size=vocab_size)
            bpe.load(bpe_path)
            self._bpe_cache[key] = bpe
        return self._bpe_cache[key]

    def _get_vocab(self, bpe_path: Path, vocab_size: int) -> BPEVocabulary:
        key = str(bpe_path)
        if key not in self._vocab_cache:
            bpe = self._get_bpe(bpe_path, vocab_size)
            self._vocab_cache[key] = BPEVocabulary(bpe)
        return self._vocab_cache[key]

    # -- test loader --------------------------------------------------------
    def _get_test_loader(self, meta: dict) -> Tuple[DataLoader, object]:
        tok = meta["tokenizer"]
        if tok == "word":
            return self._get_word_loader(meta)
        else:  # bpe or lstm
            return self._get_bpe_loader(meta)

    def _get_bpe_loader(self, meta: dict) -> Tuple[DataLoader, BPEVocabulary]:
        bpe_path = meta["bpe_path"]
        seq_len = meta["max_seq_length"]
        key = f"bpe_{bpe_path}_{seq_len}"

        if key in self._loader_cache:
            return self._loader_cache[key], self._vocab_cache[str(bpe_path)]

        vocab = self._get_vocab(bpe_path, meta["bpe_vocab_size"])
        text = self._get_text()
        encoded = vocab.bpe.encode(text)

        print(f"    Encoded sample: {len(encoded):,} tokens "
              f"(stride={self.eval_stride}, seq_len={seq_len})")

        ds = EmbeddedShakespeareDataset(encoded, seq_len, stride=self.eval_stride)
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=self.device.type == "cuda",
        )
        self._loader_cache[key] = loader
        return loader, vocab

    def _get_word_loader(self, meta: dict):
        """
        Word-level model needs the exact vocabulary from training.
        Try to load from embeddings_cache.pt (just the vocab mapping).
        Skip if unavailable (the file is 15 MB).
        """
        cache_path = config.DATA_DIR / "embeddings_cache.pt"
        if not cache_path.exists():
            print("    [SKIP] Word-level model requires data/embeddings_cache.pt "
                  "(15 MB vocab cache). Not found.")
            return None, None

        from data_loader import Vocabulary, ShakespeareTokenizer, ShakespeareDataset

        cache = torch.load(cache_path, weights_only=False)
        vocab = Vocabulary()
        vocab.word_to_idx = cache["word_to_idx"]
        vocab.idx_to_word = cache["idx_to_word"]

        text = self._get_text()
        tokenizer = ShakespeareTokenizer()
        tokens = tokenizer.tokenize(text)
        encoded = vocab.encode(tokens)

        seq_len = meta["max_seq_length"]
        ds = ShakespeareDataset(encoded, seq_len, stride=self.eval_stride)
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=self.device.type == "cuda",
        )
        return loader, vocab

    # -- model loading ------------------------------------------------------
    def _load_transformer(self, ckpt_path: Path, meta: dict, vocab):
        overrides = {
            "NUM_LAYERS": meta["num_layers"],
            "NUM_HEADS": meta["num_heads"],
            "EMBEDDING_DIM": meta["embed_dim"],
            "FFN_HIDDEN_DIM": meta["ffn_hidden_dim"],
            "MAX_SEQ_LENGTH": meta["max_seq_length"],
        }
        saved = {k: getattr(config, k) for k in overrides}
        for k, v in overrides.items():
            setattr(config, k, v)

        try:
            model = create_model(vocab_size=len(vocab), pretrained_embeddings=None, device=self.device)

            if meta.get("use_bias", False):
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                ckpt_keys = set(ckpt["model_state_dict"].keys())
                for name, module in model.named_modules():
                    bias_key = f"{name}.bias"
                    if isinstance(module, nn.Linear) and module.bias is None and bias_key in ckpt_keys:
                        module.bias = nn.Parameter(torch.zeros(module.out_features, device=self.device))
                    elif isinstance(module, nn.LayerNorm) and module.bias is None and bias_key in ckpt_keys:
                        module.bias = nn.Parameter(torch.zeros(module.normalized_shape[0], device=self.device))
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
        finally:
            for k, v in saved.items():
                setattr(config, k, v)

        model.eval()
        return model

    def _load_lstm(self, ckpt_path: Path, meta: dict, vocab):
        try:
            from lstm_model import ShakespeareLSTM
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model = ShakespeareLSTM(vocab_size=len(vocab), embed_dim=meta["embed_dim"])
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"    [SKIP] LSTM load failed: {e}")
            return None

    # -- evaluation ---------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, model, loader, vocab) -> Dict:
        model.eval()
        criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_idx,
            label_smoothing=self.label_smoothing,
        )
        loss_m = AverageMeter()
        correct = top5_correct = total = total_tok = 0
        t0 = time.time()

        for inp, tgt in tqdm(loader, desc="    Evaluating", leave=False):
            inp, tgt = inp.to(self.device), tgt.to(self.device)
            logits = model(inp)
            B, S, V = logits.shape
            lf, tf = logits.view(-1, V), tgt.view(-1)

            loss_m.update(criterion(lf, tf).item(), B)

            mask = tf != vocab.pad_idx
            ml, mt = lf[mask], tf[mask]
            preds = ml.argmax(dim=-1)
            correct += (preds == mt).sum().item()
            if V >= 5:
                top5 = ml.topk(5, dim=-1).indices
                top5_correct += (top5 == mt.unsqueeze(-1)).any(dim=-1).sum().item()
            else:
                top5_correct += (preds == mt).sum().item()
            total += mask.sum().item()
            total_tok += B * S

        elapsed = time.time() - t0
        return {
            "loss": loss_m.avg,
            "perplexity": math.exp(loss_m.avg),
            "accuracy": correct / total * 100 if total else 0.0,
            "top5_accuracy": top5_correct / total * 100 if total else 0.0,
            "tokens_per_sec": total_tok / elapsed if elapsed else 0.0,
            "num_batches": len(loader),
            "total_tokens": total,
            "eval_time_sec": elapsed,
        }

    # -- generation ---------------------------------------------------------
    def generate_samples(self, model, vocab, meta, seeds, max_len=50, temp=0.8):
        if meta["tokenizer"] == "lstm":
            return {s: "[LSTM generation not supported]" for s in seeds}
        old_seq, old_tok = config.MAX_SEQ_LENGTH, config.TOKENIZER_TYPE
        config.MAX_SEQ_LENGTH = meta["max_seq_length"]
        config.TOKENIZER_TYPE = "bpe" if meta["tokenizer"] == "bpe" else "word"
        try:
            gen = TextGenerator(model, vocab, self.device)
            out = {}
            for seed in seeds:
                try:
                    out[seed] = gen.generate(seed, max_length=max_len, temperature=temp,
                                             top_k=50, top_p=0.9, repetition_penalty=1.2)
                except Exception as e:
                    out[seed] = f"[Error: {e}]"
            return out
        finally:
            config.MAX_SEQ_LENGTH = old_seq
            config.TOKENIZER_TYPE = old_tok

    # -- single model test --------------------------------------------------
    def test_model(self, name: str, generate=False, seeds=None) -> Optional[Dict]:
        if name not in MODEL_REGISTRY:
            print(f"  [SKIP] {name} — not in registry")
            return None
        meta = MODEL_REGISTRY[name]
        path = self.models_dir / name
        if not path.exists():
            print(f"  [SKIP] {meta['display_name']} — checkpoint not found")
            return None

        print(f"\n{'-'*70}")
        print(f"  Testing: {meta['display_name']}")
        print(f"  Checkpoint: {name}")
        print(f"  Description: {meta['description']}")
        print(f"{'-'*70}")

        # -- BPE dependency check --
        if meta["bpe_path"] is not None and not meta["bpe_path"].exists():
            print(f"    [SKIP] Missing tokenizer: {meta['bpe_path']}")
            return None

        loader, vocab = self._get_test_loader(meta)
        if loader is None:
            return None

        # load model
        if meta["tokenizer"] == "lstm":
            model = self._load_lstm(path, meta, vocab)
            if model is None:
                return None
        else:
            model = self._load_transformer(path, meta, vocab)

        total_params = sum(p.numel() for p in model.parameters())
        arch = (f"{meta['num_layers']}L/{meta['num_heads']}H/"
                f"{meta['embed_dim']}d/{meta['ffn_hidden_dim']}FFN"
                if meta["num_layers"] else f"LSTM/{meta['embed_dim']}d")

        print(f"    Architecture: {arch}")
        print(f"    Parameters:   {total_params:,}")
        print(f"    Vocab Size:   {len(vocab):,}")
        print(f"    Test Batches: {len(loader):,}")

        metrics = self.evaluate(model, loader, vocab)

        print(f"    {'='*40}")
        print(f"    Loss:          {metrics['loss']:.4f}")
        print(f"    Perplexity:    {metrics['perplexity']:.1f}")
        print(f"    Accuracy:      {metrics['accuracy']:.2f}%")
        print(f"    Top-5 Acc:     {metrics['top5_accuracy']:.2f}%")
        print(f"    Tokens/sec:    {metrics['tokens_per_sec']:,.0f}")
        print(f"    Eval Time:     {metrics['eval_time_sec']:.1f}s")

        result = {
            "checkpoint": name,
            "display_name": meta["display_name"],
            "experiment": meta["experiment"],
            "architecture": arch,
            "tokenizer": meta["tokenizer"],
            "total_params": total_params,
            **metrics,
        }

        if generate:
            gen_seeds = seeds or self.DEFAULT_SEEDS
            samples = self.generate_samples(model, vocab, meta, gen_seeds)
            result["generations"] = samples
            print(f"\n    Sample Generations:")
            for seed, out in samples.items():
                print(f"      Seed: \"{seed}\"")
                print(f"      -> {out[:200]}")
                print()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    # -- test all -----------------------------------------------------------
    def test_all(self, generate=False, seeds=None, models=None) -> List[Dict]:
        set_seed(self.seed)

        print("=" * 70)
        print("SELF-CONTAINED SHAKESPEARE TESTER  (no large data files needed)")
        print("=" * 70)
        print(f"  Device:          {self.device}")
        print(f"  Batch Size:      {self.batch_size}")
        print(f"  Eval Stride:     {self.eval_stride}")
        print(f"  Label Smoothing: {self.label_smoothing}")
        print(f"  Models Dir:      {self.models_dir}")
        print(f"  Generate:        {generate}")

        targets = models or list(MODEL_REGISTRY.keys())
        available = [m for m in targets if (self.models_dir / m).exists()]
        missing = [m for m in targets if not (self.models_dir / m).exists()]

        print(f"\n  Available: {len(available)}/{len(targets)} checkpoints")
        if missing:
            print(f"  Missing:   {', '.join(missing)}")

        self.results = []
        for name in available:
            r = self.test_model(name, generate=generate, seeds=seeds)
            if r is not None:
                self.results.append(r)

        self._print_summary()
        return self.results

    # -- summary ------------------------------------------------------------
    def _print_summary(self):
        if not self.results:
            print("\nNo models were successfully tested.")
            return

        print("\n" + "=" * 100)
        print("SUMMARY  (embedded Shakespeare sample, stride=%d)" % self.eval_stride)
        print("=" * 100)

        hdr = (f"{'#':<5} {'Model':<42} {'Params':>8} {'Loss':>7} "
               f"{'PPL':>8} {'Acc%':>7} {'Top5%':>7} {'Tok/s':>9}")
        print(hdr)
        print("-" * 100)

        for r in sorted(self.results, key=lambda x: x["perplexity"]):
            ps = f"{r['total_params']/1e6:.1f}M"
            nm = r["display_name"][:42]
            print(f"{r['experiment']:<5} {nm:<42} {ps:>8} "
                  f"{r['loss']:>7.4f} {r['perplexity']:>8.1f} "
                  f"{r['accuracy']:>6.2f}% {r['top5_accuracy']:>6.2f}% "
                  f"{r['tokens_per_sec']:>9,.0f}")

        print("-" * 100)
        best = min(self.results, key=lambda x: x["perplexity"])
        print(f"\n* Best Model: {best['display_name']}")
        print(f"  PPL: {best['perplexity']:.1f} | Acc: {best['accuracy']:.2f}% | "
              f"Top-5: {best['top5_accuracy']:.2f}%")

    # -- export -------------------------------------------------------------
    def export_csv(self, path: str):
        if not self.results:
            return
        fields = ["experiment", "display_name", "checkpoint", "architecture",
                   "tokenizer", "total_params", "loss", "perplexity",
                   "accuracy", "top5_accuracy", "tokens_per_sec", "eval_time_sec"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in sorted(self.results, key=lambda x: x["perplexity"]):
                w.writerow(r)
        print(f"\nResults exported to {path}")

    def export_json(self, path: str):
        if not self.results:
            return
        ser = []
        for r in self.results:
            ser.append({k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ser, f, indent=2, ensure_ascii=False)
        print(f"\nResults exported to {path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Self-contained Shakespeare model tester (no large data files needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python non_data_reliant_tester.py                       # Test all
  python non_data_reliant_tester.py --model best_model_bpe.pt
  python non_data_reliant_tester.py --generate
  python non_data_reliant_tester.py --export results.csv
  python non_data_reliant_tester.py --stride 1            # Dense eval (slower)
  python non_data_reliant_tester.py --stride 64           # Fast eval
        """,
    )
    p.add_argument("--model", nargs="*", default=None, help="Checkpoint name(s)")
    p.add_argument("--generate", action="store_true", help="Generate text samples")
    p.add_argument("--seeds", nargs="*", default=None, help="Generation prompts")
    p.add_argument("--export", default=None, help="CSV output path")
    p.add_argument("--export-json", default=None, help="JSON output path")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--stride", type=int, default=32,
                   help="Eval stride (lower=more sequences, slower; default 32)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    tester = SelfContainedTester(
        batch_size=args.batch_size,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        eval_stride=args.stride,
    )

    tester.test_all(
        generate=args.generate,
        seeds=args.seeds,
        models=args.model,
    )

    if args.export:
        tester.export_csv(args.export)
    if args.export_json:
        tester.export_json(args.export_json)


if __name__ == "__main__":
    main()
