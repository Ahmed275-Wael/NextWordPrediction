"""
Unified Model Tester — Evaluate All Checkpoints on Shakespeare Corpus

Evaluates every saved model checkpoint on the Shakespeare test set with
consistent metrics: Loss, Perplexity, Accuracy, Top-5 Accuracy, and
sample generations.

Supports:
  - Word-level models (best_model.pt)
  - BPE scratch models (best_model_bpe.pt)
  - AWD-LSTM models (best_model_lstm.pt)
  - Pre-trained Gutenberg models (pretrained_gutenberg*.pt)
  - Fine-tuned Shakespeare models (finetuned_shakespeare*.pt)

Usage:
    python tester.py                        # Test all models
    python tester.py --model best_model_bpe # Test a specific model
    python tester.py --generate             # Include text generation samples
    python tester.py --seeds "to be" "love" # Custom generation prompts
    python tester.py --export results.csv   # Export results to CSV
"""

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from bpe_tokenizer import BPETokenizer, BPEVocabulary
from data_loader import ShakespeareDataset, download_shakespeare, prepare_data
from model import ShakespeareTransformer, TextGenerator, create_model
from utils import set_seed, count_parameters, perplexity, AverageMeter


# =============================================================================
# MODEL REGISTRY — maps checkpoint name → architecture + tokenizer metadata
# =============================================================================

MODEL_REGISTRY = OrderedDict({
    # ── Word-level baseline ──────────────────────────────────────────────
    "best_model.pt": {
        "display_name": "Word-Level Baseline",
        "experiment": "#1",
        "tokenizer": "word",
        "num_layers": 5,
        "num_heads": 6,
        "embed_dim": 300,
        "ffn_hidden_dim": 1024,
        "max_seq_length": 64,
        "bpe_vocab_size": None,  # word-level
        "bpe_path": None,
        "use_bias": True,  # legacy checkpoint trained with bias=True
        "description": "FastText-initialised word embeddings, 5L/6H/300d",
    },

    # ── BPE scratch models ───────────────────────────────────────────────
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

    # ── AWD-LSTM baseline ────────────────────────────────────────────────
    "best_model_lstm.pt": {
        "display_name": "AWD-LSTM Baseline",
        "experiment": "#6",
        "tokenizer": "lstm",  # special handling
        "num_layers": None,
        "num_heads": None,
        "embed_dim": 300,
        "ffn_hidden_dim": None,
        "max_seq_length": 128,
        "bpe_vocab_size": 5000,
        "bpe_path": config.DATA_DIR / "bpe_tokenizer_5000.json",
        "description": "Merity et al. AWD-LSTM, 3-layer, weight drop",
    },

    # ── Pre-trained Gutenberg models (19 books, 7.3M) ───────────────────
    "pretrained_gutenberg.pt": {
        "display_name": "Pre-train v1 (19 books)",
        "experiment": "#7-pretrain",
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
        "experiment": "#8-pretrain",
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

    # ── Pre-trained Gutenberg models (324 books) ────────────────────────
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

    # ── Fine-tuned Shakespeare models (7.3M, 19-book pretrain) ──────────
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

    # ── Fine-tuned Shakespeare models (23M, 324-book pretrain) ──────────
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
# TESTER CLASS
# =============================================================================

class ShakespeareTester:
    """
    Unified tester for all model checkpoints on the Shakespeare test corpus.

    Loads each checkpoint with its correct architecture and BPE tokenizer,
    evaluates on the Shakespeare test split, and records:
      Loss, Perplexity, Accuracy, Top-5 Accuracy, tokens/sec throughput.

    Optionally generates sample text from configurable seed prompts.
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
        label_smoothing: float = 0.0,  # 0 for fair evaluation (no smoothing at test)
        seed: int = config.SEED,
    ):
        self.models_dir = models_dir
        self.device = device
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.seed = seed

        # Cache loaded tokenizers to avoid reloading
        self._bpe_cache: Dict[str, BPETokenizer] = {}
        self._test_loader_cache: Dict[str, DataLoader] = {}
        self._vocab_cache: Dict[str, BPEVocabulary] = {}

        # Shakespeare text (loaded once)
        self._shakespeare_text: Optional[str] = None

        # Results
        self.results: List[Dict] = []

    # ── Shakespeare text loading ──────────────────────────────────────
    def _get_shakespeare_text(self) -> str:
        if self._shakespeare_text is None:
            self._shakespeare_text = download_shakespeare()
        return self._shakespeare_text

    # ── BPE tokenizer loading (with cache) ────────────────────────────
    def _get_bpe_tokenizer(self, bpe_path: Path, bpe_vocab_size: int) -> BPETokenizer:
        key = str(bpe_path)
        if key not in self._bpe_cache:
            bpe = BPETokenizer(vocab_size=bpe_vocab_size)
            bpe.load(bpe_path)
            self._bpe_cache[key] = bpe
        return self._bpe_cache[key]

    def _get_vocab(self, bpe_path: Path, bpe_vocab_size: int) -> BPEVocabulary:
        key = str(bpe_path)
        if key not in self._vocab_cache:
            bpe = self._get_bpe_tokenizer(bpe_path, bpe_vocab_size)
            self._vocab_cache[key] = BPEVocabulary(bpe)
        return self._vocab_cache[key]

    # ── Test dataloader (with cache per tokenizer) ────────────────────
    def _get_test_loader(self, meta: dict) -> Tuple[DataLoader, object]:
        """
        Build (or retrieve cached) test DataLoader and vocab for a model's
        tokenizer configuration.

        Returns (test_loader, vocab)
        """
        tok_type = meta["tokenizer"]

        if tok_type == "word":
            return self._get_word_test_loader(meta)
        elif tok_type == "lstm":
            # LSTM uses same BPE tokenizer as BPE scratch
            return self._get_bpe_test_loader(meta)
        else:
            return self._get_bpe_test_loader(meta)

    def _get_bpe_test_loader(self, meta: dict) -> Tuple[DataLoader, BPEVocabulary]:
        bpe_path = meta["bpe_path"]
        key = f"bpe_{bpe_path}_{meta['max_seq_length']}"
        if key in self._test_loader_cache:
            vocab = self._vocab_cache[str(bpe_path)]
            return self._test_loader_cache[key], vocab

        vocab = self._get_vocab(bpe_path, meta["bpe_vocab_size"])
        text = self._get_shakespeare_text()
        encoded = vocab.bpe.encode(text)

        # Same 80/10/10 split as training
        n = len(encoded)
        test_start = int(n * 0.9)
        test_tokens = encoded[test_start:]

        seq_len = meta["max_seq_length"]
        test_ds = ShakespeareDataset(test_tokens, seq_len)
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=self.device.type == "cuda",
        )
        self._test_loader_cache[key] = test_loader
        return test_loader, vocab

    def _get_word_test_loader(self, meta: dict) -> Tuple[DataLoader, object]:
        """Load word-level test data via prepare_data()"""
        key = "word_level"
        if key in self._test_loader_cache:
            return self._test_loader_cache[key], self._vocab_cache.get(key)

        # Temporarily set config to word-level with correct seq length
        old_tok = config.TOKENIZER_TYPE
        old_seq = config.MAX_SEQ_LENGTH
        config.TOKENIZER_TYPE = "word"
        config.MAX_SEQ_LENGTH = meta["max_seq_length"]
        try:
            vocab, _, _, _, _, test_loader, _ = prepare_data(use_cache=True)
        finally:
            config.TOKENIZER_TYPE = old_tok
            config.MAX_SEQ_LENGTH = old_seq

        self._test_loader_cache[key] = test_loader
        self._vocab_cache[key] = vocab
        return test_loader, vocab

    # ── Model loading ─────────────────────────────────────────────────
    def _load_transformer_model(
        self, checkpoint_path: Path, meta: dict, vocab
    ) -> ShakespeareTransformer:
        """Instantiate a Transformer and load checkpoint weights."""

        # Temporarily override config for correct architecture
        overrides = {
            "NUM_LAYERS": meta["num_layers"],
            "NUM_HEADS": meta["num_heads"],
            "EMBEDDING_DIM": meta["embed_dim"],
            "FFN_HIDDEN_DIM": meta["ffn_hidden_dim"],
            "MAX_SEQ_LENGTH": meta["max_seq_length"],
        }
        old_vals = {k: getattr(config, k) for k in overrides}

        for k, v in overrides.items():
            setattr(config, k, v)

        try:
            model = create_model(
                vocab_size=len(vocab),
                pretrained_embeddings=None,
                device=self.device,
            )

            # Legacy checkpoint (best_model.pt) was trained with bias=True
            # in all Linear and LayerNorm layers.  The current code creates
            # bias=False.  Retroactively add bias parameters only where the
            # checkpoint actually has them, so state_dict keys match exactly.
            if meta.get("use_bias", False):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                ckpt_keys = set(checkpoint["model_state_dict"].keys())
                for name, module in model.named_modules():
                    bias_key = f"{name}.bias"
                    if isinstance(module, nn.Linear) and module.bias is None and bias_key in ckpt_keys:
                        module.bias = nn.Parameter(
                            torch.zeros(module.out_features, device=self.device)
                        )
                    elif isinstance(module, nn.LayerNorm) and module.bias is None and bias_key in ckpt_keys:
                        module.bias = nn.Parameter(
                            torch.zeros(module.normalized_shape[0], device=self.device)
                        )
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
        finally:
            for k, v in old_vals.items():
                setattr(config, k, v)

        model.eval()
        return model

    def _load_lstm_model(self, checkpoint_path: Path, meta: dict, vocab):
        """
        Attempt to load the AWD-LSTM model.
        Falls back gracefully if the LSTM module is missing.
        """
        try:
            # Try importing the LSTM model if it exists
            from lstm_model import ShakespeareLSTM  # noqa: F401
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            # Infer architecture from checkpoint
            model = ShakespeareLSTM(
                vocab_size=len(vocab),
                embed_dim=meta["embed_dim"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(self.device)
            model.eval()
            return model
        except (ImportError, ModuleNotFoundError):
            # If no lstm_model.py, try loading the checkpoint state dict
            # and instantiate a generic wrapper
            pass

        # Fallback: try to build from checkpoint keys
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            # The checkpoint might just be a state dict from a custom LSTM
            # We cannot evaluate without the class definition
            print(f"    [SKIP] Cannot load LSTM - lstm_model.py not found")
            return None
        except Exception as e:
            print(f"    [SKIP] Error loading LSTM: {e}")
            return None

    # ── Evaluation ────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        vocab,
    ) -> Dict:
        """
        Evaluate a single model on the Shakespeare test set.

        Returns dict with: loss, perplexity, accuracy, top5_accuracy,
                           tokens_per_sec, num_batches
        """
        model.eval()

        criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_idx,
            label_smoothing=self.label_smoothing,
        )

        loss_meter = AverageMeter()
        correct = 0
        top5_correct = 0
        total = 0
        total_tokens = 0
        start_time = time.time()

        for input_ids, target_ids in tqdm(test_loader, desc="    Evaluating", leave=False):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            logits = model(input_ids)  # (B, S, V)
            B, S, V = logits.shape
            logits_flat = logits.view(-1, V)
            target_flat = target_ids.view(-1)

            # Loss
            loss = criterion(logits_flat, target_flat)
            loss_meter.update(loss.item(), B)

            # Mask padding
            mask = target_flat != vocab.pad_idx
            masked_logits = logits_flat[mask]
            masked_targets = target_flat[mask]

            # Top-1 accuracy
            preds = masked_logits.argmax(dim=-1)
            correct += (preds == masked_targets).sum().item()

            # Top-5 accuracy
            if V >= 5:
                top5_preds = masked_logits.topk(5, dim=-1).indices
                top5_correct += (top5_preds == masked_targets.unsqueeze(-1)).any(dim=-1).sum().item()
            else:
                top5_correct += (preds == masked_targets).sum().item()

            total += mask.sum().item()
            total_tokens += B * S

        elapsed = time.time() - start_time

        return {
            "loss": loss_meter.avg,
            "perplexity": perplexity(loss_meter.avg),
            "accuracy": correct / total * 100 if total > 0 else 0.0,
            "top5_accuracy": top5_correct / total * 100 if total > 0 else 0.0,
            "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
            "num_batches": len(test_loader),
            "total_tokens": total,
            "eval_time_sec": elapsed,
        }

    # ── Generation ────────────────────────────────────────────────────
    def generate_samples(
        self,
        model: nn.Module,
        vocab,
        meta: dict,
        seeds: List[str],
        max_length: int = 50,
        temperature: float = 0.8,
    ) -> Dict[str, str]:
        """Generate text samples from seed prompts."""
        if meta["tokenizer"] == "lstm":
            return {s: "[LSTM generation not supported]" for s in seeds}

        # Temporarily set correct config for generator
        old_seq = config.MAX_SEQ_LENGTH
        old_tok = config.TOKENIZER_TYPE
        config.MAX_SEQ_LENGTH = meta["max_seq_length"]
        config.TOKENIZER_TYPE = "bpe" if meta["tokenizer"] == "bpe" else "word"

        try:
            generator = TextGenerator(model, vocab, self.device)
            samples = {}
            for seed in seeds:
                try:
                    output = generator.generate(
                        seed,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=50,
                        top_p=0.9,
                        repetition_penalty=1.2,
                    )
                    samples[seed] = output
                except Exception as e:
                    samples[seed] = f"[Error: {e}]"
            return samples
        finally:
            config.MAX_SEQ_LENGTH = old_seq
            config.TOKENIZER_TYPE = old_tok

    # ── Main test runner ──────────────────────────────────────────────
    def test_model(
        self,
        checkpoint_name: str,
        generate: bool = False,
        seeds: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """
        Test a single model checkpoint.

        Args:
            checkpoint_name: Filename of the checkpoint (e.g. 'best_model_bpe.pt')
            generate: Whether to generate text samples
            seeds: Seed prompts for generation

        Returns:
            Dict with model name, metrics, and optionally generated samples
        """
        if checkpoint_name not in MODEL_REGISTRY:
            print(f"  [SKIP] {checkpoint_name} - not in registry")
            return None

        meta = MODEL_REGISTRY[checkpoint_name]
        checkpoint_path = self.models_dir / checkpoint_name

        if not checkpoint_path.exists():
            print(f"  [SKIP] {meta['display_name']} - checkpoint not found")
            return None

        print(f"\n{'-' * 70}")
        print(f"  Testing: {meta['display_name']}")
        print(f"  Checkpoint: {checkpoint_name}")
        print(f"  Description: {meta['description']}")
        print(f"{'-' * 70}")

        # Load test data + vocab
        test_loader, vocab = self._get_test_loader(meta)

        # Load model
        if meta["tokenizer"] == "lstm":
            model = self._load_lstm_model(checkpoint_path, meta, vocab)
            if model is None:
                return None
        else:
            model = self._load_transformer_model(checkpoint_path, meta, vocab)

        # Count params
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Architecture string
        if meta["num_layers"] is not None:
            arch_str = (
                f"{meta['num_layers']}L/{meta['num_heads']}H/"
                f"{meta['embed_dim']}d/{meta['ffn_hidden_dim']}FFN"
            )
        else:
            arch_str = f"LSTM/{meta['embed_dim']}d"

        print(f"    Architecture: {arch_str}")
        print(f"    Parameters:   {total_params:,}")
        print(f"    Vocab Size:   {len(vocab):,}")
        print(f"    Test Batches: {len(test_loader):,}")

        # Evaluate
        metrics = self.evaluate_model(model, test_loader, vocab)

        print(f"    {'=' * 40}")
        print(f"    Loss:          {metrics['loss']:.4f}")
        print(f"    Perplexity:    {metrics['perplexity']:.1f}")
        print(f"    Accuracy:      {metrics['accuracy']:.2f}%")
        print(f"    Top-5 Acc:     {metrics['top5_accuracy']:.2f}%")
        print(f"    Tokens/sec:    {metrics['tokens_per_sec']:,.0f}")
        print(f"    Eval Time:     {metrics['eval_time_sec']:.1f}s")

        result = {
            "checkpoint": checkpoint_name,
            "display_name": meta["display_name"],
            "experiment": meta["experiment"],
            "architecture": arch_str,
            "tokenizer": meta["tokenizer"],
            "total_params": total_params,
            "trainable_params": trainable_params,
            **metrics,
        }

        # Optional generation
        if generate:
            gen_seeds = seeds or self.DEFAULT_SEEDS
            samples = self.generate_samples(model, vocab, meta, gen_seeds)
            result["generations"] = samples

            print(f"\n    Sample Generations:")
            for seed, output in samples.items():
                print(f"      Seed: \"{seed}\"")
                print(f"      -> {output[:200]}")
                print()

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def test_all(
        self,
        generate: bool = False,
        seeds: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Test all registered model checkpoints.

        Args:
            generate: Whether to include text generation
            seeds: Custom seed prompts
            models: Optional list of specific checkpoint names to test

        Returns:
            List of result dicts (one per model)
        """
        set_seed(self.seed)

        print("=" * 70)
        print("SHAKESPEARE MODEL TESTER - Evaluating All Checkpoints")
        print("=" * 70)
        print(f"  Device:          {self.device}")
        print(f"  Batch Size:      {self.batch_size}")
        print(f"  Label Smoothing: {self.label_smoothing}")
        print(f"  Models Dir:      {self.models_dir}")
        print(f"  Generate:        {generate}")

        # Discover available checkpoints
        targets = models or list(MODEL_REGISTRY.keys())
        available = [m for m in targets if (self.models_dir / m).exists()]
        missing = [m for m in targets if not (self.models_dir / m).exists()]

        print(f"\n  Available: {len(available)}/{len(targets)} checkpoints")
        if missing:
            print(f"  Missing:   {', '.join(missing)}")

        self.results = []
        for name in available:
            result = self.test_model(name, generate=generate, seeds=seeds)
            if result is not None:
                self.results.append(result)

        # Print summary table
        self._print_summary()

        return self.results

    # ── Summary printing ──────────────────────────────────────────────
    def _print_summary(self):
        """Print a comparison table of all tested models."""
        if not self.results:
            print("\nNo models were successfully tested.")
            return

        print("\n" + "=" * 100)
        print("SUMMARY -- All Models on Shakespeare Test Set")
        print("=" * 100)

        # Header
        header = (
            f"{'#':<4} {'Model':<42} {'Params':>8} {'Loss':>7} "
            f"{'PPL':>8} {'Acc%':>7} {'Top5%':>7} {'Tok/s':>9}"
        )
        print(header)
        print("-" * 100)

        # Sort by perplexity (best first)
        sorted_results = sorted(self.results, key=lambda r: r["perplexity"])

        for i, r in enumerate(sorted_results, 1):
            params_str = f"{r['total_params']/1e6:.1f}M"
            name = r["display_name"]
            if len(name) > 42:
                name = name[:39] + "..."

            line = (
                f"{r['experiment']:<4} {name:<42} {params_str:>8} "
                f"{r['loss']:>7.4f} {r['perplexity']:>8.1f} "
                f"{r['accuracy']:>6.2f}% {r['top5_accuracy']:>6.2f}% "
                f"{r['tokens_per_sec']:>9,.0f}"
            )
            print(line)

        print("-" * 100)

        # Best model callout
        best = sorted_results[0]
        print(f"\n* Best Model: {best['display_name']}")
        print(f"  Test PPL: {best['perplexity']:.1f} | Accuracy: {best['accuracy']:.2f}% | "
              f"Top-5: {best['top5_accuracy']:.2f}%")

        # Improvement over worst
        worst = sorted_results[-1]
        improvement = (worst["perplexity"] - best["perplexity"]) / worst["perplexity"] * 100
        print(f"  PPL Reduction vs worst ({worst['display_name']}): "
              f"{worst['perplexity']:.1f} -> {best['perplexity']:.1f} ({improvement:.1f}%)")

    # ── Export ────────────────────────────────────────────────────────
    def export_csv(self, path: str):
        """Export results to CSV."""
        if not self.results:
            print("No results to export.")
            return

        import csv

        fields = [
            "experiment", "display_name", "checkpoint", "architecture",
            "tokenizer", "total_params", "loss", "perplexity",
            "accuracy", "top5_accuracy", "tokens_per_sec", "eval_time_sec",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in sorted(self.results, key=lambda x: x["perplexity"]):
                writer.writerow(r)

        print(f"\nResults exported to {path}")

    def export_json(self, path: str):
        """Export results to JSON (includes generation samples if available)."""
        if not self.results:
            print("No results to export.")
            return

        # Make Path objects serializable
        serializable = []
        for r in self.results:
            sr = {}
            for k, v in r.items():
                if isinstance(v, Path):
                    sr[k] = str(v)
                else:
                    sr[k] = v
            serializable.append(sr)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        print(f"\nResults exported to {path}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test all Shakespeare model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tester.py                                    # Test all models
  python tester.py --model best_model_bpe.pt          # Test one model
  python tester.py --generate                         # Include generation
  python tester.py --generate --seeds "to be" "love"  # Custom prompts
  python tester.py --export results.csv               # Export to CSV
  python tester.py --export-json results.json         # Export to JSON
        """,
    )
    parser.add_argument(
        "--model", type=str, nargs="*", default=None,
        help="Specific checkpoint name(s) to test (default: all)",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate text samples from each model",
    )
    parser.add_argument(
        "--seeds", type=str, nargs="*", default=None,
        help="Custom seed prompts for generation",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--export-json", type=str, default=None,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for evaluation (default: 64)",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="Label smoothing for loss computation (default: 0.0 for fair eval)",
    )
    parser.add_argument(
        "--seed", type=int, default=config.SEED,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tester = ShakespeareTester(
        batch_size=args.batch_size,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
    )

    if args.model:
        # Test specific model(s)
        tester.test_all(
            generate=args.generate,
            seeds=args.seeds,
            models=args.model,
        )
    else:
        # Test all available models
        tester.test_all(
            generate=args.generate,
            seeds=args.seeds,
        )

    # Export
    if args.export:
        tester.export_csv(args.export)

    if args.export_json:
        tester.export_json(args.export_json)


if __name__ == "__main__":
    main()
