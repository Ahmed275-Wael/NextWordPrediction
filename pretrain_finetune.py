"""
Pre-train on Expanded Project Gutenberg → Fine-tune on Shakespeare

This script implements the pre-training + fine-tuning paradigm:
1. Downloads ~300+ classic English texts from Project Gutenberg (~100MB+)
2. Trains a BPE tokenizer on the combined Gutenberg + Shakespeare corpus
3. Pre-trains a Transformer on the Gutenberg corpus (broad English understanding)
4. Fine-tunes the pre-trained model on Shakespeare (domain specialisation)

Chinchilla Scaling Analysis (Hoffmann et al., 2022):
- Expanded Gutenberg corpus ≈ 25-35M BPE tokens
- Our Transformer: 6.4M params
- Chinchilla-optimal: 20 tokens per param → 128M tokens for 6.4M params
- Actual ratio: ~30M / 6.4M ≈ 4.7:1 (was 0.9:1 with 19 books)
- This is a 5× improvement in data efficiency over the original 19-book corpus
- More data → better generalisation → lower fine-tuned PPL on Shakespeare

Usage:
    python pretrain_finetune.py                    # Full pipeline (pretrain + finetune)
    python pretrain_finetune.py --mode pretrain     # Pre-train only
    python pretrain_finetune.py --mode finetune     # Fine-tune from checkpoint
    python pretrain_finetune.py --mode evaluate     # Evaluate fine-tuned model
    python pretrain_finetune.py --mode generate     # Generate text
"""

import argparse
import math
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

import config
from utils import (
    set_seed, count_parameters, save_checkpoint,
    EarlyStopping, AverageMeter, get_lr, print_model_summary, perplexity
)
from model import create_model, ShakespeareTransformer, TextGenerator
from data_loader import ShakespeareDataset
from bpe_tokenizer import BPETokenizer, BPEVocabulary
from gutenberg import download_gutenberg_corpus, download_shakespeare


# ============================================================================
# PRE-TRAINING CONFIGURATION (overrides for Gutenberg corpus)
# ============================================================================
PRETRAIN_CONFIG = {
    # BPE tokenizer — trained on COMBINED corpus (Gutenberg + Shakespeare)
    "bpe_vocab_size": 8000,        # 8K vocab for large multi-author corpus
    
    # Architecture — SCALED UP for 50M token corpus
    # 50M tokens / 22M params ≈ 2.3:1 tokens/param — healthy regime
    # Fits in 6GB VRAM with batch_size=64, seq_len=128
    "num_layers": 6,
    "num_heads": 8,
    "embed_dim": 512,
    "ffn_hidden_dim": 2048,
    
    # Training — adjusted for larger corpus + bigger model
    "batch_size": 64,
    "learning_rate": 3e-4,         # Slightly lower for bigger model
    "weight_decay": 0.05,
    "warmup_steps": 2000,          # More warmup for much larger dataset
    "num_epochs": 10,              # 7 epochs @ 128 seq_len + 3 epochs @ 64 seq_len
    "patience": 8,                 # Less patience — data-rich regime converges faster
    "dropout": 0.1,                # Lower dropout — lots of data = less overfitting
    "attention_dropout": 0.05,
    "label_smoothing": 0.1,
    "max_seq_length": 128,
    
    # Sequence length schedule: switch to 64 at epoch 8
    # Short sequences in final epochs force the model to learn tighter local patterns
    "seq_len_switch_epoch": 8,     # Switch to short_seq_length at this epoch
    "short_seq_length": 64,        # Shorter seq for final 3 epochs
    
    # Contracting stride — DISABLED for large corpus
    "stride_initial": 128,
    "stride_min": 128,             # Same as initial = NO contraction
    "stride_contract_every": 999,  # Never triggers
    
    # Paths — v4 for scaled-up model
    "model_path": config.MODELS_DIR / "pretrained_gutenberg_v4.pt",
    "bpe_path": config.DATA_DIR / "bpe_tokenizer_expanded_8000.json",
}

FINETUNE_CONFIG = {
    # Fine-tuning — conservative LR to preserve pre-trained representations
    # Pre-training converged at LR ~1e-6, so we start gently at 3e-5
    "learning_rate": 3e-5,         # ~10× pre-train's final LR (top layer)
    "weight_decay": 0.1,           # Stronger regularisation
    "warmup_steps": 150,           # Short warmup — weights already well-initialised
    "num_epochs": 25,              # Enough room for gradual unfreezing schedule
    "patience": 8,                 # Patient — unfreezing creates temporary val spikes
    "dropout": 0.2,                # v4-best dropout setting
    "attention_dropout": 0.15,     # v4-best attention dropout
    "label_smoothing": 0.1,
    
    # Discriminative fine-tuning (ULMFiT, Howard & Ruder 2018)
    # Each layer gets LR = top_lr / (decay_factor ^ distance_from_top)
    # Bottom layers (general English) barely change; top layers (style) adapt fast
    "discriminative_lr": True,
    "lr_decay_factor": 2.6,        # Howard & Ruder's recommended decay factor
    
    # Gradual unfreezing (ULMFiT Phase 2)
    # Start with only top layer + output head unfrozen.
    # Every `unfreeze_every` epochs, unfreeze one more layer from top to bottom.
    # This prevents early gradient noise from corrupting lower pre-trained layers.
    "gradual_unfreezing": True,
    "unfreeze_every": 3,           # Unfreeze next layer every N epochs
    
    # Fixed stride — v4-best setting
    "stride_initial": 64,
    "stride_min": 64,              # Fixed: no contraction
    "stride_contract_every": 999,  # Effectively disabled
    
    # Paths
    "model_path": config.MODELS_DIR / "finetuned_shakespeare_v6.pt",
}


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_pretrain_data(
    gutenberg_text: str,
    shakespeare_text: str,
    cfg: dict
) -> Tuple[BPEVocabulary, list, list, list]:
    """
    Prepare pre-training data:
    1. Train BPE on combined corpus (Gutenberg + Shakespeare)
    2. Encode Gutenberg text only (Shakespeare saved for fine-tuning)
    3. Split into train/val/test
    
    Returns:
        vocab, train_tokens, val_tokens, test_tokens
    """
    print("\n" + "=" * 70)
    print("PREPARING PRE-TRAINING DATA")
    print("=" * 70)
    
    bpe_path = cfg["bpe_path"]
    
    # Train or load BPE tokenizer on COMBINED corpus
    bpe = BPETokenizer(vocab_size=cfg["bpe_vocab_size"])
    if bpe_path.exists():
        bpe.load(bpe_path)
        print(f"Loaded cached BPE tokenizer ({bpe.get_vocab_size()} tokens)")
    else:
        print(f"Training BPE tokenizer on Gutenberg + Shakespeare ({cfg['bpe_vocab_size']} vocab)...")
        combined = gutenberg_text + "\n\n" + shakespeare_text
        bpe.train(combined, save_path=bpe_path)
        print(f"BPE tokenizer trained on {len(combined):,} characters")
    
    vocab = BPEVocabulary(bpe)
    
    # Encode Gutenberg corpus in chunks to avoid OOM on large corpora
    print(f"\nEncoding Gutenberg corpus ({len(gutenberg_text):,} chars)...")
    CHUNK_SIZE = 10_000_000  # 10M chars per chunk
    encoded = []
    for start in range(0, len(gutenberg_text), CHUNK_SIZE):
        chunk = gutenberg_text[start:start + CHUNK_SIZE]
        encoded.extend(bpe.encode(chunk))
        done = min(start + CHUNK_SIZE, len(gutenberg_text))
        print(f"  Encoded {done:,}/{len(gutenberg_text):,} chars "
              f"({100*done/len(gutenberg_text):.0f}%) → {len(encoded):,} tokens so far")
    
    print(f"  Total: {len(encoded):,} tokens")
    print(f"  Compression: {len(gutenberg_text)/len(encoded):.1f} chars/token")
    
    # Split 90/5/5 (more training data, smaller val/test since we have plenty)
    n = len(encoded)
    train_end = int(n * 0.90)
    val_end = int(n * 0.95)
    
    train_tokens = encoded[:train_end]
    val_tokens = encoded[train_end:val_end]
    test_tokens = encoded[val_end:]
    
    print(f"\n  Pre-training splits:")
    print(f"    Train: {len(train_tokens):,} tokens")
    print(f"    Val:   {len(val_tokens):,} tokens")
    print(f"    Test:  {len(test_tokens):,} tokens")
    
    # Chinchilla analysis
    total_tokens = len(train_tokens)
    model_params = _estimate_model_params(cfg)
    ratio = model_params / total_tokens
    chinchilla_optimal = total_tokens / 20
    
    print(f"\n  Chinchilla Analysis:")
    print(f"    Training tokens:     {total_tokens:,}")
    print(f"    Model parameters:    {model_params:,}")
    print(f"    Params-per-token:    {ratio:.2f}")
    print(f"    Chinchilla optimal:  {chinchilla_optimal:,.0f} params")
    print(f"    Our model:           {ratio:.1f}× Chinchilla (acceptable for pre-training)")
    
    return vocab, train_tokens, val_tokens, test_tokens


def prepare_finetune_data(
    shakespeare_text: str,
    bpe: BPETokenizer,
) -> Tuple[list, list, list]:
    """
    Prepare fine-tuning data:
    - Encode Shakespeare text with the SAME BPE tokenizer used for pre-training
    - Split into train/val/test
    
    Returns:
        train_tokens, val_tokens, test_tokens
    """
    print("\n" + "=" * 70)
    print("PREPARING FINE-TUNING DATA (Shakespeare)")
    print("=" * 70)
    
    encoded = bpe.encode(shakespeare_text)
    print(f"  Shakespeare encoded: {len(encoded):,} tokens")
    
    n = len(encoded)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_tokens = encoded[:train_end]
    val_tokens = encoded[train_end:val_end]
    test_tokens = encoded[val_end:]
    
    print(f"  Train: {len(train_tokens):,} tokens")
    print(f"  Val:   {len(val_tokens):,} tokens")
    print(f"  Test:  {len(test_tokens):,} tokens")
    
    return train_tokens, val_tokens, test_tokens


def _estimate_model_params(cfg: dict) -> int:
    """Estimate total model parameters from config"""
    d = cfg["embed_dim"]
    n = cfg["num_layers"]
    ff = cfg["ffn_hidden_dim"]
    v = cfg["bpe_vocab_size"]
    
    # Per block: 4*d*d (attn) + 2*d*ff (FFN) + 2*d (norms)
    per_block = 4 * d * d + 2 * d * ff + 2 * d
    # Embedding (tied): v*d
    # Final norm: d
    total = n * per_block + v * d + d
    return total


def _create_dataloaders(
    train_tokens: list,
    val_tokens: list,
    test_tokens: list,
    cfg: dict,
    stride: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders from token lists"""
    seq_len = cfg["max_seq_length"]
    batch_size = cfg["batch_size"]
    initial_stride = stride or cfg["stride_initial"]
    
    train_ds = ShakespeareDataset(train_tokens, seq_len, stride=initial_stride)
    val_ds = ShakespeareDataset(val_tokens, seq_len)
    test_ds = ShakespeareDataset(test_tokens, seq_len)
    
    pin = config.DEVICE.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# TRAINING LOOP (shared between pre-training and fine-tuning)
# ============================================================================

class PretrainFinetuneTrainer:
    """
    Unified trainer for both pre-training and fine-tuning phases.
    
    Key differences between phases:
    - Pre-training: higher LR, less dropout, Gutenberg data
    - Fine-tuning: lower LR, more dropout, Shakespeare data, loads checkpoint
    """
    
    def __init__(
        self,
        model: ShakespeareTransformer,
        vocab: BPEVocabulary,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        cfg: dict,
        raw_train_data: Optional[list] = None,
        phase: str = "pretrain"
    ):
        self.model = model
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.raw_train_data = raw_train_data
        self.phase = phase
        self.device = config.DEVICE
        
        # Current stride
        self.current_stride = cfg["stride_initial"] if raw_train_data is not None else None
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_idx,
            label_smoothing=cfg["label_smoothing"]
        )
        
        # Optimizer — discriminative LR for fine-tuning, uniform for pre-training
        if cfg.get("discriminative_lr", False):
            self.optimizer = self._create_discriminative_optimizer(model, cfg)
        else:
            self.optimizer = AdamW(
                model.parameters(),
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
                betas=(0.9, 0.99),
                eps=1e-9
            )
        
        # Scheduler
        total_steps = self._estimate_total_steps(cfg["num_epochs"])
        warmup = cfg["warmup_steps"]
        
        warmup_sched = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
        cosine_sched = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup, eta_min=1e-6)
        self.scheduler = SequentialLR(self.optimizer, [warmup_sched, cosine_sched], milestones=[warmup])
        
        print(f"  LR schedule: {total_steps:,} steps (warmup={warmup}, cosine={total_steps-warmup})")
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=cfg["patience"], min_delta=0.001)
        
        # Gradual unfreezing state
        self.gradual_unfreezing = cfg.get("gradual_unfreezing", False) and phase == "finetune"
        self.unfreeze_every = cfg.get("unfreeze_every", 3)
        self.num_layers = PRETRAIN_CONFIG["num_layers"]
        self.unfrozen_from_top = 0  # How many decoder layers are currently unfrozen
        
        if self.gradual_unfreezing:
            self._freeze_all_but_top()
        
        # History
        self.history = {'train_loss': [], 'val_loss': [], 'train_ppl': [], 'val_ppl': [], 'learning_rates': []}
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def _create_discriminative_optimizer(self, model: ShakespeareTransformer, cfg: dict) -> AdamW:
        """
        Discriminative Fine-Tuning (ULMFiT, Howard & Ruder 2018).
        
        Assigns exponentially decaying learning rates from top layers to bottom:
          Layer L (top):    lr = base_lr
          Layer L-1:        lr = base_lr / decay
          Layer L-2:        lr = base_lr / decay^2
          ...
          Embeddings:       lr = base_lr / decay^(L+1)  (barely changes)
        
        This prevents catastrophic forgetting of pre-trained knowledge in
        lower layers while allowing upper layers to adapt to Shakespeare.
        """
        base_lr = cfg["learning_rate"]
        decay = cfg["lr_decay_factor"]
        num_layers = PRETRAIN_CONFIG["num_layers"]
        
        param_groups = []
        
        # Group 1: Embedding layer (lowest LR — general word representations)
        embedding_params = []
        for name, param in model.named_parameters():
            if 'embedding' in name.lower() and param.requires_grad:
                embedding_params.append(param)
        
        emb_lr = base_lr / (decay ** (num_layers + 1))
        if embedding_params:
            param_groups.append({'params': embedding_params, 'lr': emb_lr})
        
        # Group 2-N+1: Each decoder layer (increasing LR from bottom to top)
        for layer_idx in range(num_layers):
            layer_params = []
            layer_name = f'decoder.layers.{layer_idx}.'
            for name, param in model.named_parameters():
                if layer_name in name and param.requires_grad:
                    layer_params.append(param)
            
            # Distance from top: layer 0 (bottom) is furthest, layer N-1 (top) is closest
            distance_from_top = num_layers - 1 - layer_idx
            layer_lr = base_lr / (decay ** distance_from_top)
            
            if layer_params:
                param_groups.append({'params': layer_params, 'lr': layer_lr})
        
        # Group N+2: Final LayerNorm + any remaining params (highest LR)
        assigned = set()
        for group in param_groups:
            for p in group['params']:
                assigned.add(id(p))
        
        remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in assigned]
        if remaining:
            param_groups.append({'params': remaining, 'lr': base_lr})
        
        optimizer = AdamW(
            param_groups,
            weight_decay=cfg["weight_decay"],
            betas=(0.9, 0.99),
            eps=1e-9
        )
        
        # Print LR schedule
        print(f"\n  Discriminative Fine-Tuning LR Schedule (decay={decay}):")
        print(f"    Embeddings (general English):  lr = {emb_lr:.2e}")
        for i in range(num_layers):
            dist = num_layers - 1 - i
            lr = base_lr / (decay ** dist)
            print(f"    Decoder Layer {i} ({'bottom' if i == 0 else 'top' if i == num_layers-1 else 'middle'}):{'  ' if i < 10 else ' '}lr = {lr:.2e}")
        print(f"    Final Norm / Output:           lr = {base_lr:.2e}")
        print()
        
        return optimizer
    
    def _estimate_total_steps(self, num_epochs: int) -> int:
        """Estimate total steps accounting for contracting stride"""
        if self.raw_train_data is None:
            return len(self.train_loader) * num_epochs
        
        total = 0
        n = len(self.raw_train_data)
        for epoch in range(1, num_epochs + 1):
            contractions = (epoch - 1) // self.cfg["stride_contract_every"]
            stride = max(self.cfg["stride_min"], self.cfg["stride_initial"] >> contractions)
            n_samples = max(1, (n - self.cfg["max_seq_length"]) // stride)
            n_batches = (n_samples + self.cfg["batch_size"] - 1) // self.cfg["batch_size"]
            total += n_batches
        return total
    
    def _maybe_contract_stride(self, epoch: int):
        """Halve stride every N epochs"""
        if self.raw_train_data is None:
            return
        
        contractions = (epoch - 1) // self.cfg["stride_contract_every"]
        new_stride = max(self.cfg["stride_min"], self.cfg["stride_initial"] >> contractions)
        
        if new_stride != self.current_stride:
            self.current_stride = new_stride
            train_ds = ShakespeareDataset(
                self.raw_train_data, self.cfg["max_seq_length"], stride=new_stride
            )
            self.train_loader = DataLoader(
                train_ds, batch_size=self.cfg["batch_size"], shuffle=True,
                num_workers=0, pin_memory=config.DEVICE.type == 'cuda'
            )
            print(f"  ↳ Stride contracted → {new_stride} ({len(self.train_loader)} batches)")
    
    def _freeze_all_but_top(self):
        """
        Gradual Unfreezing init (ULMFiT, Howard & Ruder 2018).
        
        Freeze everything except:
          - Top decoder layer (layer N-1)
          - Final LayerNorm
          - Output projection (lm_head / tied with embeddings)
        """
        num_layers = self.num_layers
        
        # Freeze ALL parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze top decoder layer
        top_layer_idx = num_layers - 1
        top_layer_name = f'decoder.layers.{top_layer_idx}.'
        for name, param in self.model.named_parameters():
            if top_layer_name in name:
                param.requires_grad = True
        
        # Unfreeze final LayerNorm + output head + any non-layer params
        for name, param in self.model.named_parameters():
            # Final norm, output projection
            if 'decoder.norm' in name or 'output' in name or 'lm_head' in name:
                param.requires_grad = True
            # Positional encoding (if learnable)
            if 'pos' in name.lower() and 'decoder.layers' not in name:
                param.requires_grad = True
        
        self.unfrozen_from_top = 1
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Gradual Unfreezing: frozen {total - trainable:,} / {total:,} params")
        print(f"    Unfrozen: top decoder layer ({top_layer_idx}) + output head")
        print(f"    Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
    
    def _maybe_unfreeze_layer(self, epoch: int):
        """
        Unfreeze next decoder layer every `unfreeze_every` epochs (top to bottom).
        Also unfreezes embeddings when all decoder layers are unfrozen.
        Rebuilds optimizer with discriminative LR for the newly unfrozen parameters.
        """
        if not self.gradual_unfreezing:
            return
        
        # Check if it's time to unfreeze (epoch 1 already has top layer)
        if epoch <= 1:
            return
        
        # Unfreeze at epochs: unfreeze_every+1, 2*unfreeze_every+1, ...
        if (epoch - 1) % self.unfreeze_every != 0:
            return
        
        if self.unfrozen_from_top >= self.num_layers + 1:  # +1 for embeddings
            return  # Everything already unfrozen
        
        if self.unfrozen_from_top < self.num_layers:
            # Unfreeze next decoder layer (from top toward bottom)
            layer_idx = self.num_layers - 1 - self.unfrozen_from_top
            layer_name = f'decoder.layers.{layer_idx}.'
            unfrozen_count = 0
            for name, param in self.model.named_parameters():
                if layer_name in name:
                    param.requires_grad = True
                    unfrozen_count += 1
            self.unfrozen_from_top += 1
            print(f"  ↳ Unfreezing decoder layer {layer_idx} ({unfrozen_count} params) — "
                  f"{self.unfrozen_from_top}/{self.num_layers} layers active")
        else:
            # All decoder layers unfrozen — now unfreeze embeddings
            for name, param in self.model.named_parameters():
                if 'embedding' in name.lower():
                    param.requires_grad = True
            self.unfrozen_from_top += 1
            print(f"  ↳ Unfreezing embeddings — all parameters now trainable")
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"    Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
        
        # Rebuild optimizer with correct param groups (only trainable params)
        if self.cfg.get("discriminative_lr", False):
            self.optimizer = self._create_discriminative_optimizer(self.model, self.cfg)
        else:
            self.optimizer = AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.cfg["learning_rate"],
                weight_decay=self.cfg["weight_decay"],
                betas=(0.9, 0.99), eps=1e-9
            )
        
        # Rebuild scheduler for remaining epochs
        remaining_steps = len(self.train_loader) * (self.cfg["num_epochs"] - epoch + 1)
        warmup = min(50, remaining_steps // 4)  # Short warmup after unfreeze
        warmup_sched = LinearLR(self.optimizer, start_factor=0.3, end_factor=1.0, total_iters=warmup)
        cosine_sched = CosineAnnealingLR(self.optimizer, T_max=max(1, remaining_steps - warmup), eta_min=1e-6)
        self.scheduler = SequentialLR(self.optimizer, [warmup_sched, cosine_sched], milestones=[warmup])
    
    def _maybe_switch_seq_length(self, epoch: int):
        """Switch to shorter sequence length at specified epoch for tighter pattern learning"""
        switch_epoch = self.cfg.get("seq_len_switch_epoch")
        short_len = self.cfg.get("short_seq_length")
        if switch_epoch is None or short_len is None:
            return
        if epoch < switch_epoch:
            return
        if self.cfg["max_seq_length"] == short_len:
            return  # Already switched
        
        print(f"  ↳ Switching seq_length: {self.cfg['max_seq_length']} → {short_len} (epoch {epoch})")
        self.cfg["max_seq_length"] = short_len
        
        # Rebuild all dataloaders with new seq_length
        if self.raw_train_data is not None:
            stride = self.current_stride or self.cfg["stride_initial"]
            train_ds = ShakespeareDataset(self.raw_train_data, short_len, stride=stride)
            pin = config.DEVICE.type == 'cuda'
            self.train_loader = DataLoader(
                train_ds, batch_size=self.cfg["batch_size"], shuffle=True,
                num_workers=0, pin_memory=pin
            )
            print(f"    Train: {len(self.train_loader):,} batches (seq_len={short_len})")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        loss_meter = AverageMeter()
        ppl_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"{self.phase.title()}", leave=False)
        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits = self.model(input_ids)
            B, S, V = logits.shape
            loss = self.criterion(logits.view(-1, V), target_ids.view(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_NORM)
            self.optimizer.step()
            self.scheduler.step()
            
            loss_meter.update(loss.item(), B)
            ppl_meter.update(perplexity(loss.item()), B)
            pbar.set_postfix(loss=f'{loss_meter.avg:.4f}', ppl=f'{ppl_meter.avg:.1f}', lr=f'{get_lr(self.optimizer):.6f}')
        
        return loss_meter.avg, ppl_meter.avg
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate"""
        self.model.eval()
        loss_meter = AverageMeter()
        ppl_meter = AverageMeter()
        
        for input_ids, target_ids in tqdm(self.val_loader, desc="Validating", leave=False):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits = self.model(input_ids)
            B, S, V = logits.shape
            loss = self.criterion(logits.view(-1, V), target_ids.view(-1))
            
            loss_meter.update(loss.item(), B)
            ppl_meter.update(perplexity(loss.item()), B)
        
        return loss_meter.avg, ppl_meter.avg
    
    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict:
        """Full evaluation"""
        loader = loader or self.test_loader
        self.model.eval()
        loss_meter = AverageMeter()
        correct = total = 0
        
        for input_ids, target_ids in tqdm(loader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits = self.model(input_ids)
            B, S, V = logits.shape
            logits_flat = logits.view(-1, V)
            target_flat = target_ids.view(-1)
            
            loss = self.criterion(logits_flat, target_flat)
            loss_meter.update(loss.item(), B)
            
            preds = logits_flat.argmax(dim=-1)
            mask = target_flat != self.vocab.pad_idx
            correct += (preds[mask] == target_flat[mask]).sum().item()
            total += mask.sum().item()
        
        return {
            'loss': loss_meter.avg,
            'perplexity': perplexity(loss_meter.avg),
            'accuracy': correct / total * 100 if total > 0 else 0
        }
    
    def train_loop(self, start_epoch: int = 1) -> Dict:
        """Full training loop with optional resume from start_epoch"""
        num_epochs = self.cfg["num_epochs"]
        save_path = self.cfg["model_path"]
        
        print(f"\n{'='*70}")
        print(f"STARTING {self.phase.upper()} ({num_epochs} epochs, from epoch {start_epoch})")
        print(f"{'='*70}")
        print(f"  Train batches: {len(self.train_loader):,}")
        print(f"  Val batches:   {len(self.val_loader):,}")
        print(f"  Device: {self.device}")
        print(f"  LR: {self.cfg['learning_rate']}")
        print(f"{'='*70}\n")
        
        start = time.time()
        
        for epoch in range(start_epoch, num_epochs + 1):
            t0 = time.time()
            self._maybe_unfreeze_layer(epoch)
            self._maybe_contract_stride(epoch)
            self._maybe_switch_seq_length(epoch)
            
            train_loss, train_ppl = self.train_epoch()
            val_loss, val_ppl = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_ppl'].append(val_ppl)
            self.history['learning_rates'].append(get_lr(self.optimizer))
            
            dt = time.time() - t0
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.1f} | "
                  f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f} | "
                  f"LR: {get_lr(self.optimizer):.6f} | {dt:.1f}s")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, val_loss, save_path)
                print(f"  ↳ Best model saved! (Val Loss: {val_loss:.4f})")
            
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        self.early_stopping.load_best_model(self.model)
        
        total_time = time.time() - start
        print(f"\n{'='*70}")
        print(f"{self.phase.upper()} COMPLETE")
        print(f"{'='*70}")
        print(f"  Time: {total_time/60:.1f} min")
        print(f"  Best epoch: {self.best_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f} (PPL: {perplexity(self.best_val_loss):.1f})")
        
        return self.history


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pretrain(args) -> Tuple[ShakespeareTransformer, BPEVocabulary]:
    """Phase 1: Pre-train on Gutenberg corpus"""
    cfg = PRETRAIN_CONFIG
    
    # Download corpora
    gutenberg_text = download_gutenberg_corpus()
    shakespeare_text = download_shakespeare()
    
    # Prepare data (trains BPE on combined corpus)
    vocab, train_tokens, val_tokens, test_tokens = prepare_pretrain_data(
        gutenberg_text, shakespeare_text, cfg
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = _create_dataloaders(
        train_tokens, val_tokens, test_tokens, cfg
    )
    
    # Create model
    actual_vocab = vocab.bpe.get_vocab_size()
    print(f"\nCreating model (vocab={actual_vocab}, layers={cfg['num_layers']}, "
          f"heads={cfg['num_heads']}, dim={cfg['embed_dim']}, ffn={cfg['ffn_hidden_dim']})")
    
    # Temporarily override config for model creation
    old_vals = {}
    for key in ['NUM_LAYERS', 'NUM_HEADS', 'EMBEDDING_DIM', 'FFN_HIDDEN_DIM', 'DROPOUT', 'ATTENTION_DROPOUT', 'MAX_SEQ_LENGTH']:
        old_vals[key] = getattr(config, key)
    
    config.NUM_LAYERS = cfg["num_layers"]
    config.NUM_HEADS = cfg["num_heads"]
    config.EMBEDDING_DIM = cfg["embed_dim"]
    config.FFN_HIDDEN_DIM = cfg["ffn_hidden_dim"]
    config.DROPOUT = cfg["dropout"]
    config.ATTENTION_DROPOUT = cfg["attention_dropout"]
    config.MAX_SEQ_LENGTH = cfg["max_seq_length"]
    
    model = create_model(vocab_size=actual_vocab, pretrained_embeddings=None, device=config.DEVICE)
    print_model_summary(model, actual_vocab)
    
    # Restore config
    for key, val in old_vals.items():
        setattr(config, key, val)
    
    # Resume from checkpoint if requested
    start_epoch = 1
    if args.resume and Path(cfg["model_path"]).exists():
        from utils import load_checkpoint as _load_ckpt
        start_epoch, best_loss = _load_ckpt(model, None, None, cfg["model_path"])
        # Load only model weights — optimizer/scheduler will be recreated fresh
        start_epoch += 1  # Continue from next epoch
        print(f"  Resuming from epoch {start_epoch} (best loss: {best_loss:.4f})")
    
    # Train
    trainer = PretrainFinetuneTrainer(
        model, vocab, train_loader, val_loader, test_loader,
        cfg, raw_train_data=train_tokens, phase="pretrain"
    )
    
    if args.resume and start_epoch > 1:
        trainer.best_val_loss = best_loss
        trainer.best_epoch = start_epoch - 1
    
    history = trainer.train_loop(start_epoch=start_epoch)
    
    # Evaluate on pre-training test set
    print("\n" + "=" * 70)
    print("PRE-TRAINING EVALUATION (Gutenberg test set)")
    print("=" * 70)
    metrics = trainer.evaluate()
    print(f"  Test Loss: {metrics['loss']:.4f}")
    print(f"  Test PPL:  {metrics['perplexity']:.1f}")
    print(f"  Test Acc:  {metrics['accuracy']:.2f}%")
    
    # Plot
    from train import plot_training_history
    plot_training_history(history, config.LOGS_DIR / "pretrain_history.png")
    
    return model, vocab


def run_finetune(
    args,
    model: Optional[ShakespeareTransformer] = None,
    vocab: Optional[BPEVocabulary] = None
) -> Tuple[ShakespeareTransformer, BPEVocabulary]:
    """Phase 2: Fine-tune pre-trained model on Shakespeare"""
    cfg_pt = PRETRAIN_CONFIG
    cfg_ft = FINETUNE_CONFIG
    
    # Load BPE tokenizer (must be the same one used for pre-training)
    if vocab is None:
        bpe = BPETokenizer(vocab_size=cfg_pt["bpe_vocab_size"])
        bpe.load(cfg_pt["bpe_path"])
        vocab = BPEVocabulary(bpe)
    
    # Load pre-trained model if not provided
    if model is None:
        actual_vocab = vocab.bpe.get_vocab_size()
        
        # Override config for model creation
        old_vals = {}
        for key in ['NUM_LAYERS', 'NUM_HEADS', 'EMBEDDING_DIM', 'FFN_HIDDEN_DIM', 'DROPOUT', 'ATTENTION_DROPOUT', 'MAX_SEQ_LENGTH']:
            old_vals[key] = getattr(config, key)
        
        config.NUM_LAYERS = cfg_pt["num_layers"]
        config.NUM_HEADS = cfg_pt["num_heads"]
        config.EMBEDDING_DIM = cfg_pt["embed_dim"]
        config.FFN_HIDDEN_DIM = cfg_pt["ffn_hidden_dim"]
        config.DROPOUT = cfg_ft["dropout"]          # Use fine-tune dropout
        config.ATTENTION_DROPOUT = cfg_ft["attention_dropout"]
        config.MAX_SEQ_LENGTH = cfg_pt["max_seq_length"]
        
        model = create_model(vocab_size=actual_vocab, pretrained_embeddings=None, device=config.DEVICE)
        
        # Load pre-trained weights
        checkpoint = torch.load(cfg_pt["model_path"], map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pre-trained weights from {cfg_pt['model_path']}")
        print(f"  Pre-training best epoch: {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
        
        for key, val in old_vals.items():
            setattr(config, key, val)
    else:
        # Model already in memory — just update dropout for fine-tuning
        _update_dropout(model, cfg_ft["dropout"], cfg_ft["attention_dropout"])
        print("Using in-memory pre-trained model (dropout updated for fine-tuning)")
    
    # Prepare Shakespeare fine-tuning data
    shakespeare_text = download_shakespeare()
    train_tokens, val_tokens, test_tokens = prepare_finetune_data(
        shakespeare_text, vocab.bpe
    )
    
    # Create dataloaders
    # Merge finetune cfg with necessary architecture params from pretrain cfg
    ft_cfg = {**cfg_ft, "max_seq_length": cfg_pt["max_seq_length"], "batch_size": cfg_pt["batch_size"],
              "bpe_vocab_size": cfg_pt["bpe_vocab_size"]}
    
    train_loader, val_loader, test_loader = _create_dataloaders(
        train_tokens, val_tokens, test_tokens, ft_cfg
    )
    
    print_model_summary(model, vocab.bpe.get_vocab_size())
    
    # Fine-tune
    trainer = PretrainFinetuneTrainer(
        model, vocab, train_loader, val_loader, test_loader,
        ft_cfg, raw_train_data=train_tokens, phase="finetune"
    )
    history = trainer.train_loop()
    
    # Evaluate on Shakespeare test set
    print("\n" + "=" * 70)
    print("FINE-TUNING EVALUATION (Shakespeare test set)")
    print("=" * 70)
    metrics = trainer.evaluate()
    print(f"  Test Loss: {metrics['loss']:.4f}")
    print(f"  Test PPL:  {metrics['perplexity']:.1f}")
    print(f"  Test Acc:  {metrics['accuracy']:.2f}%")
    
    # Compare with previous results
    print(f"\n  Comparison with previous experiments:")
    print(f"    BPE v4 (scratch):       Test PPL 229.70, Accuracy 20.80%")
    print(f"    Pre+Fine v1 (uniform):  Test PPL 191.9,  Accuracy 22.53%")
    print(f"    Pre+Fine v2 (discrim):  Test PPL {metrics['perplexity']:.1f}, Accuracy {metrics['accuracy']:.2f}%")
    
    # Generate samples
    print("\n" + "=" * 70)
    print("SAMPLE GENERATIONS (Fine-tuned)")
    print("=" * 70)
    
    generator = TextGenerator(model, vocab, config.DEVICE)
    seeds = ["to be or not to be", "the king", "love is", "thou art", "what light through yonder"]
    
    for seed in seeds:
        output = generator.generate(seed, max_length=50, temperature=0.8)
        print(f"\n  Seed: '{seed}'")
        print(f"  → {output}")
    
    # Plot
    from train import plot_training_history
    plot_training_history(history, config.LOGS_DIR / "finetune_history.png")
    
    return model, vocab


def _update_dropout(model: nn.Module, dropout: float, attn_dropout: float):
    """Update dropout rates in an existing model for fine-tuning"""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            # Check if it's in an attention module
            module.p = dropout
    # More targeted: set attention dropout specifically
    for name, module in model.named_modules():
        if 'self_attention' in name and isinstance(module, nn.Dropout):
            module.p = attn_dropout


def run_evaluate(args, model=None, vocab=None):
    """Evaluate the fine-tuned model"""
    cfg_pt = PRETRAIN_CONFIG
    cfg_ft = FINETUNE_CONFIG
    
    if vocab is None:
        bpe = BPETokenizer(vocab_size=cfg_pt["bpe_vocab_size"])
        bpe.load(cfg_pt["bpe_path"])
        vocab = BPEVocabulary(bpe)
    
    if model is None:
        model_path = args.checkpoint or cfg_ft["model_path"]
        if not Path(model_path).exists():
            raise FileNotFoundError(f"No checkpoint at {model_path}. Run pre-train + fine-tune first.")
        
        actual_vocab = vocab.bpe.get_vocab_size()
        old_vals = {}
        for key in ['NUM_LAYERS', 'NUM_HEADS', 'EMBEDDING_DIM', 'FFN_HIDDEN_DIM', 'MAX_SEQ_LENGTH']:
            old_vals[key] = getattr(config, key)
        
        config.NUM_LAYERS = cfg_pt["num_layers"]
        config.NUM_HEADS = cfg_pt["num_heads"]
        config.EMBEDDING_DIM = cfg_pt["embed_dim"]
        config.FFN_HIDDEN_DIM = cfg_pt["ffn_hidden_dim"]
        config.MAX_SEQ_LENGTH = cfg_pt["max_seq_length"]
        
        model = create_model(vocab_size=actual_vocab, pretrained_embeddings=None, device=config.DEVICE)
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
        
        for key, val in old_vals.items():
            setattr(config, key, val)
    
    # Prepare test data
    shakespeare_text = download_shakespeare()
    _, _, test_tokens = prepare_finetune_data(shakespeare_text, vocab.bpe)
    
    seq_len = cfg_pt["max_seq_length"]
    test_ds = ShakespeareDataset(test_tokens, seq_len)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0,
                             pin_memory=config.DEVICE.type == 'cuda')
    
    # Evaluate
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, label_smoothing=0.1)
    model.eval()
    loss_m = AverageMeter()
    correct = total = 0
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            logits = model(x)
            B, S, V = logits.shape
            lf = logits.view(-1, V)
            tf = y.view(-1)
            loss = criterion(lf, tf)
            loss_m.update(loss.item(), B)
            preds = lf.argmax(-1)
            mask = tf != vocab.pad_idx
            correct += (preds[mask] == tf[mask]).sum().item()
            total += mask.sum().item()
    
    ppl = perplexity(loss_m.avg)
    acc = correct / total * 100
    print(f"\nShakespeare Test Results:")
    print(f"  Loss:       {loss_m.avg:.4f}")
    print(f"  Perplexity: {ppl:.1f}")
    print(f"  Accuracy:   {acc:.2f}%")


def run_generate(args, model=None, vocab=None):
    """Generate text with the fine-tuned model"""
    cfg_pt = PRETRAIN_CONFIG
    cfg_ft = FINETUNE_CONFIG
    
    if vocab is None:
        bpe = BPETokenizer(vocab_size=cfg_pt["bpe_vocab_size"])
        bpe.load(cfg_pt["bpe_path"])
        vocab = BPEVocabulary(bpe)
    
    if model is None:
        model_path = args.checkpoint or cfg_ft["model_path"]
        if not Path(model_path).exists():
            raise FileNotFoundError(f"No checkpoint at {model_path}")
        
        actual_vocab = vocab.bpe.get_vocab_size()
        old_vals = {}
        for key in ['NUM_LAYERS', 'NUM_HEADS', 'EMBEDDING_DIM', 'FFN_HIDDEN_DIM', 'MAX_SEQ_LENGTH']:
            old_vals[key] = getattr(config, key)
        
        config.NUM_LAYERS = cfg_pt["num_layers"]
        config.NUM_HEADS = cfg_pt["num_heads"]
        config.EMBEDDING_DIM = cfg_pt["embed_dim"]
        config.FFN_HIDDEN_DIM = cfg_pt["ffn_hidden_dim"]
        config.MAX_SEQ_LENGTH = cfg_pt["max_seq_length"]
        
        model = create_model(vocab_size=actual_vocab, pretrained_embeddings=None, device=config.DEVICE)
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        for key, val in old_vals.items():
            setattr(config, key, val)
    
    generator = TextGenerator(model, vocab, config.DEVICE)
    
    print(f"\nPrompt: '{args.prompt}'")
    output = generator.generate(args.prompt, max_length=args.max_length, temperature=args.temperature)
    print(f"\nGenerated:\n{output}")
    
    print("\n" + "-" * 50)
    for temp in [0.5, 0.8, 1.0, 1.2]:
        out = generator.generate(args.prompt, max_length=30, temperature=temp)
        print(f"  Temp {temp}: {out}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train on Gutenberg + Fine-tune on Shakespeare")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'pretrain', 'finetune', 'evaluate', 'generate'],
                        help='all=pretrain+finetune, or run a single phase')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--prompt', type=str, default='to be or not to be', help='Generation prompt')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--seed', type=int, default=config.SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("\n" + "=" * 70)
    print("GUTENBERG PRE-TRAINING + SHAKESPEARE FINE-TUNING PIPELINE")
    print("=" * 70)
    print(f"  Mode: {args.mode}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Seed: {args.seed}")
    print("=" * 70)
    
    model = None
    vocab = None
    
    if args.mode in ('all', 'pretrain'):
        model, vocab = run_pretrain(args)
    
    if args.mode in ('all', 'finetune'):
        model, vocab = run_finetune(args, model, vocab)
    
    if args.mode in ('all', 'evaluate'):
        run_evaluate(args, model, vocab)
    
    if args.mode in ('all', 'generate'):
        run_generate(args, model, vocab)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
