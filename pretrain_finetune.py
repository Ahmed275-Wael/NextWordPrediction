"""
Pre-train on Project Gutenberg → Fine-tune on Shakespeare

This script implements the pre-training + fine-tuning paradigm:
1. Downloads ~23MB of classic English literature from Project Gutenberg
2. Trains a BPE tokenizer on the combined Gutenberg + Shakespeare corpus
3. Pre-trains a Transformer on the Gutenberg corpus (broad English understanding)
4. Fine-tunes the pre-trained model on Shakespeare (domain specialisation)

Chinchilla Principle Applied:
- Gutenberg corpus ≈ 6M BPE tokens
- Chinchilla-optimal: 1 param per 20 tokens → 300K params
- But 300K is too small for complex language patterns
- Practical compromise: keep our 6.4M param model, but now the ratio is
  6.4M/6M ≈ 1:1 (vs previous 5.8:1 on Shakespeare alone)
- This means the model is slightly over-parameterised but NOT catastrophically so
- The pre-training sees enough data to learn general English patterns
- Fine-tuning on 1.1M Shakespeare tokens then specialises these patterns

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
    "bpe_vocab_size": 8000,        # Larger vocab for bigger corpus
    
    # Architecture — same as Shakespeare model (Chinchilla says we're
    # slightly over-parameterised at 6.4M params / 6M tokens, but this
    # is the practical sweet spot — we need capacity for complex patterns)
    "num_layers": 5,
    "num_heads": 6,
    "embed_dim": 300,
    "ffn_hidden_dim": 1024,
    
    # Training — gentler for larger corpus
    "batch_size": 64,
    "learning_rate": 5e-4,         # Slightly lower than Shakespeare (more data = less aggressive)
    "weight_decay": 0.05,
    "warmup_steps": 1000,          # More warmup for larger dataset
    "num_epochs": 30,              # Longer training — let contracting strides fully explore the data
    "patience": 10,                # More patience — give strides time to help
    "dropout": 0.15,               # Less dropout — more data = less overfitting risk
    "attention_dropout": 0.1,
    "label_smoothing": 0.1,
    "max_seq_length": 128,
    
    # Contracting stride
    "stride_initial": 128,
    "stride_min": 16,              # Go as low as Shakespeare — let the model see dense overlaps
    "stride_contract_every": 5,
    
    # Paths
    "model_path": config.MODELS_DIR / "pretrained_gutenberg_v2.pt",
    "bpe_path": config.DATA_DIR / "bpe_tokenizer_pretrain_8000.json",
}

FINETUNE_CONFIG = {
    # Fine-tuning — lower LR, more regularisation
    "learning_rate": 1e-4,         # 5× lower than pre-training (top layer LR)
    "weight_decay": 0.1,           # Stronger regularisation
    "warmup_steps": 200,
    "num_epochs": 45,              # Longer fine-tuning — let stride schedule run its full course
    "patience": 12,                # More patience — strides keep opening new learning
    "dropout": 0.2,                # More dropout for small fine-tune set
    "attention_dropout": 0.15,
    "label_smoothing": 0.1,
    
    # Discriminative fine-tuning (ULMFiT, Howard & Ruder 2018)
    # Each layer gets LR = top_lr / (decay_factor ^ distance_from_top)
    # Bottom layers (general English) barely change; top layers (style) adapt fast
    "discriminative_lr": True,
    "lr_decay_factor": 2.6,        # Howard & Ruder's recommended decay factor
    
    # Contracting stride — same as BPE v4
    "stride_initial": 128,
    "stride_min": 16,
    "stride_contract_every": 5,
    
    # Paths
    "model_path": config.MODELS_DIR / "finetuned_shakespeare_v2.pt",
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
    
    # Encode Gutenberg corpus (pre-training data)
    print(f"\nEncoding Gutenberg corpus...")
    encoded = bpe.encode(gutenberg_text)
    print(f"  Encoded: {len(encoded):,} tokens (from {len(gutenberg_text):,} chars)")
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
    
    def train_loop(self) -> Dict:
        """Full training loop"""
        num_epochs = self.cfg["num_epochs"]
        save_path = self.cfg["model_path"]
        
        print(f"\n{'='*70}")
        print(f"STARTING {self.phase.upper()} ({num_epochs} epochs)")
        print(f"{'='*70}")
        print(f"  Train batches: {len(self.train_loader):,}")
        print(f"  Val batches:   {len(self.val_loader):,}")
        print(f"  Device: {self.device}")
        print(f"  LR: {self.cfg['learning_rate']}")
        print(f"{'='*70}\n")
        
        start = time.time()
        
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            self._maybe_contract_stride(epoch)
            
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
    
    # Train
    trainer = PretrainFinetuneTrainer(
        model, vocab, train_loader, val_loader, test_loader,
        cfg, raw_train_data=train_tokens, phase="pretrain"
    )
    history = trainer.train_loop()
    
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
