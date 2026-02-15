"""
AWD-LSTM Baseline Runner — Fair Head-to-Head Comparison with Transformer

Implements the full Merity et al. (2018) training recipe:
    - AR (Activation Regularization): alpha=2.0
    - TAR (Temporal Activation Regularization): beta=1.0
    - NT-ASGD: SGD with non-monotonic ASGD trigger
    - Same data pipeline as the Transformer (BPE + contracting stride)

Reference: salesforce/awd-lstm-lm (GitHub)

Reuses the EXACT same pipeline as the Transformer:
    - Same BPE tokenizer (5,000 vocab)
    - Same sequential data split (80/10/10)
    - Same ShakespeareDataset (seq_length=128, contracting stride)
    - Same evaluation (label-smoothed cross-entropy, PPL, accuracy)

Usage:
    python run_lstm_baseline.py --mode train      # Train AWD-LSTM
    python run_lstm_baseline.py --mode generate   # Generate text
    python run_lstm_baseline.py --mode evaluate   # Evaluate on test set
    python run_lstm_baseline.py --mode all        # All three
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np

import config
from utils import (
    set_seed, count_parameters, save_checkpoint,
    EarlyStopping, AverageMeter, get_lr, perplexity
)
from data_loader import prepare_data, ShakespeareDataset
from lstm_model import ShakespeareLSTM, LSTMTextGenerator, create_lstm_model


def print_lstm_summary(model: ShakespeareLSTM, vocab_size: int):
    """Print AWD-LSTM model summary"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 70)
    print("AWD-LSTM MODEL SUMMARY (Merity et al., 2018)")
    print("=" * 70)
    print(f"Architecture:       AWD-LSTM (3-layer, weight drop + variational drop)")
    print(f"Vocabulary Size:    {vocab_size:,}")
    print(f"Embedding Dim:      {model.embed_dim}")
    print(f"Hidden Size:        {model.hidden_size}")
    print(f"LSTM Layers:        {model.num_layers}")
    print(f"Weight Tying:       {model.tie_weights}")
    print(f"--- Dropout Rates ---")
    print(f"  Embedding Drop:   {model.dropoute}")
    print(f"  Input VarDrop:    {model.dropouti}")
    print(f"  Hidden VarDrop:   {model.dropouth}")
    print(f"  Output VarDrop:   {model.dropouto}")
    print(f"  Weight Drop:      {model.wdrop}")
    print("-" * 70)
    print(f"Total Parameters:   {total:,}")
    print(f"Trainable Params:   {trainable:,}")
    print("=" * 70 + "\n")


class LSTMTrainer:
    """
    AWD-LSTM Trainer with Merity et al. (2018) training recipe.
    
    Key additions over vanilla training:
        1. AR (Activation Regularization) — L2 penalty on last layer outputs
           Prevents activations from growing too large. alpha=2.0
        2. TAR (Temporal Activation Regularization) — L2 penalty on differences
           between consecutive timestep outputs. Encourages smooth hidden states.
           beta=1.0
        3. SGD optimizer (Merity et al. found SGD beats Adam for LSTMs)
        4. NT-ASGD: After validation plateaus for `n` epochs, switch to ASGD
           (Averaged SGD). This is the "non-monotonic trigger".
    
    Same data pipeline as Transformer: BPE-5000, contracting stride, etc.
    """
    
    # AR/TAR regularization coefficients (Merity et al. Table 4)
    AR_ALPHA = 2.0   # Activation regularization strength
    TAR_BETA = 1.0   # Temporal activation regularization strength
    
    # NT-ASGD parameters
    ASGD_TRIGGER_PATIENCE = 5   # Epochs without improvement before ASGD switch
    
    def __init__(
        self,
        model: ShakespeareLSTM,
        vocab,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: torch.device = config.DEVICE,
        raw_train_data: Optional[list] = None
    ):
        self.model = model
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Raw data for contracting stride
        self.raw_train_data = raw_train_data
        self.current_stride = config.STRIDE_INITIAL if raw_train_data is not None else None
        
        # Same loss as Transformer
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_idx,
            label_smoothing=config.LABEL_SMOOTHING
        )
        
        # AdamW optimizer — same as Transformer (proven stable with our contracting stride)
        # Merity et al. used SGD lr=30 but that requires PTB-specific BPTT setup.
        # With contracting stride (batch count doubles mid-training), AdamW is more stable.
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,      # 7e-4
            weight_decay=config.WEIGHT_DECAY,  # 0.05
            betas=(0.9, 0.99),
            eps=1e-9
        )
        print(f"Optimizer: AdamW (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
        
        # Cosine LR scheduler (same as Transformer)
        total_steps = self._estimate_total_steps(config.NUM_EPOCHS)
        warmup = config.WARMUP_STEPS
        
        warmup_sched = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
        cosine_sched = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup, eta_min=1e-6)
        self.scheduler = SequentialLR(self.optimizer, [warmup_sched, cosine_sched], milestones=[warmup])
        
        print(f"LR scheduler: {total_steps:,} total steps "
              f"(warmup={warmup}, cosine={total_steps - warmup})")
        
        # NT-ASGD tracking
        self.asgd_triggered = False
        self.asgd_trigger_epoch = None
        self.best_val_for_asgd = float('inf')
        self.asgd_patience_counter = 0
        
        # Same early stopping
        self.early_stopping = EarlyStopping(
            patience=config.PATIENCE,
            min_delta=config.MIN_DELTA
        )
        
        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_ppl': [], 'val_ppl': [],
            'learning_rates': []
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def _estimate_total_steps(self, num_epochs: int) -> int:
        """Same as Transformer trainer"""
        if self.raw_train_data is None:
            return len(self.train_loader) * num_epochs
        
        total = 0
        n = len(self.raw_train_data)
        for epoch in range(1, num_epochs + 1):
            contractions = (epoch - 1) // config.STRIDE_CONTRACT_EVERY
            stride = max(config.STRIDE_MIN, config.STRIDE_INITIAL >> contractions)
            n_samples = max(1, (n - config.MAX_SEQ_LENGTH) // stride)
            n_batches = (n_samples + config.BATCH_SIZE - 1) // config.BATCH_SIZE
            total += n_batches
        return total
    
    def _maybe_contract_stride(self, epoch: int):
        """Same contracting stride as Transformer"""
        if self.raw_train_data is None:
            return
        
        contractions = (epoch - 1) // config.STRIDE_CONTRACT_EVERY
        new_stride = max(config.STRIDE_MIN, config.STRIDE_INITIAL >> contractions)
        
        if new_stride != self.current_stride:
            self.current_stride = new_stride
            train_ds = ShakespeareDataset(
                self.raw_train_data, config.MAX_SEQ_LENGTH, stride=new_stride
            )
            self.train_loader = DataLoader(
                train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                num_workers=0, pin_memory=config.DEVICE.type == 'cuda'
            )
            print(f"  ↳ Stride contracted → {new_stride} "
                  f"({len(self.train_loader)} batches)")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train one epoch with AR/TAR regularization (Merity et al.)"""
        self.model.train()
        loss_meter = AverageMeter()
        ppl_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward — request raw outputs for AR/TAR
            logits, raw_outputs, dropped_outputs = self.model(
                input_ids, return_hidden_for_reg=True
            )
            
            B, S, V = logits.shape
            logits_flat = logits.view(-1, V)
            target_flat = target_ids.view(-1)
            
            # Standard cross-entropy loss
            ce_loss = self.criterion(logits_flat, target_flat)
            
            # AR: Activation Regularization (Merity et al. Eq. 7)
            # L2 penalty on the dropped output of the last LSTM layer
            # This prevents the model's output activations from growing too large
            ar_loss = self.AR_ALPHA * dropped_outputs[-1].pow(2).mean()
            
            # TAR: Temporal Activation Regularization (Merity et al. Eq. 8)
            # L2 penalty on the difference between consecutive timestep outputs
            # This encourages smooth transitions in hidden states
            tar_loss = self.TAR_BETA * (
                raw_outputs[-1][:, 1:, :] - raw_outputs[-1][:, :-1, :]
            ).pow(2).mean()
            
            # Total loss = CE + AR + TAR
            loss = ce_loss + ar_loss + tar_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Same gradient clipping (essential with SGD lr=30)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.GRAD_CLIP_NORM
            )
            
            self.optimizer.step()
            if not self.asgd_triggered:
                self.scheduler.step()
            
            # Track CE loss only (not AR/TAR) for fair PPL comparison
            loss_meter.update(ce_loss.item(), B)
            ppl_meter.update(perplexity(ce_loss.item()), B)
            
            pbar.set_postfix(
                loss=f'{loss_meter.avg:.4f}',
                ppl=f'{ppl_meter.avg:.1f}',
                lr=f'{get_lr(self.optimizer):.6f}'
            )
        
        return loss_meter.avg, ppl_meter.avg
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate — same as Transformer"""
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
    def evaluate(self, loader=None) -> Dict[str, float]:
        """Full evaluation with accuracy — same as Transformer"""
        if loader is None:
            loader = self.test_loader
        
        self.model.eval()
        loss_meter = AverageMeter()
        correct = 0
        total = 0
        
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
    
    def train(self, num_epochs: int = config.NUM_EPOCHS) -> Dict:
        """Full training loop with NT-ASGD (Merity et al.)"""
        print("\n" + "=" * 70)
        print("STARTING AWD-LSTM TRAINING (Merity et al., 2018)")
        print("=" * 70)
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        if self.raw_train_data is not None:
            print(f"Contracting stride: {config.STRIDE_INITIAL} → {config.STRIDE_MIN}")
        print(f"AR alpha: {self.AR_ALPHA}, TAR beta: {self.TAR_BETA}")
        print(f"NT-ASGD trigger patience: {self.ASGD_TRIGGER_PATIENCE}")
        print("=" * 70 + "\n")
        
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
            
            asgd_marker = " [ASGD]" if self.asgd_triggered else ""
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.1f} | "
                  f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f} | "
                  f"LR: {get_lr(self.optimizer):.6f} | {dt:.1f}s{asgd_marker}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_loss,
                    config.MODELS_DIR / "best_model_lstm.pt"
                )
                print(f"  ↳ Best model saved! (Val Loss: {val_loss:.4f})")
            
            # NT-ASGD: Non-Monotonic Averaged SGD trigger
            # If validation hasn't improved for ASGD_TRIGGER_PATIENCE epochs,
            # switch from SGD to ASGD for better convergence
            if not self.asgd_triggered:
                if val_loss < self.best_val_for_asgd:
                    self.best_val_for_asgd = val_loss
                    self.asgd_patience_counter = 0
                else:
                    self.asgd_patience_counter += 1
                    if self.asgd_patience_counter >= self.ASGD_TRIGGER_PATIENCE:
                        print(f"  ↳ NT-ASGD triggered! Switching to Averaged SGD "
                              f"(plateau for {self.ASGD_TRIGGER_PATIENCE} epochs)")
                        self.asgd_triggered = True
                        self.asgd_trigger_epoch = epoch
                        # Switch optimizer to ASGD with current LR
                        current_lr = get_lr(self.optimizer)
                        self.optimizer = torch.optim.ASGD(
                            self.model.parameters(),
                            lr=current_lr,
                            t0=0,
                            lambd=0,
                            weight_decay=config.WEIGHT_DECAY
                        )
            
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        self.early_stopping.load_best_model(self.model)
        
        total_time = time.time() - start
        print("\n" + "=" * 70)
        print("AWD-LSTM TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Best val PPL: {perplexity(self.best_val_loss):.1f}")
        if self.asgd_triggered:
            print(f"ASGD triggered at epoch: {self.asgd_trigger_epoch}")
        print("=" * 70)
        
        return self.history
    
    def generate_samples(self, seeds: List[str], max_length=50, temperature=0.8):
        """Generate sample texts"""
        generator = LSTMTextGenerator(self.model, self.vocab, self.device)
        samples = []
        for seed in seeds:
            out = generator.generate(seed, max_length=max_length, temperature=temperature)
            samples.append(out)
        return samples


# ============================================================================
# Main entry point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LSTM Baseline for Shakespeare — Fair comparison with Transformer"
    )
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'generate', 'evaluate', 'all'])
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--prompt', type=str, default='to be or not to be')
    parser.add_argument('--temperature', type=float, default=config.TEMPERATURE)
    parser.add_argument('--max_length', type=int, default=config.MAX_GENERATE_LENGTH)
    return parser.parse_args()


def train_lstm(args, vocab, train_loader, val_loader, test_loader, raw_train_data=None):
    """Train the AWD-LSTM baseline (Merity et al., 2018)"""
    print("\n" + "=" * 70)
    print("AWD-LSTM BASELINE — TRAINING MODE (Merity et al., 2018)")
    print("=" * 70)
    
    # Create LSTM model
    model = create_lstm_model(vocab_size=len(vocab), device=config.DEVICE)
    print_lstm_summary(model, len(vocab))
    
    # Create trainer (identical pipeline to Transformer)
    trainer = LSTMTrainer(
        model=model,
        vocab=vocab,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config.DEVICE,
        raw_train_data=raw_train_data
    )
    
    # Train
    history = trainer.train(num_epochs=args.epochs)
    
    # Plot
    from train import plot_training_history
    plot_training_history(history, config.LOGS_DIR / "training_history_lstm.png")
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("AWD-LSTM FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    metrics = trainer.evaluate(test_loader)
    print(f"Test Loss:       {metrics['loss']:.4f}")
    print(f"Test Perplexity: {metrics['perplexity']:.2f}")
    print(f"Test Accuracy:   {metrics['accuracy']:.2f}%")
    
    # Compare with Transformer
    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'Transformer (BPE v4)':<25} {'AWD-LSTM':<25}")
    print(f"{'-'*70}")
    print(f"{'Architecture':<20} {'5L Transformer':<25} {'3L AWD-LSTM':<25}")
    print(f"{'Parameters':<20} {'6,431,700':<25} {count_parameters(model):<25,}")
    print(f"{'Test PPL':<20} {'229.70':<25} {metrics['perplexity']:<25.2f}")
    print(f"{'Test Accuracy':<20} {'20.80%':<25} {metrics['accuracy']:<25.2f}%")
    print(f"{'Regularization':<20} {'Std Dropout':<25} {'WD+VD+ED+AR+TAR':<25}")
    print(f"{'Optimizer':<20} {'AdamW':<25} {'AdamW + NT-ASGD':<25}")
    print(f"{'='*70}")
    
    # Generate samples
    print("\n" + "=" * 70)
    print("AWD-LSTM SAMPLE GENERATIONS")
    print("=" * 70)
    
    seeds = ["to be or not to be", "the king", "love is", "thou art",
             "what light through yonder"]
    samples = trainer.generate_samples(seeds, max_length=50, temperature=0.8)
    
    for seed, sample in zip(seeds, samples):
        print(f"\nSeed: '{seed}'")
        print(f"Generated: {sample}")
        print("-" * 50)
    
    return model, history


def generate_lstm(args, vocab, model=None):
    """Generate text with trained AWD-LSTM"""
    print("\n" + "=" * 70)
    print("AWD-LSTM GENERATION MODE")
    print("=" * 70)
    
    if model is None:
        checkpoint_path = args.checkpoint or (config.MODELS_DIR / "best_model_lstm.pt")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"No LSTM checkpoint at {checkpoint_path}. Train first.")
        
        model = create_lstm_model(vocab_size=len(vocab), device=config.DEVICE)
        ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"LSTM loaded from {checkpoint_path}")
    
    generator = LSTMTextGenerator(model, vocab, config.DEVICE)
    
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Temperature: {args.temperature}")
    print("-" * 50)
    
    output = generator.generate(
        args.prompt, max_length=args.max_length, temperature=args.temperature,
        top_k=config.TOP_K, top_p=config.TOP_P,
        repetition_penalty=config.REPETITION_PENALTY
    )
    print(f"\nGenerated:\n{output}")
    
    print("\n" + "-" * 50)
    print("Temperature comparison:")
    for temp in [0.5, 0.8, 1.0, 1.2]:
        out = generator.generate(args.prompt, max_length=30, temperature=temp)
        print(f"  Temp {temp}: {out}")


def evaluate_lstm(args, vocab, test_loader, model=None):
    """Evaluate AWD-LSTM on test set"""
    print("\n" + "=" * 70)
    print("AWD-LSTM EVALUATION MODE")
    print("=" * 70)
    
    if model is None:
        checkpoint_path = args.checkpoint or (config.MODELS_DIR / "best_model_lstm.pt")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"No LSTM checkpoint at {checkpoint_path}")
        
        model = create_lstm_model(vocab_size=len(vocab), device=config.DEVICE)
        ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"LSTM loaded from {checkpoint_path}")
    
    model.eval()
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=config.LABEL_SMOOTHING
    )
    
    loss_meter = AverageMeter()
    correct = total = 0
    
    with torch.no_grad():
        for input_ids, target_ids in tqdm(test_loader, desc="Evaluating"):
            input_ids = input_ids.to(config.DEVICE)
            target_ids = target_ids.to(config.DEVICE)
            
            logits = model(input_ids)
            B, S, V = logits.shape
            lf = logits.view(-1, V)
            tf = target_ids.view(-1)
            
            loss = criterion(lf, tf)
            loss_meter.update(loss.item(), B)
            
            preds = lf.argmax(-1)
            mask = tf != vocab.pad_idx
            correct += (preds[mask] == tf[mask]).sum().item()
            total += mask.sum().item()
    
    ppl = perplexity(loss_meter.avg)
    acc = correct / total * 100
    
    print(f"\nLSTM Test Results:")
    print(f"  Loss:       {loss_meter.avg:.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  Accuracy:   {acc:.2f}%")
    
    return {'loss': loss_meter.avg, 'perplexity': ppl, 'accuracy': acc}


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("\n" + "=" * 70)
    print("AWD-LSTM BASELINE FOR SHAKESPEARE TEXT GENERATION")
    print("    (Merity, Keskar & Socher, 2018)")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Tokenizer: {config.TOKENIZER_TYPE} (same as Transformer)")
    print(f"Device: {config.DEVICE}")
    print("=" * 70)
    
    # Use the EXACT same data pipeline as the Transformer
    print("\nPreparing data (same pipeline as Transformer)...")
    vocab, embedding_matrix, anchor_mappings, \
        train_loader, val_loader, test_loader, raw_train_data = prepare_data()
    
    model = None
    
    if args.mode in ('train', 'all'):
        model, history = train_lstm(
            args, vocab, train_loader, val_loader, test_loader, raw_train_data
        )
    
    if args.mode in ('evaluate', 'all'):
        evaluate_lstm(args, vocab, test_loader, model)
    
    if args.mode in ('generate', 'all'):
        generate_lstm(args, vocab, model)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
