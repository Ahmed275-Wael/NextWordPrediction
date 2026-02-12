"""
Training Pipeline for Word-Level Shakespeare Text Generation

This module handles:
1. Training loop with validation
2. Loss computation (including semantic anchor preservation)
3. Learning rate scheduling
4. Checkpointing and logging
5. Evaluation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Tuple, Optional, List
import time
from pathlib import Path
from tqdm import tqdm

import config
from utils import (
    set_seed, count_parameters, save_checkpoint, load_checkpoint,
    EarlyStopping, AverageMeter, get_lr, print_model_summary, perplexity
)
from model import ShakespeareTransformer, TextGenerator
from embeddings import SemanticAnchorLoss
from augmentation import SimpleAugmenter


class Trainer:
    """
    Trainer class for Shakespeare Transformer.
    
    Handles training, validation, and evaluation with:
    - Differential learning rates (lower for embeddings)
    - Semantic anchor preservation loss
    - Learning rate scheduling with warmup
    - Early stopping
    - Checkpoint saving
    """
    
    def __init__(
        self,
        model: ShakespeareTransformer,
        vocab,
        anchor_mappings: Dict[str, str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: torch.device = config.DEVICE,
        use_augmentation: bool = True,
        raw_train_data: Optional[list] = None
    ):
        self.model = model
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Raw training data for contracting stride (BPE only)
        self.raw_train_data = raw_train_data
        self.current_stride = config.STRIDE_INITIAL if raw_train_data is not None else None
        
        # Data augmentation (word-level only — swapping subwords isn't meaningful)
        if config.TOKENIZER_TYPE == "bpe":
            self.augmenter = SimpleAugmenter(vocab, swap_prob=0.0, enabled=False)
            print("Data augmentation disabled (BPE mode — subword swapping not applicable)")
        else:
            self.augmenter = SimpleAugmenter(vocab, swap_prob=0.1, enabled=use_augmentation)
            if use_augmentation:
                print("Data augmentation enabled (random swap)")
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_idx,
            label_smoothing=config.LABEL_SMOOTHING
        )
        
        self.anchor_loss = SemanticAnchorLoss(
            vocab=vocab,
            anchor_mappings=anchor_mappings,
            weight=config.SEMANTIC_PRESERVATION_WEIGHT
        )
        
        # Optimizer with differential learning rates
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.PATIENCE,
            min_delta=config.MIN_DELTA
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_ppl': [],
            'val_ppl': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def _create_optimizer(self) -> AdamW:
        """Create optimizer with differential learning rates"""
        if config.TOKENIZER_TYPE == "bpe":
            # BPE: no pre-trained embeddings, use same LR for everything
            # beta2=0.99 (not 0.95/0.98) because tokens-per-iter is small
            # on our ~1M token dataset — stabilises AdamW variance estimates
            optimizer = AdamW(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
                betas=(0.9, 0.99),
                eps=1e-9
            )
            total_params = sum(1 for p in self.model.parameters())
            print(f"Optimizer created (BPE mode — uniform LR):")
            print(f"  - All params: {total_params}, lr={config.LEARNING_RATE}")
        else:
            # Word-level: separate embedding parameters (lower LR for pre-trained)
            embedding_params = []
            other_params = []
            
            for name, param in self.model.named_parameters():
                if 'embedding' in name.lower():
                    embedding_params.append(param)
                else:
                    other_params.append(param)
            
            # Different learning rates
            param_groups = [
                {'params': embedding_params, 'lr': config.EMBEDDING_LR},
                {'params': other_params, 'lr': config.LEARNING_RATE}
            ]
            
            optimizer = AdamW(
                param_groups,
                weight_decay=config.WEIGHT_DECAY,
                betas=(0.9, 0.99),
                eps=1e-9
            )
            
            print(f"Optimizer created (word-level — differential LR):")
            print(f"  - Embedding params: {len(embedding_params)}, lr={config.EMBEDDING_LR}")
            print(f"  - Other params: {len(other_params)}, lr={config.LEARNING_RATE}")
        
        return optimizer
    
    def _estimate_total_steps(self, num_epochs: int) -> int:
        """
        Estimate total training steps accounting for contracting strides.
        
        Without this, the cosine scheduler thinks training is N × initial_batches steps,
        but stride contractions increase batches per epoch — causing the cosine cycle
        to end too early and restart with high LR (which causes overfitting).
        """
        if self.raw_train_data is None:
            # Word-level: constant batch count
            return len(self.train_loader) * num_epochs
        
        # Simulate the stride schedule to count total batches
        total_steps = 0
        n_tokens = len(self.raw_train_data)
        
        for epoch in range(1, num_epochs + 1):
            contractions = (epoch - 1) // config.STRIDE_CONTRACT_EVERY
            stride = max(config.STRIDE_MIN, config.STRIDE_INITIAL >> contractions)
            n_samples = max(1, (n_tokens - config.MAX_SEQ_LENGTH) // stride)
            n_batches = (n_samples + config.BATCH_SIZE - 1) // config.BATCH_SIZE
            total_steps += n_batches
        
        return total_steps
    
    def _create_scheduler(self, num_epochs: int = config.NUM_EPOCHS):
        """Create learning rate scheduler with warmup, accounting for stride contractions"""
        num_training_steps = self._estimate_total_steps(num_epochs)
        
        print(f"LR scheduler: {num_training_steps:,} total steps "
              f"(warmup: {config.WARMUP_STEPS}, cosine: {num_training_steps - config.WARMUP_STEPS})")
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.WARMUP_STEPS
        )
        
        # Main scheduler
        if config.LR_SCHEDULER == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - config.WARMUP_STEPS,
                eta_min=1e-6
            )
        else:
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=num_training_steps - config.WARMUP_STEPS
            )
        
        # Combine warmup and main scheduler
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.WARMUP_STEPS]
        )
        
        return scheduler
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        loss_meter = AverageMeter()
        ppl_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Apply data augmentation to inputs (not targets)
            input_ids = self.augmenter.augment_batch(input_ids, augment_prob=0.3)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Reshape for loss computation
            # logits: (batch, seq, vocab) -> (batch * seq, vocab)
            # target: (batch, seq) -> (batch * seq)
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            target_flat = target_ids.view(-1)
            
            # Compute main loss
            main_loss = self.criterion(logits_flat, target_flat)
            
            # Compute semantic anchor preservation loss
            anchor_loss = self.anchor_loss(self.model.get_embedding_weights())
            
            # Total loss
            total_loss = main_loss + anchor_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.GRAD_CLIP_NORM
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            loss_meter.update(main_loss.item(), batch_size)
            ppl_meter.update(perplexity(main_loss.item()), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'ppl': f'{ppl_meter.avg:.2f}',
                'lr': f'{get_lr(self.optimizer):.6f}'
            })
        
        return loss_meter.avg, ppl_meter.avg
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        ppl_meter = AverageMeter()
        
        for input_ids, target_ids in tqdm(self.val_loader, desc="Validating", leave=False):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Compute loss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            target_flat = target_ids.view(-1)
            
            loss = self.criterion(logits_flat, target_flat)
            
            # Update metrics
            loss_meter.update(loss.item(), batch_size)
            ppl_meter.update(perplexity(loss.item()), batch_size)
        
        return loss_meter.avg, ppl_meter.avg
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader = None) -> Dict[str, float]:
        """Full evaluation on test set"""
        if loader is None:
            loader = self.test_loader
        
        if loader is None:
            raise ValueError("No test loader provided")
        
        self.model.eval()
        
        loss_meter = AverageMeter()
        correct = 0
        total = 0
        
        for input_ids, target_ids in tqdm(loader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits = self.model(input_ids)
            
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            target_flat = target_ids.view(-1)
            
            # Loss
            loss = self.criterion(logits_flat, target_flat)
            loss_meter.update(loss.item(), batch_size)
            
            # Accuracy (excluding padding)
            predictions = logits_flat.argmax(dim=-1)
            mask = target_flat != self.vocab.pad_idx
            correct += (predictions[mask] == target_flat[mask]).sum().item()
            total += mask.sum().item()
        
        return {
            'loss': loss_meter.avg,
            'perplexity': perplexity(loss_meter.avg),
            'accuracy': correct / total * 100 if total > 0 else 0
        }
    
    def _maybe_contract_stride(self, epoch: int):
        """
        Contracting stride: halve the stride every STRIDE_CONTRACT_EVERY epochs.
        This progressively increases overlap, giving the model more diverse
        context windows as training progresses — like zooming in on the data.
        """
        if self.raw_train_data is None:
            return  # word-level mode — no contracting stride
        
        # Calculate what stride should be for this epoch
        contractions = (epoch - 1) // config.STRIDE_CONTRACT_EVERY
        new_stride = max(config.STRIDE_MIN, config.STRIDE_INITIAL >> contractions)  # >> is integer halving
        
        if new_stride != self.current_stride:
            from data_loader import ShakespeareDataset
            self.current_stride = new_stride
            
            # Rebuild training DataLoader with new stride
            train_dataset = ShakespeareDataset(
                self.raw_train_data, config.MAX_SEQ_LENGTH, stride=new_stride
            )
            self.train_loader = DataLoader(
                train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                num_workers=0, pin_memory=config.DEVICE.type == 'cuda'
            )
            print(f"  ↳ Stride contracted: {new_stride * 2} → {new_stride} "
                  f"({len(self.train_loader)} batches, "
                  f"{100 * (1 - new_stride / config.MAX_SEQ_LENGTH):.0f}% overlap)")
    
    def train(self, num_epochs: int = config.NUM_EPOCHS) -> Dict:
        """Full training loop"""
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        if self.raw_train_data is not None:
            print(f"Contracting stride: {config.STRIDE_INITIAL} → {config.STRIDE_MIN} "
                  f"(halving every {config.STRIDE_CONTRACT_EVERY} epochs)")
        print("=" * 70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Contracting stride: progressively increase overlap
            self._maybe_contract_stride(epoch)
            
            # Train
            train_loss, train_ppl = self.train_epoch()
            
            # Validate
            val_loss, val_ppl = self.validate()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_ppl'].append(val_ppl)
            self.history['learning_rates'].append(get_lr(self.optimizer))
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} | "
                  f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | "
                  f"LR: {get_lr(self.optimizer):.6f} | Time: {epoch_time:.1f}s")
            
            # Save best model (use different filename for BPE vs word-level)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                model_name = "best_model_bpe.pt" if config.TOKENIZER_TYPE == "bpe" else "best_model.pt"
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_loss,
                    config.MODELS_DIR / model_name
                )
                print(f"  ↳ New best model saved! (Val Loss: {val_loss:.4f})")
            
            # Early stopping check
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        
        # Load best model
        self.early_stopping.load_best_model(self.model)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time / 60:.2f} minutes")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation perplexity: {perplexity(self.best_val_loss):.2f}")
        print("=" * 70)
        
        return self.history
    
    def generate_samples(
        self,
        seeds: List[str],
        max_length: int = 50,
        temperature: float = 0.8
    ) -> List[str]:
        """Generate sample texts from seeds"""
        generator = TextGenerator(self.model, self.vocab, self.device)
        
        samples = []
        for seed in seeds:
            generated = generator.generate(
                seed,
                max_length=max_length,
                temperature=temperature
            )
            samples.append(generated)
        
        return samples


def plot_training_history(history: Dict, save_path: Optional[Path] = None):
    """Plot training history"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Perplexity
    axes[1].plot(history['train_ppl'], label='Train')
    axes[1].plot(history['val_ppl'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Training and Validation Perplexity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(history['learning_rates'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


# For testing
if __name__ == "__main__":
    print("Testing Training Pipeline")
    print("=" * 60)
    
    # This would require actual data - just test optimizer creation
    from model import create_model
    
    vocab_size = 8000
    pretrained = torch.randn(vocab_size, config.EMBEDDING_DIM)
    model = create_model(vocab_size, pretrained)
    
    # Count parameters
    total = count_parameters(model)
    print(f"Trainable parameters: {total:,}")
