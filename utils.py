"""
Utility functions for Word-Level Shakespeare Text Generation
"""

import torch
import numpy as np
import random
import os
from pathlib import Path

import config


def set_seed(seed: int = config.SEED):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: torch.nn.Module) -> int:
    """Count all parameters in a model (trainable and frozen)"""
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path: Path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, scheduler, path: Path):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = config.PATIENCE, min_delta: float = config.MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
    
    def load_best_model(self, model):
        """Load the best model state"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print("Loaded best model state")


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def print_model_summary(model, vocab_size: int):
    """Print a summary of the model architecture"""
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"Vocabulary Size: {vocab_size:,}")
    print(f"Embedding Dimension: {config.EMBEDDING_DIM}")
    print(f"Transformer Layers: {config.NUM_LAYERS}")
    print(f"Attention Heads: {config.NUM_HEADS}")
    print(f"FFN Hidden Dimension: {config.FFN_HIDDEN_DIM}")
    print(f"Dropout: {config.DROPOUT}")
    print("-" * 70)
    print(f"Total Parameters: {count_all_parameters(model):,}")
    print(f"Trainable Parameters: {count_parameters(model):,}")
    print("=" * 70 + "\n")


def perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss"""
    return np.exp(loss)
