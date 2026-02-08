"""
Main Entry Point for Word-Level Shakespeare Text Generation

Usage:
    python main.py --mode train        # Train the model
    python main.py --mode generate     # Generate text from trained model
    python main.py --mode evaluate     # Evaluate on test set
    python main.py --mode all          # Train, evaluate, and generate
"""

import argparse
import torch
from pathlib import Path

import config
from utils import set_seed, print_model_summary
from data_loader import prepare_data, ShakespeareTokenizer
from model import create_model, TextGenerator
from train import Trainer, plot_training_history
from embeddings import analyze_embedding_space


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Word-Level Shakespeare Text Generation with Transformer"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['train', 'generate', 'evaluate', 'all'],
        help='Mode to run: train, generate, evaluate, or all'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=config.SEED,
        help='Random seed'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.NUM_EPOCHS,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from or use for generation'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default="to be or not to be",
        help='Prompt for text generation'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=config.TEMPERATURE,
        help='Sampling temperature for generation'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=config.MAX_GENERATE_LENGTH,
        help='Maximum length for generation'
    )
    
    return parser.parse_args()


def train(args, vocab, embedding_matrix, anchor_mappings, 
          train_loader, val_loader, test_loader):
    """Train the model"""
    print("\n" + "=" * 70)
    print("TRAINING MODE")
    print("=" * 70)
    
    # Create model
    model = create_model(
        vocab_size=len(vocab),
        pretrained_embeddings=embedding_matrix,  # None for BPE
        device=config.DEVICE
    )
    
    # Print model summary
    print_model_summary(model, len(vocab))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        vocab=vocab,
        anchor_mappings=anchor_mappings,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config.DEVICE
    )
    
    # Train
    history = trainer.train(num_epochs=args.epochs)
    
    # Plot history (separate files for word-level vs BPE)
    plot_name = "training_history_bpe.png" if config.TOKENIZER_TYPE == "bpe" else "training_history.png"
    plot_training_history(history, config.LOGS_DIR / plot_name)
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Perplexity: {test_metrics['perplexity']:.2f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    
    # Generate samples
    print("\n" + "=" * 70)
    print("SAMPLE GENERATIONS")
    print("=" * 70)
    
    seeds = [
        "to be or not to be",
        "the king",
        "love is",
        "thou art",
        "what light through yonder"
    ]
    
    samples = trainer.generate_samples(seeds, max_length=50, temperature=0.8)
    
    for seed, sample in zip(seeds, samples):
        print(f"\nSeed: '{seed}'")
        print(f"Generated: {sample}")
        print("-" * 50)
    
    # Analyze embedding space (word-level only — BPE subwords aren't meaningful words)
    if config.TOKENIZER_TYPE == "word":
        print("\n" + "=" * 70)
        print("EMBEDDING SPACE ANALYSIS")
        print("=" * 70)
        
        embedding_weights = model.get_embedding_weights().detach().cpu()
        analysis = analyze_embedding_space(embedding_weights, vocab)
        
        for word, neighbors in analysis.items():
            neighbor_str = ", ".join([f"{w} ({s:.3f})" for w, s in neighbors])
            print(f"'{word}': {neighbor_str}")
    else:
        print("\n(Embedding space analysis skipped for BPE — subword tokens aren't full words)")
    
    return model, history


def generate(args, vocab, model=None):
    """Generate text using trained model"""
    print("\n" + "=" * 70)
    print("GENERATION MODE")
    print("=" * 70)
    
    if model is None:
        # Load from checkpoint
        checkpoint_path = args.checkpoint or (config.MODELS_DIR / "best_model.pt")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}. "
                "Please train the model first or specify a checkpoint path."
            )
        
        # Create model and load weights
        model = create_model(
            vocab_size=len(vocab),
            pretrained_embeddings=None,  # Will be loaded from checkpoint
            device=config.DEVICE
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
    
    # Create generator
    generator = TextGenerator(model, vocab, config.DEVICE)
    
    # Generate from prompt
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Temperature: {args.temperature}")
    print(f"Max length: {args.max_length}")
    print("-" * 50)
    
    generated = generator.generate(
        seed_text=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=config.TOP_K,
        top_p=config.TOP_P,
        repetition_penalty=config.REPETITION_PENALTY
    )
    
    print(f"\nGenerated text:\n{generated}")
    
    # Generate with different temperatures
    print("\n" + "-" * 50)
    print("Comparing temperatures:")
    print("-" * 50)
    
    for temp in [0.5, 0.8, 1.0, 1.2]:
        output = generator.generate(
            seed_text=args.prompt,
            max_length=30,
            temperature=temp
        )
        print(f"\nTemp {temp}: {output}")
    
    return generated


def evaluate(args, vocab, test_loader, model=None):
    """Evaluate model on test set"""
    print("\n" + "=" * 70)
    print("EVALUATION MODE")
    print("=" * 70)
    
    if model is None:
        checkpoint_path = args.checkpoint or (config.MODELS_DIR / "best_model.pt")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        model = create_model(
            vocab_size=len(vocab),
            pretrained_embeddings=None,
            device=config.DEVICE
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
    
    # Evaluate
    model.eval()
    
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=config.LABEL_SMOOTHING
    )
    
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, target_ids in test_loader:
            input_ids = input_ids.to(config.DEVICE)
            target_ids = target_ids.to(config.DEVICE)
            
            logits = model(input_ids)
            
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            target_flat = target_ids.view(-1)
            
            loss = criterion(logits_flat, target_flat)
            total_loss += loss.item() * batch_size
            
            predictions = logits_flat.argmax(dim=-1)
            mask = target_flat != vocab.pad_idx
            total_correct += (predictions[mask] == target_flat[mask]).sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(test_loader.dataset)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_tokens * 100
    
    print(f"\nTest Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return {'loss': avg_loss, 'perplexity': ppl, 'accuracy': accuracy}


def main():
    """Main entry point"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Print configuration
    print("\n" + "=" * 70)
    tokenizer_label = "WORD-LEVEL" if config.TOKENIZER_TYPE == "word" else "BPE SUBWORD"
    print(f"{tokenizer_label} SHAKESPEARE TEXT GENERATION")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Tokenizer: {config.TOKENIZER_TYPE}")
    print(f"Device: {config.DEVICE}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    
    # Prepare data
    print("\nPreparing data...")
    vocab, embedding_matrix, anchor_mappings, train_loader, val_loader, test_loader = prepare_data()
    
    model = None
    
    # Run based on mode
    if args.mode == 'train' or args.mode == 'all':
        model, history = train(
            args, vocab, embedding_matrix, anchor_mappings,
            train_loader, val_loader, test_loader
        )
    
    if args.mode == 'evaluate' or args.mode == 'all':
        evaluate(args, vocab, test_loader, model)
    
    if args.mode == 'generate' or args.mode == 'all':
        generate(args, vocab, model)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
