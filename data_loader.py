"""
Data Loading and Preprocessing for Word-Level Shakespeare Text Generation

This module handles:
1. Loading Shakespeare text
2. Tokenization (word-level)
3. Vocabulary building
4. FastText embedding alignment
5. Dataset creation for PyTorch
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from pathlib import Path
import re
import urllib.request
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import config


class ShakespeareTokenizer:
    """
    Word-level tokenizer for Shakespeare text.
    Handles archaic contractions and preserves important punctuation.
    """
    
    def __init__(self):
        # Regex pattern for tokenization
        # Handles contractions like 'tis, 'twas, o'er, etc.
        self.token_pattern = re.compile(
            r"'t\w+|"           # 'tis, 'twas, 'twill
            r"o'er|e'er|ne'er|e'en|"  # Common contractions
            r"\w+'\w+|"        # Other contractions (don't, I'll)
            r"\w+|"            # Regular words
            r"[.,!?;:'\"\-\(\)]"  # Punctuation (kept separate)
        , re.IGNORECASE)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase for consistency
        text = text.lower()
        # Find all tokens
        tokens = self.token_pattern.findall(text)
        return tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text"""
        text = ' '.join(tokens)
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:\'\"\)])', r'\1', text)
        text = re.sub(r'([\(\"])\s+', r'\1', text)
        return text


class Vocabulary:
    """
    Vocabulary class for mapping words to indices and vice versa.
    Includes special tokens and handles unknown words.
    """
    
    def __init__(self, min_frequency: int = config.MIN_WORD_FREQUENCY,
                 max_vocab_size: int = config.MAX_VOCAB_SIZE):
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size
        
        # Initialize with special tokens
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        
        # Add special tokens
        for i, token in enumerate(config.SPECIAL_TOKENS):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token
        
        self.pad_idx = self.word_to_idx[config.PAD_TOKEN]
        self.unk_idx = self.word_to_idx[config.UNK_TOKEN]
        self.bos_idx = self.word_to_idx[config.BOS_TOKEN]
        self.eos_idx = self.word_to_idx[config.EOS_TOKEN]
    
    def build(self, tokens: List[str]):
        """Build vocabulary from list of tokens"""
        # Count word frequencies
        self.word_counts = Counter(tokens)
        
        # Filter by frequency and limit size
        filtered_words = [
            word for word, count in self.word_counts.most_common()
            if count >= self.min_frequency
        ]
        
        # Limit vocabulary size (accounting for special tokens)
        max_words = self.max_vocab_size - len(config.SPECIAL_TOKENS)
        filtered_words = filtered_words[:max_words]
        
        # Add to vocabulary
        for word in filtered_words:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        print(f"Vocabulary built: {len(self.word_to_idx):,} words")
        print(f"  - Special tokens: {len(config.SPECIAL_TOKENS)}")
        print(f"  - Regular words: {len(self.word_to_idx) - len(config.SPECIAL_TOKENS):,}")
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices"""
        return [self.word_to_idx.get(token, self.unk_idx) for token in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens"""
        return [self.idx_to_word.get(idx, config.UNK_TOKEN) for idx in indices]
    
    def __len__(self) -> int:
        return len(self.word_to_idx)
    
    def get_unk_words(self, tokens: List[str]) -> List[str]:
        """Get list of words that would be mapped to UNK"""
        return [t for t in set(tokens) if t not in self.word_to_idx]
    
    def get_archaic_words(self) -> List[str]:
        """Get archaic Shakespeare words in vocabulary"""
        archaic = []
        for word in self.word_to_idx:
            if word in config.SEMANTIC_ANCHORS:
                archaic.append(word)
        return archaic


class ShakespeareDataset(Dataset):
    """
    PyTorch Dataset for Shakespeare text.
    Creates (input_sequence, target_sequence) pairs for next-word prediction.
    
    Uses a stride parameter to control overlap between sequences.
    stride=1 means maximum overlap (every starting position).
    stride=seq_length means no overlap (non-overlapping chunks).
    stride=seq_length//2 is a good balance (50% overlap).
    """
    
    def __init__(self, encoded_text: List[int], seq_length: int = config.MAX_SEQ_LENGTH, stride: int = None):
        self.encoded_text = encoded_text
        self.seq_length = seq_length
        # Default stride: half the sequence length (50% overlap)
        self.stride = stride if stride is not None else max(1, seq_length // 2)
        
        # Create valid starting indices with stride
        self.start_indices = list(range(0, len(encoded_text) - seq_length, self.stride))
        self.num_samples = len(self.start_indices)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.start_indices[idx]
        # Input sequence
        input_seq = self.encoded_text[start:start + self.seq_length]
        # Target sequence (shifted by 1)
        target_seq = self.encoded_text[start + 1:start + self.seq_length + 1]
        
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )


def download_shakespeare() -> str:
    """Download Shakespeare text from URL"""
    # Use different filename based on dataset size
    filename = f"shakespeare_{config.DATASET_SIZE}.txt"
    data_path = config.DATA_DIR / filename
    
    if data_path.exists():
        print(f"Loading Shakespeare text from {data_path}")
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    else:
        print(f"Downloading Shakespeare text ({config.DATASET_SIZE})...")
        print(f"URL: {config.SHAKESPEARE_URL}")
        urllib.request.urlretrieve(config.SHAKESPEARE_URL, data_path)
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        print(f"Saved to {data_path}")
    
    # Clean Gutenberg headers/footers if present
    if config.DATASET_SIZE == "full":
        text = clean_gutenberg_text(text)
    
    print(f"Text length: {len(text):,} characters")
    return text


def clean_gutenberg_text(text: str) -> str:
    """
    Remove Project Gutenberg headers, footers, and licensing text.
    Keep only the actual Shakespeare content.
    """
    # Find start of actual content (after Gutenberg header)
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT",
        "THE SONNETS",  # First work in complete works
    ]
    
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Move past the marker line
            newline_idx = text.find('\n', idx)
            if newline_idx != -1:
                start_idx = max(start_idx, newline_idx + 1)
    
    # Find end of actual content (before Gutenberg footer)
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "End of Project Gutenberg",
        "End of the Project Gutenberg",
    ]
    
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = min(end_idx, idx)
    
    cleaned = text[start_idx:end_idx]
    
    # Remove table of contents and other non-play content
    # The actual plays start after "THE SONNETS" section
    
    print(f"Cleaned Gutenberg text: {len(text):,} → {len(cleaned):,} characters")
    return cleaned


def load_fasttext_model():
    """
    Load pre-trained word vectors using gensim.
    Downloads if not available locally.
    Uses gensim's built-in model downloader (no C++ compilation needed).
    """
    import gensim.downloader as api
    
    print("Loading pre-trained word vectors...")
    print("(First run downloads ~1GB, subsequent runs load from gensim cache)")
    
    # Try to load FastText subword model first (best for OOV handling)
    # Falls back to GloVe if unavailable
    try:
        # FastText with subword information - best for Shakespeare's archaic words
        model = api.load('fasttext-wiki-news-subwords-300')
        print("Loaded: FastText Wiki News Subwords (300d)")
    except Exception as e:
        print(f"FastText unavailable: {e}")
        print("Falling back to GloVe...")
        try:
            model = api.load('glove-wiki-gigaword-300')
            print("Loaded: GloVe Wiki Gigaword (300d)")
        except Exception as e2:
            print(f"GloVe unavailable: {e2}")
            print("Falling back to GloVe 100d (smaller)...")
            model = api.load('glove-wiki-gigaword-100')
            print("Loaded: GloVe Wiki Gigaword (100d)")
    
    return model


def save_embeddings_cache(vocab: Vocabulary, embedding_matrix: torch.Tensor, 
                          anchor_mappings: Dict[str, str], cache_path: Path):
    """Save aligned embeddings to disk for fast loading"""
    cache_data = {
        'word_to_idx': vocab.word_to_idx,
        'idx_to_word': vocab.idx_to_word,
        'embedding_matrix': embedding_matrix,
        'anchor_mappings': anchor_mappings,
        'vocab_size': len(vocab)
    }
    torch.save(cache_data, cache_path)
    print(f"Embeddings cached to {cache_path}")


def load_embeddings_cache(cache_path: Path) -> Optional[Tuple[Vocabulary, torch.Tensor, Dict[str, str]]]:
    """Load cached embeddings if available"""
    if not cache_path.exists():
        return None
    
    print(f"Loading cached embeddings from {cache_path}")
    cache_data = torch.load(cache_path, weights_only=False)
    
    # Reconstruct vocabulary
    vocab = Vocabulary()
    vocab.word_to_idx = cache_data['word_to_idx']
    vocab.idx_to_word = cache_data['idx_to_word']
    
    return vocab, cache_data['embedding_matrix'], cache_data['anchor_mappings']


def align_embeddings_with_fasttext(
    vocab: Vocabulary,
    ft_model,
    use_semantic_anchors: bool = True
) -> Tuple[torch.Tensor, Dict[str, str]]:
    """
    Create embedding matrix aligned with pre-trained word vectors.
    
    Strategy:
    1. For words in model: use pre-trained embedding
    2. For archaic words with anchors: initialize near modern equivalent
    3. For other OOV: random initialization with appropriate scale
    
    Args:
        vocab: Vocabulary object
        ft_model: Gensim KeyedVectors model
        use_semantic_anchors: Whether to use semantic anchor initialization
    
    Returns:
        embedding_matrix: Tensor of shape (vocab_size, embedding_dim)
        anchor_mappings: Dict mapping archaic words to their anchors used
    """
    vocab_size = len(vocab)
    embedding_dim = ft_model.vector_size
    
    # Initialize embedding matrix
    embedding_matrix = torch.zeros(vocab_size, embedding_dim)
    
    # Track statistics
    stats = {
        'in_model': 0,
        'in_model_lower': 0,
        'anchor_initialized': 0,
        'random_initialized': 0
    }
    anchor_mappings = {}
    
    print("\nAligning vocabulary with pre-trained embeddings...")
    
    for word, idx in tqdm(vocab.word_to_idx.items(), desc="Processing words"):
        # Skip special tokens (will remain zero-initialized or learned)
        if word in config.SPECIAL_TOKENS:
            # Small random initialization for special tokens
            embedding_matrix[idx] = torch.randn(embedding_dim) * 0.01
            continue
        
        # Strategy 1: Direct lookup (exact match)
        if word in ft_model:
            embedding_matrix[idx] = torch.from_numpy(ft_model[word].copy())
            stats['in_model'] += 1
        
        # Strategy 1b: Try lowercase
        elif word.lower() in ft_model:
            embedding_matrix[idx] = torch.from_numpy(ft_model[word.lower()].copy())
            stats['in_model_lower'] += 1
        
        # Strategy 2: Semantic anchor initialization for archaic words
        elif use_semantic_anchors and word in config.SEMANTIC_ANCHORS:
            anchor = config.SEMANTIC_ANCHORS[word]
            # Handle multi-word anchors (e.g., "it_is" -> average of "it" and "is")
            if '_' in anchor:
                parts = anchor.split('_')
                valid_parts = [p for p in parts if p in ft_model]
                if valid_parts:
                    anchor_emb = np.mean([ft_model[p] for p in valid_parts], axis=0)
                else:
                    anchor_emb = np.random.randn(embedding_dim) * 0.1
            elif anchor in ft_model:
                anchor_emb = ft_model[anchor]
            else:
                anchor_emb = np.random.randn(embedding_dim) * 0.1
            
            # Add small noise to differentiate from anchor
            noise = np.random.randn(embedding_dim) * 0.05
            embedding_matrix[idx] = torch.from_numpy(anchor_emb.copy() + noise)
            anchor_mappings[word] = anchor
            stats['anchor_initialized'] += 1
        
        # Strategy 3: Random initialization for unknown words
        else:
            embedding_matrix[idx] = torch.randn(embedding_dim) * 0.1
            stats['random_initialized'] += 1
    
    # Print statistics
    print("\nEmbedding Alignment Statistics:")
    print(f"  - Direct lookup:          {stats['in_model']:,}")
    print(f"  - Lowercase lookup:       {stats['in_model_lower']:,}")
    print(f"  - Semantic anchor init:   {stats['anchor_initialized']:,}")
    print(f"  - Random initialized:     {stats['random_initialized']:,}")
    print(f"  - Total vocabulary:       {vocab_size:,}")
    
    if anchor_mappings:
        print(f"\nSemantic Anchors Used ({len(anchor_mappings)}):")
        for archaic, modern in list(anchor_mappings.items())[:10]:
            print(f"  '{archaic}' → '{modern}'")
        if len(anchor_mappings) > 10:
            print(f"  ... and {len(anchor_mappings) - 10} more")
    
    return embedding_matrix, anchor_mappings


def create_dataloaders(
    tokens: List[str],
    vocab: Vocabulary,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    batch_size: int = config.BATCH_SIZE,
    seq_length: int = config.MAX_SEQ_LENGTH
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        tokens: List of tokens
        vocab: Vocabulary object
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        batch_size: Batch size
        seq_length: Sequence length
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Encode tokens
    encoded = vocab.encode(tokens)
    
    # Calculate split points
    n = len(encoded)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Split data
    train_data = encoded[:train_end]
    val_data = encoded[train_end:val_end]
    test_data = encoded[val_end:]
    
    print(f"\nData splits:")
    print(f"  - Training:   {len(train_data):,} tokens")
    print(f"  - Validation: {len(val_data):,} tokens")
    print(f"  - Test:       {len(test_data):,} tokens")
    
    # Create datasets
    train_dataset = ShakespeareDataset(train_data, seq_length)
    val_dataset = ShakespeareDataset(val_data, seq_length)
    test_dataset = ShakespeareDataset(test_data, seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    print(f"\nDataLoaders created:")
    print(f"  - Training batches:   {len(train_loader):,}")
    print(f"  - Validation batches: {len(val_loader):,}")
    print(f"  - Test batches:       {len(test_loader):,}")
    
    return train_loader, val_loader, test_loader


def prepare_data(use_cache: bool = True) -> Tuple:
    """
    Main function to prepare all data.
    Supports both word-level and BPE tokenization based on config.TOKENIZER_TYPE.
    
    Args:
        use_cache: If True, use cached embeddings/tokenizer if available (much faster)
    
    Returns:
        vocab: Vocabulary object (word-level or BPE wrapper)
        embedding_matrix: Pre-trained embedding matrix (None for BPE)
        anchor_mappings: Semantic anchor mappings used (empty for BPE)
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
        raw_train_data: Raw encoded training tokens for stride rebuilding (None for word-level)
    """
    print("=" * 70)
    print("PREPARING DATA")
    print("=" * 70)
    
    # 1. Download/load Shakespeare text
    text = download_shakespeare()
    
    if config.TOKENIZER_TYPE == "bpe":
        return _prepare_data_bpe(text, use_cache)
    else:
        return _prepare_data_word(text, use_cache)


def _prepare_data_bpe(text: str, use_cache: bool = True) -> Tuple:
    """Prepare data using BPE tokenization"""
    from bpe_tokenizer import BPETokenizer, BPEVocabulary
    
    print(f"\nUsing BPE tokenization (vocab_size={config.BPE_VOCAB_SIZE})")
    
    bpe_path = config.DATA_DIR / f"bpe_tokenizer_{config.BPE_VOCAB_SIZE}.json"
    
    # Train or load BPE tokenizer
    bpe = BPETokenizer(vocab_size=config.BPE_VOCAB_SIZE)
    if use_cache and bpe_path.exists():
        bpe.load(bpe_path)
    else:
        print("Training BPE tokenizer on Shakespeare corpus...")
        bpe.train(text, save_path=bpe_path)
    
    # Encode the full text
    encoded = bpe.encode(text)
    print(f"\nBPE encoded: {len(encoded):,} tokens (from {len(text):,} characters)")
    print(f"Compression ratio: {len(text) / len(encoded):.1f} chars per token")
    
    # Create vocabulary wrapper
    vocab = BPEVocabulary(bpe)
    
    # Split data
    n = len(encoded)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_data = encoded[:train_end]
    val_data = encoded[train_end:val_end]
    test_data = encoded[val_end:]
    
    print(f"\nData splits:")
    print(f"  - Training:   {len(train_data):,} tokens")
    print(f"  - Validation: {len(val_data):,} tokens")
    print(f"  - Test:       {len(test_data):,} tokens")
    
    # Create datasets with initial stride
    initial_stride = config.STRIDE_INITIAL
    train_dataset = ShakespeareDataset(train_data, config.MAX_SEQ_LENGTH, stride=initial_stride)
    val_dataset = ShakespeareDataset(val_data, config.MAX_SEQ_LENGTH)
    test_dataset = ShakespeareDataset(test_data, config.MAX_SEQ_LENGTH)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=config.DEVICE.type == 'cuda'
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=config.DEVICE.type == 'cuda'
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=config.DEVICE.type == 'cuda'
    )
    
    print(f"\nDataLoaders created (initial stride={initial_stride}):")
    print(f"  - Training batches:   {len(train_loader):,}")
    print(f"  - Validation batches: {len(val_loader):,}")
    print(f"  - Test batches:       {len(test_loader):,}")
    print(f"  - Stride schedule:    {initial_stride} → {config.STRIDE_MIN} (halving every {config.STRIDE_CONTRACT_EVERY} epochs)")
    
    # BPE uses randomly initialized embeddings (no pre-trained matrix)
    embedding_matrix = None
    anchor_mappings = {}
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE (BPE)")
    print("=" * 70)
    
    # Return train_data so Trainer can rebuild DataLoader with contracting stride
    return vocab, embedding_matrix, anchor_mappings, train_loader, val_loader, test_loader, train_data


def _prepare_data_word(text: str, use_cache: bool = True) -> Tuple:
    """Prepare data using word-level tokenization (original approach)"""
    
    # Cache file path
    cache_path = config.DATA_DIR / "embeddings_cache.pt"
    
    # 2. Tokenize
    tokenizer = ShakespeareTokenizer()
    tokens = tokenizer.tokenize(text)
    print(f"\nTokenized: {len(tokens):,} tokens")
    
    # 3. Try to load from cache first
    cached = None
    if use_cache:
        cached = load_embeddings_cache(cache_path)
    
    if cached is not None:
        vocab, embedding_matrix, anchor_mappings = cached
        print(f"Loaded {len(vocab):,} word embeddings from cache (skipped FastText loading!)")
    else:
        # 3b. Build vocabulary from scratch
        vocab = Vocabulary()
        vocab.build(tokens)
        
        # 4. Load FastText and align embeddings
        ft_model = load_fasttext_model()
        embedding_matrix, anchor_mappings = align_embeddings_with_fasttext(vocab, ft_model)
        
        # Free FastText model memory
        del ft_model
        
        # 5. Cache for next time
        if use_cache:
            save_embeddings_cache(vocab, embedding_matrix, anchor_mappings, cache_path)
    
    # 6. Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(tokens, vocab)
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE (Word-Level)")
    print("=" * 70)
    
    # No raw_train_data needed for word-level (no contracting stride)
    return vocab, embedding_matrix, anchor_mappings, train_loader, val_loader, test_loader, None


# For testing
if __name__ == "__main__":
    vocab, emb_matrix, anchors, train_dl, val_dl, test_dl = prepare_data()
    
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Embedding matrix shape: {emb_matrix.shape}")
    print(f"Number of anchors: {len(anchors)}")
    
    # Test a batch
    for x, y in train_dl:
        print(f"\nSample batch:")
        print(f"  Input shape: {x.shape}")
        print(f"  Target shape: {y.shape}")
        print(f"  Sample input: {vocab.decode(x[0].tolist()[:10])}")
        print(f"  Sample target: {vocab.decode(y[0].tolist()[:10])}")
        break
