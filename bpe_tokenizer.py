"""
BPE (Byte-Pair Encoding) Tokenizer for Shakespeare Text Generation

This module implements BPE tokenization as an alternative to word-level tokenization.
BPE splits text into subword units, which:
1. Eliminates OOV (unknown word) problems
2. Reduces vocabulary size while covering all text
3. Typically achieves lower perplexity than word-level models

Uses HuggingFace's 'tokenizers' library for fast BPE training and encoding.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import config


class BPETokenizer:
    """
    BPE tokenizer that trains on Shakespeare text.
    
    Wraps HuggingFace's tokenizers library for:
    - Training a custom BPE vocabulary on the Shakespeare corpus
    - Encoding text to subword token IDs
    - Decoding token IDs back to text
    """
    
    def __init__(self, vocab_size: int = config.BPE_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.tokenizer = None
        
        # Special tokens (same as word-level for consistency)
        self.special_tokens = [
            config.PAD_TOKEN,   # 0
            config.UNK_TOKEN,   # 1
            config.BOS_TOKEN,   # 2
            config.EOS_TOKEN,   # 3
        ]
        
        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3
    
    def train(self, text: str, save_path: Optional[Path] = None):
        """
        Train BPE tokenizer on the given text.
        
        Args:
            text: Raw text to train on
            save_path: Optional path to save the trained tokenizer
        """
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.normalizers import Lowercase
        
        # Create BPE tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token=config.UNK_TOKEN))
        
        # Normalize to lowercase (consistent with word-level)
        self.tokenizer.normalizer = Lowercase()
        
        # Pre-tokenize on whitespace (then BPE splits further)
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Train
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2,
            show_progress=True,
        )
        
        # Train from text (write to temp file for the trainer)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(text)
            temp_path = f.name
        
        self.tokenizer.train([temp_path], trainer)
        
        # Clean up temp file
        Path(temp_path).unlink()
        
        # Update indices after training
        self.pad_idx = self.tokenizer.token_to_id(config.PAD_TOKEN)
        self.unk_idx = self.tokenizer.token_to_id(config.UNK_TOKEN)
        self.bos_idx = self.tokenizer.token_to_id(config.BOS_TOKEN)
        self.eos_idx = self.tokenizer.token_to_id(config.EOS_TOKEN)
        
        actual_vocab_size = self.tokenizer.get_vocab_size()
        print(f"BPE Tokenizer trained:")
        print(f"  - Vocabulary size: {actual_vocab_size:,}")
        print(f"  - Special tokens: {len(self.special_tokens)}")
        
        # Save if path provided
        if save_path:
            self.save(save_path)
        
        return self
    
    def save(self, path: Path):
        """Save tokenizer to disk"""
        self.tokenizer.save(str(path))
        print(f"BPE tokenizer saved to {path}")
    
    def load(self, path: Path):
        """Load tokenizer from disk"""
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(path))
        
        self.pad_idx = self.tokenizer.token_to_id(config.PAD_TOKEN)
        self.unk_idx = self.tokenizer.token_to_id(config.UNK_TOKEN)
        self.bos_idx = self.tokenizer.token_to_id(config.BOS_TOKEN)
        self.eos_idx = self.tokenizer.token_to_id(config.EOS_TOKEN)
        
        print(f"BPE tokenizer loaded from {path}")
        print(f"  - Vocabulary size: {self.tokenizer.get_vocab_size():,}")
        return self
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into BPE subword tokens"""
        encoding = self.tokenizer.encode(text)
        return encoding.tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(ids)
    
    def decode_tokens(self, ids: List[int]) -> List[str]:
        """Decode token IDs to individual tokens (for display)"""
        return [self.tokenizer.id_to_token(i) or config.UNK_TOKEN for i in ids]
    
    def get_vocab_size(self) -> int:
        """Get actual vocabulary size"""
        return self.tokenizer.get_vocab_size()
    
    def token_to_id(self, token: str) -> int:
        """Convert token to ID"""
        result = self.tokenizer.token_to_id(token)
        return result if result is not None else self.unk_idx
    
    def id_to_token(self, idx: int) -> str:
        """Convert ID to token"""
        return self.tokenizer.id_to_token(idx) or config.UNK_TOKEN


class BPEVocabulary:
    """
    Vocabulary wrapper for BPE tokenizer.
    Provides the same interface as the word-level Vocabulary class
    so it can be used interchangeably throughout the codebase.
    """
    
    def __init__(self, bpe_tokenizer: BPETokenizer):
        self.bpe = bpe_tokenizer
        
        # Match the word-level Vocabulary interface
        self.pad_idx = bpe_tokenizer.pad_idx
        self.unk_idx = bpe_tokenizer.unk_idx
        self.bos_idx = bpe_tokenizer.bos_idx
        self.eos_idx = bpe_tokenizer.eos_idx
        
        # Build word_to_idx and idx_to_word from BPE vocab
        self.word_to_idx = {}
        self.idx_to_word = {}
        vocab = bpe_tokenizer.tokenizer.get_vocab()
        for token, idx in vocab.items():
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
    
    def encode(self, tokens_or_text) -> List[int]:
        """
        Encode input to IDs.
        Accepts either a string (BPE encodes it) or list of tokens.
        """
        if isinstance(tokens_or_text, str):
            return self.bpe.encode(tokens_or_text)
        elif isinstance(tokens_or_text, list):
            # If given a list of BPE tokens, convert to IDs
            return [self.word_to_idx.get(t, self.unk_idx) for t in tokens_or_text]
        return []
    
    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens"""
        return [self.idx_to_word.get(idx, config.UNK_TOKEN) for idx in indices]
    
    def decode_to_text(self, indices: List[int]) -> str:
        """Convert indices directly to readable text"""
        return self.bpe.decode(indices)
    
    def __len__(self) -> int:
        return self.bpe.get_vocab_size()
    
    def get_archaic_words(self) -> List[str]:
        """Get archaic words in vocabulary (fewer for BPE since words are split)"""
        archaic = []
        for word in config.SEMANTIC_ANCHORS:
            if word in self.word_to_idx:
                archaic.append(word)
        return archaic


# For testing
if __name__ == "__main__":
    # Test BPE tokenizer
    sample_text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles,
    And by opposing end them. To die, to sleep;
    """
    
    print("Testing BPE Tokenizer")
    print("=" * 60)
    
    tokenizer = BPETokenizer(vocab_size=200)
    tokenizer.train(sample_text)
    
    test = "To be or not to be"
    tokens = tokenizer.tokenize(test)
    ids = tokenizer.encode(test)
    decoded = tokenizer.decode(ids)
    
    print(f"\nInput:    '{test}'")
    print(f"Tokens:   {tokens}")
    print(f"IDs:      {ids}")
    print(f"Decoded:  '{decoded}'")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
