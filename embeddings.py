"""
Embedding Module for Word-Level Shakespeare Text Generation

This module handles:
1. Pre-trained embedding initialization (from FastText)
2. Semantic anchor preservation during training
3. Embedding fine-tuning with differential learning rates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

import config


class ShakespeareEmbedding(nn.Module):
    """
    Embedding layer initialized from FastText with semantic anchor preservation.
    
    Features:
    - Pre-trained initialization from FastText
    - Optional freezing of embeddings
    - Semantic anchor preservation loss
    - Positional encoding for Transformer
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        max_seq_length: int = config.MAX_SEQ_LENGTH,
        dropout: float = config.DROPOUT,
        pretrained_weights: Optional[torch.Tensor] = None,
        freeze: bool = config.FREEZE_EMBEDDINGS,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Word embedding layer
        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        
        # Initialize with pre-trained weights if provided
        if pretrained_weights is not None:
            assert pretrained_weights.shape == (vocab_size, embedding_dim), \
                f"Shape mismatch: {pretrained_weights.shape} vs ({vocab_size}, {embedding_dim})"
            self.word_embedding.weight.data.copy_(pretrained_weights)
            print(f"Initialized embeddings from pre-trained weights")
        
        # Optionally freeze embeddings
        if freeze:
            self.word_embedding.weight.requires_grad = False
            print("Embeddings frozen (not trainable)")
        else:
            print("Embeddings trainable (will fine-tune)")
        
        # Positional encoding (sinusoidal - not learned)
        self.register_buffer(
            'positional_encoding',
            self._create_positional_encoding(max_seq_length, embedding_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for embeddings
        self.scale = torch.sqrt(torch.tensor(embedding_dim, dtype=torch.float32))
    
    def _create_positional_encoding(
        self,
        max_seq_length: int,
        embedding_dim: int
    ) -> torch.Tensor:
        """
        Create sinusoidal positional encoding.
        
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        position = torch.arange(max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        
        pe = torch.zeros(max_seq_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Shape: (1, max_seq_length, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of word indices, shape (batch_size, seq_length)
        
        Returns:
            Embedded and position-encoded tensor, shape (batch_size, seq_length, embedding_dim)
        """
        seq_length = x.size(1)
        
        # Get word embeddings and scale
        word_emb = self.word_embedding(x) * self.scale
        
        # Add positional encoding
        pos_emb = self.positional_encoding[:, :seq_length, :]
        
        # Combine and apply dropout
        embedded = self.dropout(word_emb + pos_emb)
        
        return embedded
    
    def get_embedding_weights(self) -> torch.Tensor:
        """Get the embedding weight matrix"""
        return self.word_embedding.weight


class SemanticAnchorLoss(nn.Module):
    """
    Auxiliary loss to preserve semantic relationships during fine-tuning.
    
    Encourages archaic words to remain close to their modern equivalents
    in the embedding space.
    """
    
    def __init__(
        self,
        vocab,  # Vocabulary object
        anchor_mappings: Dict[str, str],
        weight: float = config.SEMANTIC_PRESERVATION_WEIGHT
    ):
        super().__init__()
        
        self.weight = weight
        self.anchor_pairs = []  # List of (archaic_idx, modern_idx) pairs
        
        # Build anchor pairs
        for archaic, modern in anchor_mappings.items():
            if archaic in vocab.word_to_idx:
                archaic_idx = vocab.word_to_idx[archaic]
                
                # Handle multi-word anchors
                if '_' in modern:
                    # For multi-word, we'll use the first word as anchor
                    # (simplified - could also average)
                    modern = modern.split('_')[0]
                
                if modern in vocab.word_to_idx:
                    modern_idx = vocab.word_to_idx[modern]
                    self.anchor_pairs.append((archaic_idx, modern_idx))
        
        print(f"SemanticAnchorLoss: {len(self.anchor_pairs)} anchor pairs")
    
    def forward(self, embedding_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic preservation loss.
        
        Penalizes when archaic words drift too far from their modern anchors.
        
        Args:
            embedding_weights: Embedding matrix, shape (vocab_size, embedding_dim)
        
        Returns:
            Scalar loss value
        """
        if len(self.anchor_pairs) == 0 or self.weight == 0:
            return torch.tensor(0.0, device=embedding_weights.device)
        
        total_loss = 0.0
        
        for archaic_idx, modern_idx in self.anchor_pairs:
            archaic_emb = embedding_weights[archaic_idx]
            modern_emb = embedding_weights[modern_idx]
            
            # Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
            similarity = F.cosine_similarity(
                archaic_emb.unsqueeze(0),
                modern_emb.unsqueeze(0)
            )
            
            # Loss: penalize low similarity (want similarity close to 1)
            # Use (1 - sim) so loss is 0 when perfectly similar
            total_loss += (1 - similarity)
        
        # Average and weight
        avg_loss = total_loss / len(self.anchor_pairs)
        
        return self.weight * avg_loss


class EmbeddingWithTiedWeights(nn.Module):
    """
    Embedding layer with weight tying to output projection.
    
    Tying input embeddings to output weights:
    - Reduces parameters
    - Improves coherence (similar words predict each other)
    - Standard practice in language models
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        max_seq_length: int = config.MAX_SEQ_LENGTH,
        dropout: float = config.DROPOUT,
        pretrained_weights: Optional[torch.Tensor] = None,
        pad_idx: int = 0
    ):
        super().__init__()
        
        # Core embedding
        self.embedding = ShakespeareEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pretrained_weights=pretrained_weights,
            freeze=False,  # Must be trainable for tying to work well
            pad_idx=pad_idx
        )
        
        # Output projection (tied to embedding weights)
        self.output_projection = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        # Tie weights
        self.output_projection.weight = self.embedding.word_embedding.weight
        print("Embedding weights tied to output projection")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for input"""
        return self.embedding(x)
    
    def project_to_vocab(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits"""
        return self.output_projection(hidden)
    
    def get_embedding_weights(self) -> torch.Tensor:
        """Get the embedding weight matrix"""
        return self.embedding.get_embedding_weights()


def analyze_embedding_space(
    embedding_weights: torch.Tensor,
    vocab,
    words_to_analyze: Optional[list] = None,
    top_k: int = 5
) -> Dict:
    """
    Analyze the embedding space by finding similar words.
    
    Args:
        embedding_weights: Embedding matrix
        vocab: Vocabulary object
        words_to_analyze: List of words to find neighbors for
        top_k: Number of similar words to return
    
    Returns:
        Dictionary mapping words to their nearest neighbors
    """
    if words_to_analyze is None:
        words_to_analyze = ['king', 'love', 'death', 'thou', 'hath', 'wherefore']
    
    # Normalize embeddings for cosine similarity
    normalized = F.normalize(embedding_weights, p=2, dim=1)
    
    results = {}
    
    for word in words_to_analyze:
        if word not in vocab.word_to_idx:
            continue
        
        word_idx = vocab.word_to_idx[word]
        word_emb = normalized[word_idx].unsqueeze(0)
        
        # Compute similarities to all words
        similarities = torch.mm(word_emb, normalized.t()).squeeze()
        
        # Get top-k (excluding the word itself)
        top_values, top_indices = torch.topk(similarities, top_k + 1)
        
        neighbors = []
        for val, idx in zip(top_values[1:], top_indices[1:]):  # Skip first (itself)
            neighbor_word = vocab.idx_to_word[idx.item()]
            neighbors.append((neighbor_word, val.item()))
        
        results[word] = neighbors
    
    return results


# For testing
if __name__ == "__main__":
    # Test embedding module
    vocab_size = 1000
    batch_size = 4
    seq_length = 32
    
    # Create random pre-trained weights
    pretrained = torch.randn(vocab_size, config.EMBEDDING_DIM)
    
    # Test basic embedding
    emb = ShakespeareEmbedding(
        vocab_size=vocab_size,
        pretrained_weights=pretrained
    )
    
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    out = emb(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: ({batch_size}, {seq_length}, {config.EMBEDDING_DIM})")
    
    # Test tied weights
    tied_emb = EmbeddingWithTiedWeights(
        vocab_size=vocab_size,
        pretrained_weights=pretrained
    )
    
    embedded = tied_emb(x)
    logits = tied_emb.project_to_vocab(embedded)
    
    print(f"\nTied embedding output: {embedded.shape}")
    print(f"Projection to vocab: {logits.shape}")
