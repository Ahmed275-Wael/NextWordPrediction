"""
Data Augmentation Module for Shakespeare Text Generation

Implements embedding-based synonym replacement that preserves semantic meaning.
"""

import torch
import numpy as np
import random
from typing import List, Optional, Dict
from collections import defaultdict


class EmbeddingAugmenter:
    """
    Augments text by replacing words with semantically similar ones
    based on embedding similarity.
    """
    
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        vocab,
        replacement_prob: float = 0.15,
        top_k: int = 5,
        min_similarity: float = 0.5
    ):
        """
        Args:
            embedding_matrix: Pre-trained embedding matrix (vocab_size x embed_dim)
            vocab: Vocabulary object with word2idx mapping
            replacement_prob: Probability of replacing each word
            top_k: Number of similar words to consider
            min_similarity: Minimum cosine similarity for replacement
        """
        self.embedding_matrix = embedding_matrix
        self.vocab = vocab
        self.replacement_prob = replacement_prob
        self.top_k = top_k
        self.min_similarity = min_similarity
        
        # Pre-compute normalized embeddings for fast similarity
        self.normalized_embeddings = self._normalize_embeddings()
        
        # Cache for similar words (computed lazily)
        self._similarity_cache: Dict[int, List[int]] = {}
        
        # Words to never replace (special tokens, punctuation)
        self.protected_tokens = set([
            vocab.pad_token, vocab.unk_token,
            '.', ',', '!', '?', ';', ':', "'", '"', '-', '(', ')'
        ])
        self.protected_indices = set([
            vocab.word_to_idx.get(t, -1) for t in self.protected_tokens
        ])
        
        print(f"EmbeddingAugmenter initialized:")
        print(f"  - Replacement probability: {replacement_prob}")
        print(f"  - Top-k candidates: {top_k}")
        print(f"  - Min similarity: {min_similarity}")
    
    def _normalize_embeddings(self) -> torch.Tensor:
        """Normalize embeddings for cosine similarity computation"""
        norms = self.embedding_matrix.norm(dim=1, keepdim=True)
        # Avoid division by zero
        norms = norms.clamp(min=1e-8)
        return self.embedding_matrix / norms
    
    def _find_similar_words(self, word_idx: int) -> List[int]:
        """Find top-k similar words for a given word index"""
        if word_idx in self._similarity_cache:
            return self._similarity_cache[word_idx]
        
        # Get embedding for this word
        word_emb = self.normalized_embeddings[word_idx]
        
        # Compute cosine similarity with all words
        similarities = torch.mv(self.normalized_embeddings, word_emb)
        
        # Get top-k+1 (including the word itself)
        top_values, top_indices = torch.topk(similarities, self.top_k + 1)
        
        # Filter: remove the word itself and words below threshold
        similar_words = []
        for sim, idx in zip(top_values.tolist(), top_indices.tolist()):
            if idx != word_idx and sim >= self.min_similarity:
                if idx not in self.protected_indices:
                    similar_words.append(idx)
        
        # Cache the result
        self._similarity_cache[word_idx] = similar_words
        return similar_words
    
    def augment_sequence(self, token_indices: List[int]) -> List[int]:
        """
        Augment a sequence by replacing some tokens with similar ones.
        
        Args:
            token_indices: List of token indices
            
        Returns:
            Augmented list of token indices
        """
        augmented = []
        
        for idx in token_indices:
            # Skip protected tokens
            if idx in self.protected_indices:
                augmented.append(idx)
                continue
            
            # Random chance to replace
            if random.random() < self.replacement_prob:
                similar_words = self._find_similar_words(idx)
                if similar_words:
                    # Randomly pick one of the similar words
                    replacement = random.choice(similar_words)
                    augmented.append(replacement)
                else:
                    augmented.append(idx)
            else:
                augmented.append(idx)
        
        return augmented
    
    def augment_batch(
        self,
        batch: torch.Tensor,
        augment_prob: float = 0.5
    ) -> torch.Tensor:
        """
        Augment a batch of sequences.
        
        Args:
            batch: Tensor of shape (batch_size, seq_len)
            augment_prob: Probability of augmenting each sequence
            
        Returns:
            Augmented batch tensor
        """
        augmented_batch = batch.clone()
        
        for i in range(batch.size(0)):
            if random.random() < augment_prob:
                sequence = batch[i].tolist()
                augmented_seq = self.augment_sequence(sequence)
                augmented_batch[i] = torch.tensor(augmented_seq)
        
        return augmented_batch


class SimpleAugmenter:
    """
    Simpler augmentation without pre-computed similarities.
    Uses random swap and occasionally skips words.
    """
    
    def __init__(
        self,
        vocab,
        swap_prob: float = 0.1,
        enabled: bool = True
    ):
        self.vocab = vocab
        self.swap_prob = swap_prob
        self.enabled = enabled
        
        # Protected tokens
        self.protected_indices = set([
            vocab.pad_idx, vocab.unk_idx
        ])
    
    def random_swap(self, token_indices: List[int]) -> List[int]:
        """Randomly swap adjacent tokens"""
        if len(token_indices) < 3:
            return token_indices
        
        augmented = token_indices.copy()
        
        # Try a few swaps
        n_swaps = max(1, int(len(token_indices) * self.swap_prob))
        
        for _ in range(n_swaps):
            # Pick random position (not first or last)
            pos = random.randint(1, len(augmented) - 2)
            
            # Don't swap protected tokens
            if (augmented[pos] not in self.protected_indices and
                augmented[pos + 1] not in self.protected_indices):
                # Swap
                augmented[pos], augmented[pos + 1] = augmented[pos + 1], augmented[pos]
        
        return augmented
    
    def augment_batch(
        self,
        batch: torch.Tensor,
        augment_prob: float = 0.3
    ) -> torch.Tensor:
        """Augment a batch with random swaps"""
        if not self.enabled:
            return batch
            
        augmented_batch = batch.clone()
        
        for i in range(batch.size(0)):
            if random.random() < augment_prob:
                sequence = batch[i].tolist()
                augmented_seq = self.random_swap(sequence)
                augmented_batch[i] = torch.tensor(augmented_seq, dtype=batch.dtype)
        
        return augmented_batch
