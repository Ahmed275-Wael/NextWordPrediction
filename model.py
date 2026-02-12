"""
Full Model: Combining Embeddings and Transformer for Shakespeare Text Generation

This module combines:
1. Fine-tuned FastText embeddings with semantic anchors
2. Transformer decoder for next-word prediction
3. Generation utilities for text synthesis
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

import config
from embeddings import ShakespeareEmbedding, EmbeddingWithTiedWeights, SemanticAnchorLoss
from transformer import TransformerDecoder


class ShakespeareTransformer(nn.Module):
    """
    Complete Transformer Language Model for Shakespeare Text Generation.
    
    Architecture:
    1. Embedding layer (FastText-initialized with semantic anchors)
    2. Positional encoding (sinusoidal)
    3. Transformer decoder (N layers of self-attention + FFN)
    4. Output projection (optionally tied to embeddings)
    
    Features:
    - Pre-trained embedding initialization
    - Semantic anchor preservation
    - Weight tying between embeddings and output
    - Causal (autoregressive) attention
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = config.EMBEDDING_DIM,
        num_layers: int = config.NUM_LAYERS,
        num_heads: int = config.NUM_HEADS,
        ffn_hidden_dim: int = config.FFN_HIDDEN_DIM,
        dropout: float = config.DROPOUT,
        max_seq_length: int = config.MAX_SEQ_LENGTH,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        tie_weights: bool = True,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.tie_weights = tie_weights
        
        # Embedding layer (with positional encoding)
        if tie_weights:
            self.embedding = EmbeddingWithTiedWeights(
                vocab_size=vocab_size,
                embedding_dim=embed_dim,
                max_seq_length=max_seq_length,
                dropout=dropout,
                pretrained_weights=pretrained_embeddings,
                pad_idx=pad_idx
            )
        else:
            self.embedding = ShakespeareEmbedding(
                vocab_size=vocab_size,
                embedding_dim=embed_dim,
                max_seq_length=max_seq_length,
                dropout=dropout,
                pretrained_weights=pretrained_embeddings,
                freeze=False,
                pad_idx=pad_idx
            )
            # Separate output projection
            self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Transformer decoder
        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights following GPT-2/nanoGPT conventions.
        
        Key insight: Residual projections (attention out_proj and FFN fc2) are
        initialised with std = 0.02 / sqrt(2 * num_layers). This prevents the
        residual stream from growing proportionally to sqrt(N) at init, ensuring
        stable early training — critical for small datasets where early gradients
        shape the entire learning trajectory.
        """
        residual_std = 0.02 / math.sqrt(2 * config.NUM_LAYERS)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.requires_grad:
                    # Scaled init for residual projections, normal init for others
                    if hasattr(module, '_is_residual') and module._is_residual:
                        nn.init.normal_(module.weight, mean=0.0, std=residual_std)
                    else:
                        nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        store_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for next-word prediction.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_length)
            store_attention: Whether to store attention weights
        
        Returns:
            Logits over vocabulary of shape (batch_size, seq_length, vocab_size)
        """
        # Embed input tokens
        embedded = self.embedding(input_ids)  # (batch, seq, embed_dim)
        
        # Pass through transformer decoder
        hidden = self.decoder(embedded, store_attention)  # (batch, seq, embed_dim)
        
        # Project to vocabulary
        if self.tie_weights:
            logits = self.embedding.project_to_vocab(hidden)
        else:
            logits = self.output_projection(hidden)
        
        return logits  # (batch, seq, vocab_size)
    
    def get_embedding_weights(self) -> torch.Tensor:
        """Get the embedding weight matrix"""
        return self.embedding.get_embedding_weights()
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """Get attention weights from all layers"""
        return self.decoder.get_attention_weights()


class TextGenerator:
    """
    Text generation utilities for Shakespeare-style text.
    
    Supports:
    - Greedy decoding
    - Temperature sampling
    - Top-k sampling
    - Top-p (nucleus) sampling
    - Repetition penalty
    """
    
    def __init__(
        self,
        model: ShakespeareTransformer,
        vocab,  # Vocabulary object
        device: torch.device = config.DEVICE
    ):
        self.model = model
        self.vocab = vocab
        self.device = device
        
        # Special token indices
        self.pad_idx = vocab.pad_idx
        self.bos_idx = vocab.bos_idx
        self.eos_idx = vocab.eos_idx
        self.unk_idx = vocab.unk_idx
    
    def generate(
        self,
        seed_text: str,
        max_length: int = config.MAX_GENERATE_LENGTH,
        temperature: float = config.TEMPERATURE,
        top_k: int = config.TOP_K,
        top_p: float = config.TOP_P,
        repetition_penalty: float = config.REPETITION_PENALTY,
        stop_on_eos: bool = True
    ) -> str:
        """
        Generate text continuation from seed.
        
        Args:
            seed_text: Initial text to continue from
            max_length: Maximum number of words to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k tokens for sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            stop_on_eos: Stop when EOS token is generated
        
        Returns:
            Generated text (including seed)
        """
        self.model.eval()
        
        # Tokenize seed text based on tokenizer type
        if config.TOKENIZER_TYPE == "bpe":
            # BPE: encode directly from text
            current_tokens = self.vocab.encode(seed_text)
        else:
            # Word-level: tokenize then encode
            from data_loader import ShakespeareTokenizer
            tokenizer = ShakespeareTokenizer()
            seed_tokens = tokenizer.tokenize(seed_text)
            current_tokens = self.vocab.encode(seed_tokens)
        
        # Track generated tokens for repetition penalty
        generated = list(current_tokens)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input (limit to max sequence length)
                input_tokens = generated[-config.MAX_SEQ_LENGTH:]
                input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
                
                # Get model predictions
                logits = self.model(input_ids)  # (1, seq_len, vocab_size)
                
                # Get logits for next token (last position)
                next_token_logits = logits[0, -1, :]  # (vocab_size,)
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, generated, repetition_penalty
                    )
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)
                
                # Sample from filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Check for EOS
                if stop_on_eos and next_token == self.eos_idx:
                    break
                
                # Append generated token
                generated.append(next_token)
        
        # Decode back to text
        if config.TOKENIZER_TYPE == "bpe":
            output_text = self.vocab.decode_to_text(generated)
        else:
            from data_loader import ShakespeareTokenizer
            tokenizer = ShakespeareTokenizer()
            generated_tokens = self.vocab.decode(generated)
            output_text = tokenizer.detokenize(generated_tokens)
        
        return output_text
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: List[int],
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        for token_id in set(generated[-50:]):  # Look at last 50 tokens
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
        return logits
    
    def _top_k_filtering(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """Filter to keep only top-k tokens"""
        if top_k <= 0:
            return logits
        
        # Get the top-k values
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """Filter using nucleus (top-p) sampling"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def generate_batch(
        self,
        seed_texts: List[str],
        **kwargs
    ) -> List[str]:
        """Generate multiple texts (sequential, not batched)"""
        return [self.generate(seed, **kwargs) for seed in seed_texts]


def create_model(
    vocab_size: int,
    pretrained_embeddings: Optional[torch.Tensor] = None,
    device: torch.device = config.DEVICE
) -> ShakespeareTransformer:
    """
    Factory function to create the model.
    
    Args:
        vocab_size: Size of vocabulary
        pretrained_embeddings: Optional pre-trained embedding matrix
        device: Device to place model on
    
    Returns:
        Initialized ShakespeareTransformer model
    """
    model = ShakespeareTransformer(
        vocab_size=vocab_size,
        embed_dim=config.EMBEDDING_DIM,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        ffn_hidden_dim=config.FFN_HIDDEN_DIM,
        dropout=config.DROPOUT,
        max_seq_length=config.MAX_SEQ_LENGTH,
        pretrained_embeddings=pretrained_embeddings,
        tie_weights=True,
        pad_idx=0
    )
    
    model = model.to(device)
    
    return model


# For testing
if __name__ == "__main__":
    print("Testing Shakespeare Transformer Model")
    print("=" * 60)
    
    vocab_size = 8000
    batch_size = 4
    seq_length = 32
    
    # Create random pre-trained embeddings
    pretrained = torch.randn(vocab_size, config.EMBEDDING_DIM)
    
    # Create model
    model = create_model(vocab_size, pretrained)
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(config.DEVICE)
    
    logits = model(input_ids, store_attention=True)
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_length}, {vocab_size})")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test attention weights
    attention_weights = model.get_attention_weights()
    print(f"\nAttention weights stored: {len(attention_weights)} layers")
    if attention_weights:
        print(f"Attention shape per layer: {attention_weights[0].shape}")
