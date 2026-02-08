"""
Transformer Architecture for Word-Level Shakespeare Text Generation

This module implements:
1. Multi-Head Self-Attention
2. Feed-Forward Network
3. Transformer Decoder Block
4. Full Transformer Decoder Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

import config


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Allows the model to jointly attend to information from different
    representation subspaces at different positions.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(
        self,
        embed_dim: int = config.EMBEDDING_DIM,
        num_heads: int = config.NUM_HEADS,
        dropout: float = config.ATTENTION_DROPOUT,
        bias: bool = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # For storing attention weights (useful for visualization)
        self.attention_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        store_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_dim)
            attention_mask: Optional mask of shape (seq_length, seq_length)
                           True/1 values indicate positions to MASK (ignore)
            store_attention: Whether to store attention weights for visualization
        
        Returns:
            Output tensor of shape (batch_size, seq_length, embed_dim)
        """
        batch_size, seq_length, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq, embed)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head: (batch, seq, num_heads, head_dim)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose for attention: (batch, num_heads, seq, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores: (batch, num_heads, seq, seq)
        # Q @ K^T / sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask (for causal/autoregressive attention)
        if attention_mask is not None:
            # attention_mask: (seq, seq) with True where we want to mask
            # Add large negative value to masked positions (softmax → ~0)
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(0).unsqueeze(0),  # Add batch and head dims
                float('-inf')
            )
        
        # Softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Store for visualization if requested
        if store_attention:
            self.attention_weights = attention_probs.detach()
        
        # Apply attention to values: (batch, num_heads, seq, head_dim)
        context = torch.matmul(attention_probs, V)
        
        # Reshape back: (batch, seq, num_heads, head_dim) → (batch, seq, embed)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.embed_dim)
        
        # Output projection
        output = self.out_proj(context)
        
        return output


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Or with GELU activation:
    FFN(x) = GELU(xW1 + b1)W2 + b2
    """
    
    def __init__(
        self,
        embed_dim: int = config.EMBEDDING_DIM,
        hidden_dim: int = config.FFN_HIDDEN_DIM,
        dropout: float = config.DROPOUT,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_dim)
        
        Returns:
            Output tensor of shape (batch_size, seq_length, embed_dim)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerDecoderBlock(nn.Module):
    """
    Single Transformer Decoder Block.
    
    Consists of:
    1. Masked Multi-Head Self-Attention (with residual + LayerNorm)
    2. Feed-Forward Network (with residual + LayerNorm)
    
    We use Pre-LayerNorm (more stable training) instead of Post-LayerNorm.
    """
    
    def __init__(
        self,
        embed_dim: int = config.EMBEDDING_DIM,
        num_heads: int = config.NUM_HEADS,
        ffn_hidden_dim: int = config.FFN_HIDDEN_DIM,
        dropout: float = config.DROPOUT,
        attention_dropout: float = config.ATTENTION_DROPOUT
    ):
        super().__init__()
        
        # Layer normalization (Pre-LN)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention
        self.self_attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            embed_dim=embed_dim,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout
        )
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        store_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of transformer decoder block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_dim)
            attention_mask: Causal mask for autoregressive generation
            store_attention: Whether to store attention weights
        
        Returns:
            Output tensor of shape (batch_size, seq_length, embed_dim)
        """
        # Self-attention with residual connection (Pre-LN)
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, attention_mask, store_attention)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward with residual connection (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class TransformerDecoder(nn.Module):
    """
    Full Transformer Decoder for Language Modeling.
    
    Stack of N transformer decoder blocks with:
    - Causal (autoregressive) attention masking
    - Final layer normalization
    """
    
    def __init__(
        self,
        num_layers: int = config.NUM_LAYERS,
        embed_dim: int = config.EMBEDDING_DIM,
        num_heads: int = config.NUM_HEADS,
        ffn_hidden_dim: int = config.FFN_HIDDEN_DIM,
        dropout: float = config.DROPOUT,
        attention_dropout: float = config.ATTENTION_DROPOUT,
        max_seq_length: int = config.MAX_SEQ_LENGTH
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Register causal mask buffer
        self.register_buffer(
            'causal_mask',
            self._create_causal_mask(max_seq_length)
        )
    
    def _create_causal_mask(self, size: int) -> torch.Tensor:
        """
        Create causal (look-ahead) mask for autoregressive generation.
        
        Returns a matrix where position i can only attend to positions <= i.
        True values indicate positions to MASK (ignore).
        
        Example for size=4:
        [[False, True,  True,  True ],   # Position 0: only see 0
         [False, False, True,  True ],   # Position 1: see 0, 1
         [False, False, False, True ],   # Position 2: see 0, 1, 2
         [False, False, False, False]]   # Position 3: see all
        """
        mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        store_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through all decoder layers.
        
        Args:
            x: Embedded input of shape (batch_size, seq_length, embed_dim)
            store_attention: Whether to store attention weights in each layer
        
        Returns:
            Output tensor of shape (batch_size, seq_length, embed_dim)
        """
        seq_length = x.size(1)
        
        # Get causal mask for current sequence length
        causal_mask = self.causal_mask[:seq_length, :seq_length]
        
        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, causal_mask, store_attention)
        
        # Final layer normalization
        x = self.final_norm(x)
        
        return x
    
    def get_attention_weights(self) -> list:
        """Get attention weights from all layers (if stored)"""
        weights = []
        for layer in self.layers:
            if layer.self_attention.attention_weights is not None:
                weights.append(layer.self_attention.attention_weights)
        return weights


def count_transformer_parameters(
    num_layers: int,
    embed_dim: int,
    num_heads: int,
    ffn_hidden_dim: int,
    vocab_size: int
) -> dict:
    """
    Calculate the number of parameters in the transformer.
    
    Returns breakdown by component.
    """
    # Per attention layer
    # Q, K, V projections: 3 * (embed_dim * embed_dim + embed_dim)
    # Output projection: embed_dim * embed_dim + embed_dim
    attention_params = 4 * (embed_dim * embed_dim + embed_dim)
    
    # Per FFN layer
    # FC1: embed_dim * ffn_hidden_dim + ffn_hidden_dim
    # FC2: ffn_hidden_dim * embed_dim + embed_dim
    ffn_params = (embed_dim * ffn_hidden_dim + ffn_hidden_dim + 
                  ffn_hidden_dim * embed_dim + embed_dim)
    
    # LayerNorm: 2 * embed_dim per layer (2 norms per block)
    norm_params = 4 * embed_dim
    
    # Per block total
    block_params = attention_params + ffn_params + norm_params
    
    # All layers
    decoder_params = num_layers * block_params
    
    # Final norm
    final_norm_params = 2 * embed_dim
    
    # Embedding (if not tied)
    embedding_params = vocab_size * embed_dim
    
    # Output projection (if not tied)
    output_params = embed_dim * vocab_size
    
    return {
        'per_attention': attention_params,
        'per_ffn': ffn_params,
        'per_block': block_params,
        'all_blocks': decoder_params,
        'final_norm': final_norm_params,
        'embedding': embedding_params,
        'output_projection': output_params,
        'total_with_tied': decoder_params + final_norm_params + embedding_params,
        'total_without_tied': decoder_params + final_norm_params + embedding_params + output_params
    }


# For testing
if __name__ == "__main__":
    print("Testing Transformer Components")
    print("=" * 60)
    
    batch_size = 4
    seq_length = 32
    embed_dim = config.EMBEDDING_DIM
    
    # Test Multi-Head Attention
    print("\n1. Testing Multi-Head Self-Attention")
    mha = MultiHeadSelfAttention()
    x = torch.randn(batch_size, seq_length, embed_dim)
    
    # Create causal mask
    causal_mask = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1)
    
    out = mha(x, causal_mask, store_attention=True)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Attention shape: {mha.attention_weights.shape}")
    
    # Test FFN
    print("\n2. Testing Feed-Forward Network")
    ffn = FeedForwardNetwork()
    out = ffn(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {out.shape}")
    
    # Test Decoder Block
    print("\n3. Testing Transformer Decoder Block")
    block = TransformerDecoderBlock()
    out = block(x, causal_mask)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {out.shape}")
    
    # Test Full Decoder
    print("\n4. Testing Full Transformer Decoder")
    decoder = TransformerDecoder()
    out = decoder(x, store_attention=True)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Attention weights stored: {len(decoder.get_attention_weights())} layers")
    
    # Parameter count
    print("\n5. Parameter Count")
    params = count_transformer_parameters(
        num_layers=config.NUM_LAYERS,
        embed_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        ffn_hidden_dim=config.FFN_HIDDEN_DIM,
        vocab_size=8000
    )
    for key, value in params.items():
        print(f"   {key}: {value:,}")
