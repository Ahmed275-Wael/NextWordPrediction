"""
AWD-LSTM Language Model for Shakespeare Text Generation

Implements the full AWD-LSTM (Merity, Keskar & Socher, 2018):
    "Regularizing and Optimizing LSTM Language Models"
    https://arxiv.org/abs/1708.02182
    Reference: salesforce/awd-lstm-lm (GitHub)

Key techniques vs vanilla LSTM:
    1. Weight Dropout (DropConnect) on recurrent hidden-to-hidden weights
    2. Variational Dropout — same mask across all timesteps
    3. Embedding Dropout — randomly zeros entire word rows
    4. AR (Activation Regularization) — L2 on LSTM outputs
    5. TAR (Temporal Activation Regularization) — L2 on consecutive diffs
    6. Weight tying between embedding and output projection

Architecture:
    Embedding(300d, embed_drop) → VarDrop → LSTM_L1(300→1150, wdrop)
    → VarDrop → LSTM_L2(1150→1150, wdrop) → VarDrop → LSTM_L3(1150→300, wdrop)
    → VarDrop → Linear(tied) → Logits

Expected: Val PPL ~120-150 on Shakespeare BPE-5000
(AWD-LSTM was PPL 57.3 on PTB with 24M params and 10K word-level vocab)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

import config


# ============================================================================
# AWD-LSTM Building Blocks
# ============================================================================

class LockedDropout(nn.Module):
    """
    Variational Dropout (Gal & Ghahramani, 2016).
    
    Applies the SAME dropout mask across all timesteps, unlike standard
    dropout which samples a new mask per timestep. This prevents the
    network from learning to "route around" dropped units at different
    timesteps, resulting in much more effective regularization for RNNs.
    
    Reference: Eq. 5 in Merity et al. (2018)
    """
    def forward(self, x: torch.Tensor, dropout: float = 0.5) -> torch.Tensor:
        if not self.training or dropout == 0:
            return x
        # x: (batch, seq_len, features) — mask is (batch, 1, features)
        # so the SAME mask is applied to every timestep
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)  # Scale to preserve expected values
        return x * mask


class WeightDrop(nn.Module):
    """
    DropConnect on recurrent hidden-to-hidden weights (Wan et al., 2013).
    
    Instead of dropping activations, this drops individual WEIGHTS in
    the weight_hh matrices. This is the #1 most important technique
    in AWD-LSTM — it prevents co-adaptation in the recurrent connections
    without disrupting the temporal information flow.
    
    Implementation: Before each forward pass, copy weight_hh, apply
    dropout to it, and swap it in. After forward, swap the original back.
    
    Reference: Section 3 of Merity et al. (2018)
    """
    def __init__(self, module: nn.Module, weights: List[str], dropout: float = 0.0):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        
        # Save original weights as parameters (rename originals)
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w)
            # Delete the original parameter from the module
            delattr(self.module, name_w)
            # Register the original under a new name (raw_*)
            self.module.register_parameter(name_w + '_raw', nn.Parameter(raw_w.data))
            # Use object.__setattr__ to bypass PyTorch's type checking
            # (nn.RNN.__setattr__ rejects non-Parameter tensors)
            object.__setattr__(self.module, name_w, raw_w.data)
    
    def _setweights(self):
        """Apply dropout to raw weights and set them for forward pass."""
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            if self.training:
                # DropConnect: zero out individual weights
                mask = raw_w.new_ones(raw_w.size()).bernoulli_(1 - self.dropout)
                w = raw_w * mask / (1 - self.dropout)
            else:
                w = raw_w
            # Bypass PyTorch's type check on nn.RNN modules
            object.__setattr__(self.module, name_w, w)
    
    def forward(self, *args):
        self._setweights()
        return self.module(*args)


def embedded_dropout(embed: nn.Embedding, words: torch.Tensor,
                     dropout: float = 0.1, scale: Optional[float] = None) -> torch.Tensor:
    """
    Embedding-level dropout (Gal & Ghahramani, 2016).
    
    Randomly zeros out ENTIRE embedding vectors for some words in the
    vocabulary. This is different from applying dropout to the output —
    it simulates the effect of rare words by completely removing some
    words' representations during training.
    
    Reference: Section 4.1 of Merity et al. (2018)
    """
    if dropout > 0 and embed.training:
        # Create a mask over the vocabulary (not over the sequence)
        mask = embed.weight.new_empty(embed.weight.size(0), 1).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)
        masked_weight = embed.weight * mask
    else:
        masked_weight = embed.weight
    
    if scale is not None:
        masked_weight = masked_weight * scale
    
    padding_idx = embed.padding_idx
    return F.embedding(
        words, masked_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse
    )


# ============================================================================
# AWD-LSTM Model
# ============================================================================

class ShakespeareLSTM(nn.Module):
    """
    AWD-LSTM Language Model (Merity et al., 2018).
    
    Full implementation of "Regularizing and Optimizing LSTM Language Models"
    with all 5 regularization techniques.
    
    Architecture (3-layer, following the paper):
        Embed(300d, embed_drop=0.1) → VarDrop(0.3)
        → LSTM_L1(300→1150, wdrop=0.5)  → VarDrop(0.3)
        → LSTM_L2(1150→1150, wdrop=0.5) → VarDrop(0.3)
        → LSTM_L3(1150→300, wdrop=0.5)  → VarDrop(0.4)
        → Linear(tied, 300→vocab) → Logits
    
    Parameters (~10M with BPE-5000):
        Embedding:    5000 × 300 = 1,500,000  (tied with output)
        LSTM L1:      4×(300+1150)×1150 = 6,670,000
        LSTM L2:      4×(1150+1150)×1150 = 10,580,000
        LSTM L3:      4×(1150+300)×300 = 1,740,000  
        # Actually smaller: last layer outputs embed_dim for weight tying
        Total:        ~9-11M (comparable reference: AWD-LSTM used 24M on PTB)
        
    Hyper-parameters (Merity et al. Table 4, adapted for our scale):
        dropoute (embedding dropout):    0.1
        dropouti (input variational):    0.3  (paper: 0.65)
        dropouth (hidden variational):   0.3  (paper: 0.3)
        dropouto (output variational):   0.4  (paper: 0.4)
        wdrop (weight dropout):          0.5  (paper: 0.5)
        AR alpha:                        2.0
        TAR beta:                        1.0
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_size: int = 1150,
        num_layers: int = 3,
        tie_weights: bool = True,
        pad_idx: int = 0,
        # AWD-LSTM dropout rates (Merity et al. Table 4)
        dropoute: float = 0.1,    # Embedding dropout
        dropouti: float = 0.3,    # Input variational dropout (after embedding)
        dropouth: float = 0.3,    # Hidden-to-hidden variational dropout
        dropouto: float = 0.4,    # Output variational dropout
        wdrop: float = 0.5,       # Weight dropout (DropConnect) on recurrent weights
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_weights = tie_weights
        
        # Store dropout rates
        self.dropoute = dropoute
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropouto = dropouto
        self.wdrop = wdrop
        
        # Variational dropout module (same mask across timesteps)
        self.lockdrop = LockedDropout()
        
        # Embedding (NO standard dropout — we use embedded_dropout instead)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        # Scaling factor (matches Transformer for fair comparison)
        self.scale = (embed_dim ** 0.5)
        
        # Build LSTM layers individually (needed for per-layer variational dropout)
        # Layer sizes follow Merity et al.: last layer output = embed_dim for tying
        layer_sizes = [embed_dim] + [hidden_size] * (num_layers - 1) + [embed_dim]
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            lstm_layer = nn.LSTM(
                input_size=layer_sizes[i],
                hidden_size=layer_sizes[i + 1],
                num_layers=1,
                batch_first=True,
            )
            # Wrap with WeightDrop (DropConnect on weight_hh)
            if wdrop > 0:
                lstm_layer = WeightDrop(lstm_layer, ['weight_hh_l0'], dropout=wdrop)
            self.rnns.append(lstm_layer)
        
        # Output projection (embed_dim → vocab_size)
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying: output_proj shares weights with embedding
        if tie_weights:
            self.output_proj.weight = self.embedding.weight
        
        # Initialise weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialise weights following Merity et al. (2018)"""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embedding.weight[self.embedding.padding_idx].zero_()
        
        for rnn_wrapper in self.rnns:
            # Get the actual LSTM module (may be wrapped in WeightDrop)
            lstm = rnn_wrapper.module if isinstance(rnn_wrapper, WeightDrop) else rnn_wrapper
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name or 'weight_hh_l0_raw' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # Forget gate bias = 1 (Jozefowicz et al., 2015)
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)
        
        if not self.tie_weights:
            nn.init.xavier_uniform_(self.output_proj.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        store_attention: bool = False,  # ignored, API compatibility
        return_hidden_for_reg: bool = False,  # return raw outputs for AR/TAR
    ):
        """
        Forward pass — predicts at EVERY position.
        
        Args:
            input_ids: (B, S) token indices
            hidden: Optional list of (h, c) tuples per layer
            return_hidden_for_reg: if True, also return raw LSTM outputs
                                   for AR/TAR regularization
            
        Returns:
            logits: (B, S, vocab_size)
            OR (logits, raw_outputs, dropped_outputs) if return_hidden_for_reg
        """
        # 1. Embedding dropout — drops entire word vectors
        emb = embedded_dropout(
            self.embedding, input_ids,
            dropout=self.dropoute if self.training else 0,
            scale=self.scale
        )
        # (B, S, embed_dim)
        
        # 2. Input variational dropout
        emb = self.lockdrop(emb, self.dropouti if self.training else 0)
        
        # 3. Pass through each LSTM layer with variational dropout between them
        raw_outputs = []      # For AR regularization
        dropped_outputs = []  # For TAR regularization
        
        current = emb
        new_hidden = []
        
        for i, rnn in enumerate(self.rnns):
            h_i = hidden[i] if hidden is not None else None
            
            if isinstance(rnn, WeightDrop):
                # WeightDrop applies DropConnect internally
                raw_out, new_h = rnn(current, h_i)
            else:
                raw_out, new_h = rnn(current, h_i)
            
            new_hidden.append(new_h)
            raw_outputs.append(raw_out)
            
            # Apply variational dropout between LSTM layers
            if i < self.num_layers - 1:
                # Hidden variational dropout (between layers)
                current = self.lockdrop(raw_out, self.dropouth if self.training else 0)
            else:
                # Output variational dropout (after last layer)
                current = self.lockdrop(raw_out, self.dropouto if self.training else 0)
            
            dropped_outputs.append(current)
        
        # 4. Output projection (last layer output is already embed_dim thanks to architecture)
        logits = self.output_proj(current)  # (B, S, vocab_size)
        
        if return_hidden_for_reg:
            return logits, raw_outputs, dropped_outputs
        return logits
    
    def generate_step(self, input_ids, hidden=None):
        """
        Single-step generation (for autoregressive decoding).
        No dropout during generation since model.eval() disables it.
        """
        emb = self.embedding(input_ids) * self.scale
        
        current = emb
        new_hidden = []
        
        for i, rnn in enumerate(self.rnns):
            h_i = hidden[i] if hidden is not None else None
            if isinstance(rnn, WeightDrop):
                out, new_h = rnn(current, h_i)
            else:
                out, new_h = rnn(current, h_i)
            new_hidden.append(new_h)
            current = out
        
        logits = self.output_proj(current)
        return logits, new_hidden
    
    def get_embedding_weights(self) -> torch.Tensor:
        """API compatibility with Transformer model"""
        return self.embedding.weight


class LSTMTextGenerator:
    """
    Text generation for the AWD-LSTM model.
    Same interface as TextGenerator but uses LSTM hidden states
    for efficient autoregressive generation.
    """
    
    def __init__(self, model: ShakespeareLSTM, vocab, device=config.DEVICE):
        self.model = model
        self.vocab = vocab
        self.device = device
    
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
        """Generate text continuation from seed."""
        self.model.eval()
        
        # Encode seed
        current_tokens = self.vocab.encode(seed_text)
        generated = list(current_tokens)
        
        with torch.no_grad():
            # Process seed through LSTM to build up hidden state
            seed_ids = torch.tensor([current_tokens], dtype=torch.long, device=self.device)
            _, hidden = self.model.generate_step(seed_ids)
            
            for _ in range(max_length):
                # Get prediction for last token
                last_token = torch.tensor([[generated[-1]]], dtype=torch.long, device=self.device)
                logits, hidden = self.model.generate_step(last_token, hidden)
                
                next_logits = logits[0, -1, :]  # (vocab_size,)
                
                # Repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(generated[-50:]):
                        if next_logits[token_id] > 0:
                            next_logits[token_id] /= repetition_penalty
                        else:
                            next_logits[token_id] *= repetition_penalty
                
                # Temperature
                next_logits = next_logits / temperature
                
                # Top-k
                if top_k > 0:
                    top_k_vals = min(top_k, next_logits.size(-1))
                    mask = next_logits < torch.topk(next_logits, top_k_vals)[0][..., -1, None]
                    next_logits[mask] = float('-inf')
                
                # Top-p
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cumprobs > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    indices_to_remove = remove.scatter(0, sorted_idx, remove)
                    next_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                if stop_on_eos and next_token == self.vocab.eos_idx:
                    break
                
                generated.append(next_token)
        
        return self.vocab.decode_to_text(generated)


def create_lstm_model(
    vocab_size: int,
    device: torch.device = config.DEVICE
) -> ShakespeareLSTM:
    """
    Factory function — creates AWD-LSTM model.
    
    Uses EMBEDDING_DIM from config (300) to match Transformer.
    3-layer LSTM with 1150 hidden units (Merity et al. architecture).
    
    Dropout rates tuned for our data scale (~1.1M tokens):
        - Reduced from paper values since we have less data
        - Paper used PTB (1M tokens) but with 24M params
        - We use ~10M params, so somewhat less aggressive dropout
    """
    model = ShakespeareLSTM(
        vocab_size=vocab_size,
        embed_dim=config.EMBEDDING_DIM,    # 300 (same as Transformer)
        hidden_size=1150,                   # Merity et al. default
        num_layers=3,                       # Merity et al. default
        tie_weights=True,
        pad_idx=0,
        # Dropout rates — slightly reduced from paper for our scale
        dropoute=0.1,     # Embedding dropout (paper: 0.1)
        dropouti=0.3,     # Input variational dropout (paper: 0.65 — we reduce)
        dropouth=0.25,    # Hidden variational dropout (paper: 0.3)
        dropouto=0.4,     # Output variational dropout (paper: 0.4)
        wdrop=0.5,        # Weight dropout/DropConnect (paper: 0.5) — CRITICAL
    )
    return model.to(device)


# ============================================================================
# For testing
# ============================================================================
if __name__ == "__main__":
    print("Testing AWD-LSTM (ShakespeareLSTM)")
    print("=" * 60)
    
    vocab_size = 5000
    batch_size = 4
    seq_length = 128
    
    model = create_lstm_model(vocab_size)
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    
    # Forward pass
    x = torch.randint(0, vocab_size, (batch_size, seq_length)).to(config.DEVICE)
    logits = model(x)
    print(f"\nInput:  {x.shape}")
    print(f"Output: {logits.shape}")
    
    # Forward with AR/TAR outputs
    logits, raw_outs, drop_outs = model(x, return_hidden_for_reg=True)
    print(f"Raw outputs: {len(raw_outs)} layers, shape {raw_outs[-1].shape}")
    
    # Verify weight tying
    assert model.output_proj.weight is model.embedding.weight, "Weight tying failed!"
    print("Weight tying: OK")
    
    # Test generation step
    model.eval()
    gen_input = torch.randint(0, vocab_size, (1, 10)).to(config.DEVICE)
    gen_logits, gen_hidden = model.generate_step(gen_input)
    print(f"\nGeneration step: input {gen_input.shape} → logits {gen_logits.shape}")
    print(f"Hidden states: {len(gen_hidden)} layers")
