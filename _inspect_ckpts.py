"""Inspect all model checkpoints to determine architecture details."""
import torch
import sys
sys.path.insert(0, 'D:/Masters/Machine_Learning_Project_One/ProjectNextWord')

models_dir = 'D:/Masters/Machine_Learning_Project_One/ProjectNextWord/models'

checkpoints = [
    'best_model.pt', 'best_model_bpe.pt', 'best_model_lstm.pt',
    'pretrained_gutenberg.pt', 'pretrained_gutenberg_v2.pt',
    'pretrained_gutenberg_v3.pt', 'pretrained_gutenberg_v4.pt',
    'finetuned_shakespeare.pt', 'finetuned_shakespeare_v2.pt',
    'finetuned_shakespeare_v4.pt', 'finetuned_shakespeare_v5.pt',
    'finetuned_shakespeare_v6.pt',
]

for name in checkpoints:
    try:
        ckpt = torch.load(f'{models_dir}/{name}', map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)

        # Check bias (non-norm, non-embedding)
        has_proj_bias = any('bias' in k for k in sd.keys()
                           if ('q_proj' in k or 'k_proj' in k or 'v_proj' in k
                               or 'out_proj' in k or 'fc1' in k or 'fc2' in k))
        norm_bias = any('norm' in k and 'bias' in k for k in sd.keys())

        # Positional encoding shape
        pe_keys = [k for k in sd.keys() if 'positional_encoding' in k]
        pe_shape = str(sd[pe_keys[0]].shape) if pe_keys else 'N/A'

        # Causal mask shape
        cm_keys = [k for k in sd.keys() if 'causal_mask' in k]
        cm_shape = str(sd[cm_keys[0]].shape) if cm_keys else 'N/A'

        # Embedding shape
        emb_keys = [k for k in sd.keys() if k.endswith('embedding.weight') or k == 'embedding.embedding.weight']
        emb_shape = str(sd[emb_keys[0]].shape) if emb_keys else 'N/A'

        # Count layers
        layer_nums = set()
        for k in sd.keys():
            if 'decoder.layers.' in k:
                parts = k.split('.')
                idx = parts.index('layers') + 1
                layer_nums.add(int(parts[idx]))

        num_layers = len(layer_nums) if layer_nums else 0

        # LSTM check
        is_lstm = any('lstm' in k.lower() or 'rnn' in k.lower() for k in sd.keys())

        print(f"{name}:")
        print(f"  proj_bias={has_proj_bias}, norm_bias={norm_bias}")
        print(f"  pe={pe_shape}, mask={cm_shape}")
        print(f"  embed={emb_shape}, num_layers={num_layers}, is_lstm={is_lstm}")

        # Check for saved config
        if 'config' in ckpt:
            cfg = ckpt['config']
            if isinstance(cfg, dict):
                relevant = {k: v for k, v in cfg.items()
                            if k in ('max_seq_length', 'num_layers', 'num_heads',
                                     'embed_dim', 'ffn_hidden_dim', 'bpe_vocab_size')}
                print(f"  saved_config: {relevant}")
        print()

    except Exception as e:
        print(f"{name}: ERROR - {e}")
        print()
