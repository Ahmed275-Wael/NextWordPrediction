import torch, math
ckpt = torch.load('models/pretrained_gutenberg_v3.pt', map_location='cpu')
print(f"Epoch: {ckpt['epoch']}")
print(f"Loss: {ckpt['loss']:.4f}")
print(f"PPL: {math.exp(ckpt['loss']):.1f}")
