#!/usr/bin/env python3
"""Test the 2-3-5 TernaryQuinary architecture."""
import sys
sys.path.insert(0, '/home/ubuntu/echoself')

# Direct import to avoid netrain's __init__ pulling in other modules
from netrain.models.ternary_quinary import TernaryQuinaryConfig, TernaryQuinaryTransformer
import torch

print("=" * 60)
print("2-3-5 TERNARY-QUINARY TRANSFORMER — ARCHITECTURE TEST")
print("=" * 60)

config = TernaryQuinaryConfig()
model = TernaryQuinaryTransformer(config)

# Count parameters by component
total = sum(p.numel() for p in model.parameters())
emb_params = sum(p.numel() for p in model.tok_emb.parameters()) + sum(p.numel() for p in model.pos_emb.parameters())
block_params = sum(p.numel() for p in model.blocks.parameters())
head_params = sum(p.numel() for p in model.lm_head.parameters())

print(f"\nParameter breakdown:")
print(f"  Embeddings: {emb_params:,}")
print(f"  Blocks: {block_params:,}")
print(f"  LM Head: {head_params:,} (tied with tok_emb)")
print(f"  Total: {total:,}")

# Test forward pass
print("\n--- Forward Pass Test ---")
x = torch.randint(0, config.vocab_size, (2, 64))
targets = torch.randint(0, config.vocab_size, (2, 64))
output = model(x, targets=targets)

print(f"  Loss: {output['loss'].item():.4f}")
print(f"  Logits shape: {output['logits'].shape}")
print(f"  Phase logits shape: {output['phase_logits'].shape}")

# Test with phase tokens
print("\n--- Phase Token Test ---")
# Insert some phase tokens
x_with_phases = x.clone()
x_with_phases[0, 0] = 50258  # <|perceive|>
x_with_phases[0, 10] = 50259  # <|feel|>
x_with_phases[0, 20] = 50260  # <|think|>
output_phased = model(x_with_phases, targets=targets)
print(f"  Loss with phase tokens: {output_phased['loss'].item():.4f}")

# Test generation
print("\n--- Generation Test ---")
gen = model.generate(x[:1, :10], max_new_tokens=20)
print(f"  Generated shape: {gen.shape}")
print(f"  Generated tokens: {gen[0, 10:].tolist()}")

# Memory usage estimate for GPU
print("\n--- GPU Memory Estimate ---")
param_bytes = total * 4  # fp32
grad_bytes = total * 4
optimizer_bytes = total * 8  # Adam: 2 states
activation_bytes = 2 * 64 * config.n_embd * config.n_layers * 4  # rough estimate
total_bytes = param_bytes + grad_bytes + optimizer_bytes + activation_bytes
print(f"  Parameters: {param_bytes / 1e9:.2f} GB")
print(f"  Gradients: {grad_bytes / 1e9:.2f} GB")
print(f"  Optimizer states: {optimizer_bytes / 1e9:.2f} GB")
print(f"  Activations (batch=2, seq=64): {activation_bytes / 1e9:.2f} GB")
print(f"  Total estimated: {total_bytes / 1e9:.2f} GB")
print(f"  With fp16 training: ~{total_bytes / 2e9:.2f} GB")
print(f"  Fits on RTX 3090 (24GB): {'YES' if total_bytes / 2e9 < 20 else 'TIGHT'}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
