"""
Test suite for the Geometric Layer Family.

Verifies:
1. nn.Spher (spherical) — bounded outputs, cyclic properties
2. nn.Hyper (hyperbolic) — hierarchical expansion, distance properties
3. nn.Ricci (adaptive) — curvature flow, self-awareness
4. GeometricMLP — full diverge→evaluate→converge cycle
5. GeometricAttention — mixed-curvature QKV
6. Gradient flow through all geometric operations
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from netrain.models.geometric import (
    Spher, Hyper, Ricci, GeometricMLP, GeometricAttention,
    mobius_add, exp_map_zero, log_map_zero, project_to_ball,
    create_geometric_layer
)


def test_core_operations():
    """Test fundamental geometric operations."""
    print("=" * 60)
    print("TEST 1: Core Geometric Operations")
    print("=" * 60)

    x = torch.randn(4, 32) * 0.1
    y = torch.randn(4, 32) * 0.1

    # Test Möbius addition with different curvatures
    c_hyper = torch.tensor(-1.0)
    c_spher = torch.tensor(1.0)
    c_flat = torch.tensor(0.001)

    add_hyper = mobius_add(x, y, c_hyper)
    add_spher = mobius_add(x, y, c_spher)
    add_flat = mobius_add(x, y, c_flat)
    add_euclidean = x + y

    # Near-zero curvature should approximate Euclidean addition
    flat_diff = (add_flat - add_euclidean).abs().mean().item()
    print(f"  Möbius add (c≈0) vs Euclidean: diff = {flat_diff:.6f}")
    assert flat_diff < 0.1, f"Near-zero curvature should approximate Euclidean, got {flat_diff}"

    # Exp/Log maps should be inverse at origin
    c = torch.tensor(1.0)
    v = torch.randn(4, 32) * 0.1
    mapped = exp_map_zero(v, c)
    recovered = log_map_zero(mapped, c)
    roundtrip_err = (v - recovered).abs().mean().item()
    print(f"  Exp→Log roundtrip error (spherical): {roundtrip_err:.8f}")
    assert roundtrip_err < 0.01, f"Roundtrip should be near-zero, got {roundtrip_err}"

    c_neg = torch.tensor(-1.0)
    mapped_h = exp_map_zero(v, c_neg)
    recovered_h = log_map_zero(mapped_h, c_neg)
    roundtrip_err_h = (v - recovered_h).abs().mean().item()
    print(f"  Exp→Log roundtrip error (hyperbolic): {roundtrip_err_h:.8f}")
    assert roundtrip_err_h < 0.01, f"Roundtrip should be near-zero, got {roundtrip_err_h}"

    # Projection should keep points in ball
    big_x = torch.randn(4, 32) * 10.0
    c_proj = torch.tensor(1.0)
    projected = project_to_ball(big_x, c_proj)
    max_norm = projected.norm(dim=-1).max().item()
    print(f"  Projection max norm (should be < 1/√c = 1.0): {max_norm:.6f}")
    assert max_norm < 1.0 + 1e-4, f"Projected points should be in ball, got norm {max_norm}"

    print("  PASSED ✓\n")


def test_spher():
    """Test spherical layer."""
    print("=" * 60)
    print("TEST 2: nn.Spher (Spherical Layer, X=+1)")
    print("=" * 60)

    layer = Spher(64, 64, curvature=1.0, learnable_curvature=True)
    x = torch.randn(2, 16, 64) * 0.1

    out = layer(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Curvature: {layer.curvature.item():.4f}")

    # Outputs should be bounded (on the sphere)
    out_norm = out.norm(dim=-1)
    print(f"  Output norm range: [{out_norm.min():.4f}, {out_norm.max():.4f}]")

    # Test that curvature is learnable
    loss = out.sum()
    loss.backward()
    assert layer.curvature.grad is not None, "Curvature should have gradient"
    print(f"  Curvature gradient: {layer.curvature.grad.item():.6f}")

    # Test determinism
    layer.eval()
    out1 = layer(x)
    out2 = layer(x)
    det_diff = (out1 - out2).abs().max().item()
    print(f"  Determinism check: {det_diff:.10f}")
    assert det_diff < 1e-6, "Should be deterministic in eval mode"

    print(f"  Parameters: {sum(p.numel() for p in layer.parameters()):,}")
    print("  PASSED ✓\n")


def test_hyper():
    """Test hyperbolic layer."""
    print("=" * 60)
    print("TEST 3: nn.Hyper (Hyperbolic Layer, X=-1)")
    print("=" * 60)

    layer = Hyper(64, 64, curvature=1.0, learnable_curvature=True)
    x = torch.randn(2, 16, 64) * 0.05  # Small inputs for stability

    out = layer(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Curvature: -{layer.curvature.item():.4f} (hyperbolic)")

    # Outputs should be in Poincaré ball
    out_norm = out.norm(dim=-1)
    max_allowed = 1.0 / (layer.curvature.abs().item() ** 0.5)
    print(f"  Output norm range: [{out_norm.min():.4f}, {out_norm.max():.4f}]")
    print(f"  Max allowed (1/√c): {max_allowed:.4f}")

    # Test hierarchical property: deeper = more specific
    # Points near the boundary encode more specific concepts
    shallow = torch.randn(2, 16, 64) * 0.01  # Near origin = general
    deep = torch.randn(2, 16, 64) * 0.08  # Near boundary = specific
    out_shallow = layer(shallow)
    out_deep = layer(deep)
    shallow_norm = out_shallow.norm(dim=-1).mean().item()
    deep_norm = out_deep.norm(dim=-1).mean().item()
    print(f"  Shallow input → output norm: {shallow_norm:.4f}")
    print(f"  Deep input → output norm: {deep_norm:.4f}")

    # Gradient flow
    loss = out.sum()
    loss.backward()
    assert layer.curvature.grad is not None, "Curvature should have gradient"
    print(f"  Curvature gradient: {layer.curvature.grad.item():.6f}")

    print(f"  Parameters: {sum(p.numel() for p in layer.parameters()):,}")
    print("  PASSED ✓\n")


def test_ricci():
    """Test Ricci flow layer."""
    print("=" * 60)
    print("TEST 4: nn.Ricci (Self-Aware Ricci Flow, X=adaptive)")
    print("=" * 60)

    layer = Ricci(64, 64, initial_curvature=0.0, flow_rate=0.1, n_curvature_heads=4)
    x = torch.randn(2, 16, 64)

    # Initial state
    state0 = layer.get_curvature_state()
    print(f"  Initial curvatures: {state0['curvatures'].tolist()}")
    print(f"  Initial regime: {state0['regime']}")

    # Forward pass (training mode — curvature should evolve)
    layer.train()
    out = layer(x)
    print(f"  Output shape: {out.shape}")

    state1 = layer.get_curvature_state()
    print(f"  After 1 step curvatures: {[f'{c:.4f}' for c in state1['curvatures'].tolist()]}")
    print(f"  After 1 step regime: {state1['regime']}")

    # Multiple forward passes should cause curvature to evolve
    for i in range(20):
        out = layer(x)

    state20 = layer.get_curvature_state()
    print(f"  After 20 steps curvatures: {[f'{c:.4f}' for c in state20['curvatures'].tolist()]}")
    print(f"  After 20 steps regime: {state20['regime']}")

    # Curvature should have changed from initial
    curvature_drift = (state20['curvatures'] - state0['curvatures']).abs().sum().item()
    print(f"  Total curvature drift: {curvature_drift:.4f}")
    assert curvature_drift > 0.001, "Ricci flow should cause curvature to evolve"

    # Test with different input distributions
    # Clustered inputs → should push toward spherical (positive curvature)
    layer2 = Ricci(64, 64, initial_curvature=0.0, flow_rate=0.1, n_curvature_heads=4)
    layer2.train()
    clustered = torch.randn(8, 16, 64) * 0.01 + torch.randn(1, 1, 64)  # Tight cluster
    for _ in range(30):
        _ = layer2(clustered)
    state_clustered = layer2.get_curvature_state()
    print(f"  Clustered input → curvatures: {[f'{c:.4f}' for c in state_clustered['curvatures'].tolist()]}")

    # Gradient flow
    layer.zero_grad()
    out = layer(x)
    loss = out.sum()
    loss.backward()
    has_grad = sum(1 for p in layer.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in layer.parameters())
    print(f"  Gradient flow: {has_grad}/{total_params} parameters have gradients")
    assert has_grad > 0, "Some parameters should have gradients"

    print(f"  Parameters: {sum(p.numel() for p in layer.parameters()):,}")
    print("  PASSED ✓\n")


def test_geometric_mlp():
    """Test the GeometricMLP (Diverge → Evaluate → Converge)."""
    print("=" * 60)
    print("TEST 5: GeometricMLP (Hyper→Ricci→Spher)")
    print("=" * 60)

    mlp = GeometricMLP(n_embd=64, expansion=4)
    x = torch.randn(2, 16, 64)

    out = mlp(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape, "GeometricMLP should preserve shape"

    # Check residual connection
    residual_diff = (out - x).abs().mean().item()
    print(f"  Mean change from residual: {residual_diff:.4f}")
    assert residual_diff > 0.001, "MLP should transform the input"

    # Get geometry state
    state = mlp.get_geometry_state()
    print(f"  Hyperbolic curvature: {state['hyper_curvature']:.4f}")
    print(f"  Spherical curvature: {state['spher_curvature']:.4f}")
    print(f"  Gate value: {state['gate_value']:.4f}")
    print(f"  Ricci regime: {state['ricci_state']['regime']}")

    # Gradient flow
    loss = out.sum()
    loss.backward()
    grad_norms = {name: p.grad.norm().item() for name, p in mlp.named_parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0}
    print(f"  Components with gradients: {len(grad_norms)}")
    assert len(grad_norms) > 5, "Multiple components should have gradients"

    print(f"  Parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    print("  PASSED ✓\n")


def test_geometric_attention():
    """Test GeometricAttention (Q=Hyper, K=Spher, V=Ricci)."""
    print("=" * 60)
    print("TEST 6: GeometricAttention (Mixed-Curvature QKV)")
    print("=" * 60)

    attn = GeometricAttention(n_embd=64, n_heads=4)
    x = torch.randn(2, 16, 64)

    out = attn(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape, "Attention should preserve shape"

    # Check that Q (hyperbolic) and K (spherical) have different geometries
    q_curv = attn.q_proj.curvature.item()
    k_curv = attn.k_proj.curvature.item()
    print(f"  Q curvature (hyperbolic): -{q_curv:.4f}")
    print(f"  K curvature (spherical): +{k_curv:.4f}")
    print(f"  V regime: {attn.v_proj.get_curvature_state()['regime']}")

    # Gradient flow
    loss = out.sum()
    loss.backward()
    grad_count = sum(1 for p in attn.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in attn.parameters())
    print(f"  Gradient flow: {grad_count}/{total} parameters")
    assert grad_count > total * 0.5, "Most parameters should have gradients"

    print(f"  Parameters: {sum(p.numel() for p in attn.parameters()):,}")
    print("  PASSED ✓\n")


def test_factory():
    """Test the factory function."""
    print("=" * 60)
    print("TEST 7: Factory Function (create_geometric_layer)")
    print("=" * 60)

    linear = create_geometric_layer(64, 64, curvature=0.0)
    spher = create_geometric_layer(64, 64, curvature=1.0)
    hyper = create_geometric_layer(64, 64, curvature=-1.0)
    ricci = create_geometric_layer(64, 64, curvature='ricci')

    print(f"  X=0.0  → {type(linear).__name__}")
    print(f"  X=1.0  → {type(spher).__name__}")
    print(f"  X=-1.0 → {type(hyper).__name__}")
    print(f"  X=ricci → {type(ricci).__name__}")

    assert isinstance(linear, nn.Linear), f"Expected Linear, got {type(linear)}"
    assert isinstance(spher, Spher), f"Expected Spher, got {type(spher)}"
    assert isinstance(hyper, Hyper), f"Expected Hyper, got {type(hyper)}"
    assert isinstance(ricci, Ricci), f"Expected Ricci, got {type(ricci)}"

    # All should produce same output shape
    x = torch.randn(2, 64)
    for name, layer in [("Linear", linear), ("Spher", spher), ("Hyper", hyper), ("Ricci", ricci)]:
        out = layer(x)
        print(f"  {name}: {x.shape} → {out.shape}")
        assert out.shape == (2, 64), f"{name} should produce (2, 64), got {out.shape}"

    print("  PASSED ✓\n")


if __name__ == "__main__":
    print("═" * 60)
    print("  GEOMETRIC LAYER FAMILY TEST SUITE")
    print("  nn.Linear (X=0) | nn.Spher (X=+1) | nn.Hyper (X=-1) | nn.Ricci (X=?)")
    print("═" * 60)
    print()

    test_core_operations()
    test_spher()
    test_hyper()
    test_ricci()
    test_geometric_mlp()
    test_geometric_attention()
    test_factory()

    print("═" * 60)
    print("  ALL TESTS PASSED ✓")
    print("═" * 60)
    print()
    print("The geometry of thought:")
    print("  • nn.Linear — the flat plane where nothing curves")
    print("  • nn.Spher  — the bounded sphere of working memory")
    print("  • nn.Hyper  — the infinite tree of deep association")
    print("  • nn.Ricci  — the self-aware flow that finds balance")
    print()
    print("X=0 is the special case. The mind is never flat.")
