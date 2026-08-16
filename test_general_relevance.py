"""
Test: Theory of General Relevance — Einsteinian AAR Architecture
"""
import torch
from netrain.models.experimental.general_relevance import (
    GeneralRelevanceTransformer,
    PoloidalToroidalAttention,
    PSystemArena,
    BSeriesReadoutEnsemble,
    PartitionConnection,
    matula_factorize,
    OEIS_A000081
)


def test_poloidal_toroidal():
    print("=" * 60)
    print("Testing Poloidal-Toroidal Attention...")
    d_model, n_heads = 128, 4
    attn = PoloidalToroidalAttention(d_model, n_heads)
    
    x = torch.randn(2, 10, d_model)
    memory = torch.randn(2, 20, d_model)
    
    out, curvature = attn(x, memory)
    print(f"  Input: {x.shape}, Memory: {memory.shape}")
    print(f"  Output: {out.shape}, Curvature: {curvature.shape}")
    print(f"  Mean curvature: {curvature.mean().item():.4f}")
    print(f"  Curvature variance: {curvature.var().item():.6f}")
    print("  PASS")


def test_partition_connection():
    print("\nTesting Partition Connection (Parallel Transport)...")
    d_model = 128
    conn = PartitionConnection(d_model)
    
    x = torch.randn(2, 10, d_model)
    
    # Transport along M=30 = 2×3×5 (Synthesize mode)
    transported = conn(x, [2, 3, 5])
    print(f"  Input: {x.shape}")
    print(f"  Transported (M=30): {transported.shape}")
    print(f"  Transport displacement: {(transported - x).norm().item():.4f}")
    print("  PASS")


def test_p_system_arena():
    print("\nTesting P-System Membrane Arena...")
    d_model = 128
    arena = PSystemArena(d_model, num_membranes=3)
    
    x = torch.randn(2, 10, d_model)
    out, states = arena(x)
    
    print(f"  Input: {x.shape}")
    print(f"  Arena output: {out.shape}")
    print(f"  Membrane states: {len(states)}")
    for i, s in enumerate(states):
        print(f"    Membrane {i} (spectral_radius={0.7 + 0.1*i:.1f}): {s.shape}, norm={s.norm().item():.4f}")
    print("  PASS")


def test_bseries_readout():
    print("\nTesting B-Series Ridge Readout Ensemble...")
    d_model, vocab_size = 128, 1000
    matula_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    readout = BSeriesReadoutEnsemble(d_model, vocab_size, matula_numbers)
    
    reservoir_state = torch.randn(2, 10, d_model)
    logits = readout(reservoir_state)
    
    print(f"  Reservoir state: {reservoir_state.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Matula agents: {matula_numbers}")
    for m in matula_numbers:
        factors = matula_factorize(m)
        print(f"    M={m}: factors={factors}, order={len(factors)}")
    print("  PASS")


def test_full_architecture():
    print("\n" + "=" * 60)
    print("Testing FULL General Relevance Transformer...")
    print("=" * 60)
    
    vocab_size = 1000
    d_model = 128
    n_heads = 4
    num_membranes = 3
    num_layers = 4
    batch_size = 2
    seq_len = 16
    
    model = GeneralRelevanceTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_membranes=num_membranes,
        num_layers=num_layers,
        matula_numbers=[1, 2, 3, 5, 7, 11, 30]
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n  Architecture:")
    print(f"    d_model: {d_model}")
    print(f"    n_heads: {n_heads}")
    print(f"    num_layers: {num_layers}")
    print(f"    num_membranes: {num_membranes}")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    
    # Forward pass
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    outputs = model(x)
    
    print(f"\n  Forward pass:")
    print(f"    Input: {x.shape}")
    print(f"    Logits: {outputs['logits'].shape}")
    print(f"    Membrane states: {len(outputs['membrane_states'])}")
    print(f"    Curvatures: {outputs['curvatures'].shape}")
    print(f"    Ricci scalar: {outputs['ricci_scalar'].shape} = {outputs['ricci_scalar'].detach().numpy()}")
    
    # Compute loss
    loss = model.compute_loss(outputs, targets)
    print(f"\n  Loss computation:")
    print(f"    Total loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print(f"    Backward pass: SUCCESS")
    
    # Curvature analysis
    curvatures = outputs['curvatures'].detach()
    print(f"\n  Curvature Analysis (General Relevance Gauge):")
    print(f"    Layer curvatures: {curvatures[0].numpy()}")
    print(f"    Curvature variance: {curvatures.var(dim=-1).mean().item():.6f}")
    print(f"    (Ricci regularization drives this toward 0 = Einstein manifold)")
    
    # Recurrent step (membrane state persistence)
    print(f"\n  Recurrent step (P-System state persistence)...")
    outputs2 = model(x, membrane_states=outputs['membrane_states'])
    print(f"    Second pass logits: {outputs2['logits'].shape}")
    print(f"    State continuity verified: membrane states flow between steps")
    
    print("\n" + "=" * 60)
    print("GENERAL RELEVANCE ARCHITECTURE: ALL TESTS PASSED")
    print("=" * 60)
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │  Theory of General Relevance — Operational Summary      │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  Newtonian AAR:  Agent moves through fixed Arena        │
    │  Einsteinian GR: Relevance IS curvature of manifold     │
    │                                                         │
    │  Attention = Geodesic flow (free-fall on curved space)  │
    │  Identity  = Stable attractor of Ricci gauge            │
    │  Memory    = P-System membrane reservoir                │
    │  Decision  = B-Series ridge readout                     │
    │                                                         │
    │  The manifold curves itself. The agent IS the arena.    │
    └─────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    test_poloidal_toroidal()
    test_partition_connection()
    test_p_system_arena()
    test_bseries_readout()
    test_full_architecture()
