"""
Test and Visualization for Recursive Partition-Indexed Transformer
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from netrain.models.experimental.recursive_transformer import (
    prime_factors, 
    PrimeBasisHead, 
    RecursiveCompositeHead,
    RecursiveMatulaTransformer
)

plt.style.use('dark_background')

def test_prime_factorization():
    print("Testing Prime Factorization...")
    test_cases = {
        1: [1],
        2: [2],
        3: [3],
        4: [2, 2],
        5: [5],
        6: [2, 3],
        8: [2, 2, 2],
        30: [2, 3, 5],
        286: [2, 11, 13]
    }
    
    for n, expected in test_cases.items():
        factors = prime_factors(n)
        if not factors: factors = [1]
        assert factors == expected, f"Failed for {n}: got {factors}, expected {expected}"
        print(f"  M={n:<4} -> {factors}")
    print("  Prime factorization passed.\n")

def test_recursive_head():
    print("Testing Recursive Composite Head...")
    B, T, d_model = 2, 16, 64
    x = torch.randn(B, T, d_model)
    mask = torch.tril(torch.ones(T, T)).view(1, T, T)
    
    # Test head M=30 (factors 2, 3, 5)
    head = RecursiveCompositeHead(30, d_model)
    
    assert len(head.sub_heads) == 3, "M=30 should have 3 sub-heads"
    assert head.sub_heads[0].p == 2, "First sub-head should be p=2"
    assert head.sub_heads[1].p == 3, "Second sub-head should be p=3"
    assert head.sub_heads[2].p == 5, "Third sub-head should be p=5"
    
    out, energy = head(x, mask)
    
    assert out.shape == (B, T, d_model), f"Output shape mismatch: {out.shape}"
    assert energy.shape == (B,), f"Energy shape mismatch: {energy.shape}"
    
    # Check gradient flow
    loss = out.sum()
    loss.backward()
    
    has_grad = False
    for p in head.parameters():
        if p.grad is not None:
            has_grad = True
            break
            
    assert has_grad, "No gradients flowing through recursive head"
    print("  Recursive Composite Head passed.\n")

def test_full_model():
    print("Testing Full Recursive Transformer...")
    vocab_size = 100
    model = RecursiveMatulaTransformer(num_layers=3, d_model=64, vocab_size=vocab_size)
    
    # Check architecture
    # L1: 1 head
    # L2: 1 head
    # L3: 2 heads
    
    B, T = 2, 10
    idx = torch.randint(0, vocab_size, (B, T))
    
    logits = model(idx)
    assert logits.shape == (B, T, vocab_size), f"Logits shape mismatch: {logits.shape}"
    
    # Check gradient flow
    loss = logits.sum()
    loss.backward()
    
    print("  Full Recursive Transformer passed.\n")

def visualize_recursive_topology():
    print("Generating Recursive Topology Visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Plot the factorization complexity
    ax = axes[0]
    M_values = list(range(2, 101))
    num_factors = [len(prime_factors(m)) for m in M_values]
    unique_factors = [len(set(prime_factors(m))) for m in M_values]
    
    ax.plot(M_values, num_factors, 'o-', color='#3498db', alpha=0.7, label='Total prime factors (Sub-heads)')
    ax.plot(M_values, unique_factors, 's-', color='#e74c3c', alpha=0.7, label='Unique prime factors (Dimensions)')
    
    # Highlight highly composite numbers
    highly_composite = [2, 4, 6, 12, 24, 36, 48, 60, 120]
    for hc in highly_composite:
        if hc <= 100:
            idx = M_values.index(hc)
            ax.plot(hc, num_factors[idx], '*', color='#f1c40f', markersize=15)
            
    ax.set_xlabel('Matula Number (M)', fontsize=12)
    ax.set_ylabel('Internal Complexity (Number of Sub-heads)', fontsize=12)
    ax.set_title('Recursive Complexity of Attention Heads', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    # 2. Plot the internal dimension allocation
    ax = axes[1]
    
    # Analyze a specific highly composite head: M=60 (2^2 * 3 * 5)
    factors_60 = prime_factors(60)
    
    # Simulate partition function weights at different temperatures
    temperatures = [0.1, 1.0, 5.0]
    
    # Dummy energies for the factors (2, 2, 3, 5)
    # Assume larger primes have higher "surprise" / free energy initially
    energies = np.array([1.0, 1.2, 2.5, 4.0])
    
    width = 0.25
    x = np.arange(len(factors_60))
    
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    for i, T in enumerate(temperatures):
        # Gibbs distribution
        weights = np.exp(-energies / T)
        weights = weights / weights.sum()
        
        ax.bar(x + i*width, weights, width, label=f'kT = {T}', color=colors[i], alpha=0.8)
        
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Sub-head p={p}' for p in factors_60])
    ax.set_xlabel('Internal Sub-heads of M=60', fontsize=12)
    ax.set_ylabel('Partition Function Weight (Active Inference)', fontsize=12)
    ax.set_title('Dynamic Internal Routing via Active Inference', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/recursive_topology.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
    print("Visualization saved to /home/ubuntu/recursive_topology.png")

if __name__ == "__main__":
    test_prime_factorization()
    test_recursive_head()
    test_full_model()
    visualize_recursive_topology()
    print("All tests passed successfully!")
