"""
Recursive Partition-Indexed Transformer
=======================================
Implementation of the recursive architecture where each attention head
IS a nested transformer, parameterized by the partition function over
its Matula prime factorization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

# ============================================================================
# Matula Number Utilities
# ============================================================================

def get_primes(n: int) -> List[int]:
    """Generate primes up to n."""
    sieve = [True] * (n + 1)
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            for i in range(p * p, n + 1, p):
                sieve[i] = False
    return [p for p in range(2, n + 1) if sieve[p]]

# Precompute primes
PRIMES = get_primes(10000)

def prime_factors(n: int) -> List[int]:
    """Return the prime factors of n (with multiplicity)."""
    if n <= 1:
        return []
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
        if d * d > n:
            if n > 1:
                factors.append(n)
            break
    return factors

# OEIS A000081 (rooted trees with n nodes)
# Used to determine how many heads (Matula numbers) exist at each layer
A000081 = [0, 1, 1, 2, 4, 9, 20, 48, 115, 286, 719, 1842, 4766]

# ============================================================================
# Recursive Attention Architecture
# ============================================================================

class PrimeBasisHead(nn.Module):
    """
    The grounding of the recursion. When M is prime (or 1), 
    it is implemented as a standard attention head of dimension M.
    """
    def __init__(self, p: int, d_model: int):
        super().__init__()
        self.p = max(1, p)  # Dimension of this basis head
        self.d_model = d_model
        
        # Projections
        self.q_proj = nn.Linear(d_model, self.p, bias=False)
        self.k_proj = nn.Linear(d_model, self.p, bias=False)
        self.v_proj = nn.Linear(d_model, self.p, bias=False)
        self.o_proj = nn.Linear(self.p, d_model, bias=False)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        
        q = self.q_proj(x)  # (B, T, p)
        k = self.k_proj(x)  # (B, T, p)
        v = self.v_proj(x)  # (B, T, p)
        
        # Attention scores (variational free energy / surprise)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.p)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        
        # Output
        out = torch.matmul(attn, v)  # (B, T, p)
        out = self.o_proj(out)       # (B, T, d_model)
        
        # Compute free energy of this prime mode (for the partition function)
        # Energy = negative log likelihood of the attention distribution
        # We approximate this by the entropy of the attention weights
        energy = -(attn * torch.log(attn + 1e-9)).sum(dim=-1).mean(dim=-1) # (B,)
        
        return out, energy

class RecursiveCompositeHead(nn.Module):
    """
    A composite head parameterized by the partition function over its prime factors.
    This head IS a nested transformer.
    """
    def __init__(self, M: int, d_model: int):
        super().__init__()
        self.M = M
        self.d_model = d_model
        
        # 1. Factor M into primes
        self.factors = prime_factors(M)
        if not self.factors:
            self.factors = [1]  # Fallback for M=1
            
        # 2. Create sub-heads for each prime factor
        # This is the recursive step (grounded at the primes)
        self.sub_heads = nn.ModuleList([
            PrimeBasisHead(p, d_model) for p in self.factors
        ])
        
        # 3. The partition function temperature (learnable, but modulated by ESN in full model)
        self.beta = nn.Parameter(torch.ones(len(self.factors)))
        
        # 4. Polytope vertex figure projection (combines the sub-head outputs)
        # Instead of simple addition, we use a learned projection from the concatenated space
        total_p = sum(self.factors)
        self.vertex_proj = nn.Linear(d_model * len(self.factors), d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        
        # Run all sub-heads (the internal nested transformer)
        sub_outs = []
        energies = []
        
        for sub_head in self.sub_heads:
            out, energy = sub_head(x, mask)
            sub_outs.append(out)
            energies.append(energy)
            
        # Stack energies: (B, num_factors)
        E = torch.stack(energies, dim=-1)
        
        # Quantize attention via the partition function
        # Z = sum_p exp(-beta_p * E_p)
        # We use softmax to get the normalized probabilities (the Gibbs measure)
        # shape: (B, num_factors)
        gibbs_weights = F.softmax(-self.beta * E, dim=-1)
        
        # Apply the partition weights to the sub-head outputs
        # Reshape for broadcasting: (B, 1, 1, num_factors)
        w = gibbs_weights.view(B, 1, 1, len(self.factors))
        
        # Stack outputs: (B, T, C, num_factors)
        stacked_outs = torch.stack(sub_outs, dim=-1)
        
        # Weighted sum according to the partition function
        # This is the active inference update!
        weighted_out = (stacked_outs * w).sum(dim=-1) # (B, T, C)
        
        # Alternatively, we can use the vertex projection for a richer topological blend
        # concat_outs = torch.cat(sub_outs, dim=-1) # (B, T, C * num_factors)
        # projected_out = self.vertex_proj(concat_outs)
        
        # Total free energy of this composite head
        total_energy = (gibbs_weights * E).sum(dim=-1)
        
        return weighted_out, total_energy

class RecursiveAttentionLayer(nn.Module):
    """
    A full transformer layer where heads are instantiated according to
    the Matula numbers for this specific depth.
    """
    def __init__(self, layer_idx: int, d_model: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        
        # Determine how many heads this layer gets based on OEIS A000081
        # (For computational feasibility in tests, we cap this)
        num_heads = A000081[layer_idx] if layer_idx < len(A000081) else 10
        num_heads = min(num_heads, 12) # Cap for memory in this experimental version
        
        # In a full implementation, we would map Matula numbers precisely.
        # Here, we generate 'num_heads' composite heads with sequential "Matula" numbers
        # starting from an offset.
        offset = sum(A000081[1:layer_idx]) if layer_idx > 1 else 1
        
        self.heads = nn.ModuleList([
            RecursiveCompositeHead(offset + i, d_model) 
            for i in range(num_heads)
        ])
        
        self.out_proj = nn.Linear(d_model * num_heads, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Run all heads in parallel
        head_outs = []
        for head in self.heads:
            out, _ = head(x, mask)
            head_outs.append(out)
            
        # Concatenate and project
        concat = torch.cat(head_outs, dim=-1)
        out = self.out_proj(concat)
        
        return self.layer_norm(x + out)

class RecursiveMatulaTransformer(nn.Module):
    """
    The complete recursive architecture.
    """
    def __init__(self, num_layers: int = 4, d_model: int = 128, vocab_size: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            RecursiveAttentionLayer(i, d_model) 
            for i in range(1, num_layers + 1)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, T, T)
        
        x = self.embed(idx)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
