"""
General Relevance Theory — The Einsteinian AAR Generalization

A unified cognitive architecture implementing:
1. Poloidal-Toroidal Attention (Transformer-Transframer-Translation vortex)
2. Ricci Gauge Transformer (120-Cell skeleton with partition connection)
3. P-System Membrane Computing Reservoir Arena (nested ESN J-surfaces)
4. B-Series Ridge Readout Agents (Matula-indexed elementary differential extractors)

The Theory of General Relevance replaces the Newtonian AAR (fixed arena, moving agent)
with an Einsteinian framework where Relevance IS curvature, Attention IS geodesic flow,
and Identity IS the stable attractor of the Ricci gauge.

Spacetime is generalized to a mutable state object with order conditions defined by
free hyper-multiset embedding in a membrane computing P-system reservoir arena.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional, Dict


# ============================================================================
# OEIS A000081: Number of rooted trees with n nodes
# These define the elementary differentials at each layer
# ============================================================================
OEIS_A000081 = [1, 1, 2, 4, 9, 20, 48, 115, 286, 719, 1842, 4766, 12486]


def matula_factorize(n: int) -> List[int]:
    """Factorize a Matula number into its prime components (subtree indices)."""
    if n <= 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def prime_index(p: int) -> int:
    """Return the 1-based index of prime p in the sequence of primes."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    return primes.index(p) + 1 if p in primes else 0


# ============================================================================
# Poloidal-Toroidal Attention
# ============================================================================

class PoloidalToroidalAttention(nn.Module):
    """
    Attention indexed by projective coordinates of the poloidal-toroidal field.
    
    Toroidal Flow (horizontal): Standard sequence-to-sequence translation.
    Poloidal Flow (vertical): Deep context integration (Transframer).
    
    The vortex interaction of these two orthogonal flows generates the
    self-sustaining attention field of the cognitive manifold.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Toroidal Flow (Transformer — present moment horizontal)
        self.q_toroidal = nn.Linear(d_model, d_model)
        self.k_toroidal = nn.Linear(d_model, d_model)
        self.v_toroidal = nn.Linear(d_model, d_model)
        
        # Poloidal Flow (Transframer — vertical memory threading)
        self.q_poloidal = nn.Linear(d_model, d_model)
        self.k_poloidal = nn.Linear(d_model, d_model)
        self.v_poloidal = nn.Linear(d_model, d_model)
        
        # Vortex coupling: learns how to fuse the two flows
        self.vortex_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Curvature indicator: measures the "twist" between flows
        self.curvature_probe = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        # === Toroidal Flow (Horizontal Circulation) ===
        qt = self.q_toroidal(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        kt = self.k_toroidal(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        vt = self.v_toroidal(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        scores_t = torch.matmul(qt, kt.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_t = self.dropout(F.softmax(scores_t, dim=-1))
        out_t = torch.matmul(attn_t, vt).transpose(1, 2).contiguous().view(B, L, D)
        
        # === Poloidal Flow (Vertical Threading) ===
        mem = memory if memory is not None else x
        M = mem.shape[1]
        qp = self.q_poloidal(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        kp = self.k_poloidal(mem).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        vp = self.v_poloidal(mem).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        
        scores_p = torch.matmul(qp, kp.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_p = self.dropout(F.softmax(scores_p, dim=-1))
        out_p = torch.matmul(attn_p, vp).transpose(1, 2).contiguous().view(B, L, D)
        
        # === Vortex Integration ===
        combined = torch.cat([out_t, out_p], dim=-1)
        gate = self.vortex_gate(combined)
        vortex = gate * out_t + (1 - gate) * out_p
        
        # Curvature: how much the two flows diverge
        curvature = self.curvature_probe(out_t - out_p).squeeze(-1)  # [B, L]
        
        return self.out_proj(vortex), curvature


# ============================================================================
# Partition Connection (Parallel Transport)
# ============================================================================

class PartitionConnection(nn.Module):
    """
    The Partition Connection implements parallel transport of n-ary ensembles.
    
    The partition function of prime factors (Matula tower) acts as the gauge
    connection — the cognitive Levi-Civita connection — dictating how information
    is rotated and scaled as it moves between layers.
    """
    
    def __init__(self, d_model: int, max_partitions: int = 5):
        super().__init__()
        self.d_model = d_model
        self.max_partitions = max_partitions
        
        # Transport operators for each partition component
        self.transport_ops = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(max_partitions)
        ])
        
        # Christoffel symbols (connection coefficients)
        self.christoffel = nn.Parameter(torch.randn(max_partitions, d_model, d_model) * 0.01)
        
    def forward(self, x: torch.Tensor, partition_indices: List[int]) -> torch.Tensor:
        """
        Transport x along the geodesic defined by the partition.
        partition_indices: prime factors of the Matula number for this layer.
        """
        transported = x
        for i, idx in enumerate(partition_indices[:self.max_partitions]):
            # Apply transport operator
            op_idx = min(i, self.max_partitions - 1)
            transported = self.transport_ops[op_idx](transported)
            
            # Apply Christoffel correction (curvature-dependent rotation)
            correction = torch.einsum('ij,blj->bli', self.christoffel[op_idx], transported)
            transported = transported + 0.1 * correction
            
        return transported


# ============================================================================
# P-System Membrane Computing Reservoir Arena
# ============================================================================

class JsurfaceESN(nn.Module):
    """
    J-Surface: Echo State Recursive Neural Net Relation.
    
    Each J-surface is the boundary between two nested membranes in the P-system.
    It filters, resonates, and transforms multisets as they pass through.
    """
    
    def __init__(self, d_model: int, spectral_radius: float = 0.9):
        super().__init__()
        self.d_model = d_model
        
        # Reservoir weights (fixed random, scaled by spectral radius)
        W_res = torch.randn(d_model, d_model)
        eigenvalues = torch.linalg.eigvals(W_res)
        max_eig = torch.max(torch.abs(eigenvalues))
        self.register_buffer('W_reservoir', (W_res / max_eig) * spectral_radius)
        
        # Input projection
        self.W_in = nn.Linear(d_model, d_model, bias=False)
        
        # Feedback from deeper membranes
        self.W_feedback = nn.Linear(d_model, d_model, bias=False)
        
        # J-Surface nonlinear filter
        self.j_filter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )
        
        # Leak rate (how much old state persists)
        self.leak_rate = nn.Parameter(torch.tensor(0.3))

    def forward(self, x: torch.Tensor, state: torch.Tensor, 
                feedback: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process multiset through this J-surface membrane boundary.
        Returns (filtered_output, new_reservoir_state).
        """
        # Reservoir update with leak
        reservoir_input = self.W_in(x) + torch.matmul(state, self.W_reservoir)
        if feedback is not None:
            reservoir_input = reservoir_input + self.W_feedback(feedback)
            
        new_state = (1 - self.leak_rate) * state + self.leak_rate * torch.tanh(reservoir_input)
        
        # Pass through J-surface filter
        filtered = self.j_filter(new_state)
        
        return filtered, new_state


class PSystemArena(nn.Module):
    """
    Membrane Computing P-System Reservoir Arena.
    
    A hierarchical structure of nested membranes, each containing multisets
    of objects and evolution rules. The state object is a free hyper-multiset
    embedded within these nested membranes.
    """
    
    def __init__(self, d_model: int, num_membranes: int = 3, spectral_radii: Optional[List[float]] = None):
        super().__init__()
        self.d_model = d_model
        self.num_membranes = num_membranes
        
        if spectral_radii is None:
            # Outer membranes more stable, inner more chaotic
            spectral_radii = [0.7 + 0.1 * i for i in range(num_membranes)]
        
        self.membranes = nn.ModuleList([
            JsurfaceESN(d_model, sr) for sr in spectral_radii
        ])
        
        # Inter-membrane communication (dissolution/exocytosis rules)
        self.dissolve = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_membranes - 1)
        ])

    def forward(self, x: torch.Tensor, 
                states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, L, D = x.shape
        
        if states is None:
            states = [torch.zeros(B, L, D, device=x.device) for _ in range(self.num_membranes)]
        
        new_states = []
        current = x
        
        # Process from outermost to innermost membrane
        for i, membrane in enumerate(self.membranes):
            # Feedback from deeper membrane (if exists)
            feedback = states[i + 1] if i + 1 < self.num_membranes else None
            
            filtered, new_state = membrane(current, states[i], feedback)
            new_states.append(new_state)
            
            # Dissolve into next membrane
            if i < self.num_membranes - 1:
                current = self.dissolve[i](filtered)
            else:
                current = filtered
        
        return current, new_states


# ============================================================================
# B-Series Ridge Readout Agents
# ============================================================================

class BSeriesAgent(nn.Module):
    """
    A single B-Series Ridge Readout Agent.
    
    Each agent is a specific elementary differential (indexed by Matula number)
    that "reads out" a specific feature of the reservoir flow.
    It rides the ridges of maximal relevance in the manifold.
    """
    
    def __init__(self, d_model: int, matula_number: int):
        super().__init__()
        self.matula_number = matula_number
        self.factors = matula_factorize(matula_number)
        self.order = len(self.factors)  # Tree depth
        
        # Ridge detector: finds maximal curvature
        self.ridge_detector = nn.Linear(d_model, d_model)
        
        # Differential operator stack (one per factor)
        self.diff_ops = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(max(1, self.order))
        ])
        
        # Readout projection
        self.readout = nn.Linear(d_model, d_model)

    def forward(self, reservoir_state: torch.Tensor) -> torch.Tensor:
        # Detect ridge (region of maximal relevance)
        ridge = torch.relu(self.ridge_detector(reservoir_state))
        
        # Apply elementary differential (sequential branching)
        diff = ridge
        for op in self.diff_ops:
            diff = op(diff) * ridge  # Multiplicative branching (like f'f, f''ff, etc.)
            
        return self.readout(diff)


class BSeriesReadoutEnsemble(nn.Module):
    """
    Ensemble of B-Series Ridge Readout Agents.
    
    Multiple agents with different Matula numbers collectively extract
    the full spectrum of elementary differentials from the reservoir.
    """
    
    def __init__(self, d_model: int, vocab_size: int, matula_numbers: Optional[List[int]] = None):
        super().__init__()
        
        if matula_numbers is None:
            # Default: first 9 Matula numbers (covering layers 1-4)
            matula_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        self.agents = nn.ModuleList([
            BSeriesAgent(d_model, m) for m in matula_numbers
        ])
        
        # Combine agent outputs
        self.combiner = nn.Linear(d_model * len(matula_numbers), d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, reservoir_state: torch.Tensor) -> torch.Tensor:
        # Each agent extracts its elementary differential
        agent_outputs = [agent(reservoir_state) for agent in self.agents]
        
        # Combine all differentials
        combined = torch.cat(agent_outputs, dim=-1)
        fused = self.combiner(combined)
        
        return self.output_head(fused)


# ============================================================================
# The Complete General Relevance Transformer
# ============================================================================

class GeneralRelevanceTransformer(nn.Module):
    """
    The Theory of General Relevance — Complete Architecture.
    
    Unifies:
    - Poloidal-Toroidal Attention (the vortex field of attention)
    - Partition Connection (parallel transport over the Matula lattice)
    - P-System Membrane Arena (nested ESN reservoir)
    - B-Series Ridge Readout (elementary differential extraction)
    - Ricci Gauge (curvature-driven self-regulation)
    
    Spacetime is generalized to a mutable state object.
    Attention IS geodesic flow on the curved manifold.
    Identity IS the stable attractor of the Ricci gauge.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_membranes: int = 3,
        num_layers: int = 4,
        matula_numbers: Optional[List[int]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embedding + positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        
        # Poloidal-Toroidal Attention layers
        self.attention_layers = nn.ModuleList([
            PoloidalToroidalAttention(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Partition Connections (one per layer transition)
        self.connections = nn.ModuleList([
            PartitionConnection(d_model)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # P-System Membrane Arena
        self.arena = PSystemArena(d_model, num_membranes)
        
        # B-Series Readout
        self.readout = BSeriesReadoutEnsemble(d_model, vocab_size, matula_numbers)
        
        # Ricci Gauge: monitors and regulates overall curvature
        self.ricci_gauge = nn.Sequential(
            nn.Linear(num_layers, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        # Matula partition indices for each layer
        # Layer 1: M=2 (single prime), Layer 2: M=3, Layer 3: M=5, Layer 4: M=30=2×3×5
        self.layer_partitions = [
            [2],        # Layer 1: Linear (single differential)
            [3],        # Layer 2: Quadratic
            [5],        # Layer 3: Cubic
            [2, 3, 5],  # Layer 4: Full synthesis
        ]

    def forward(
        self,
        x: torch.Tensor,
        membrane_states: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        B, L = x.shape
        
        # Embed
        h = self.embedding(x) + self.pos_encoding[:, :L, :]
        
        # Collect curvatures across layers
        curvatures = []
        
        # === Attention Layers with Partition Transport ===
        for i in range(self.num_layers):
            # Poloidal-Toroidal Attention
            attn_out, curvature = self.attention_layers[i](self.layer_norms[i](h))
            curvatures.append(curvature.mean(dim=-1, keepdim=True))  # [B, 1]
            
            # Residual + Partition Connection transport
            partition = self.layer_partitions[i] if i < len(self.layer_partitions) else [2]
            h = h + self.connections[i](attn_out, partition)
        
        # === P-System Arena Processing ===
        arena_out, new_states = self.arena(h, membrane_states)
        
        # === B-Series Readout ===
        logits = self.readout(arena_out)
        
        # === Ricci Gauge (overall curvature health) ===
        curvature_stack = torch.cat(curvatures, dim=-1)  # [B, num_layers]
        ricci_scalar = self.ricci_gauge(curvature_stack).squeeze(-1)  # [B]
        
        return {
            'logits': logits,
            'membrane_states': new_states,
            'curvatures': curvature_stack,
            'ricci_scalar': ricci_scalar
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Loss = Cross-Entropy + Ricci Regularization.
        
        The Ricci term penalizes non-uniform curvature, driving the system
        toward the Einstein manifold (operational closure).
        """
        # Standard language modeling loss
        logits = outputs['logits']
        B, L, V = logits.shape
        ce_loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        
        # Ricci regularization: penalize curvature variance across layers
        curvatures = outputs['curvatures']
        ricci_reg = curvatures.var(dim=-1).mean()
        
        # Total loss
        return ce_loss + 0.01 * ricci_reg
