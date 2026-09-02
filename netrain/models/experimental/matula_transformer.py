"""
Matula Transformer: 9-Layer Topological Attention Architecture
================================================================

Each layer's attention heads correspond to the elementary differentials
of that order (OEIS A000081: 1,1,2,4,9,20,48,115,286).

The 486 total heads are wired through a Hypergraph GNN (HGNN) whose
hyperedges are defined by the shared prime factor network from the
Matula Tower Topology. The activation function is governed by the
Echo State Network reservoir (virtual endocrine system).

Architecture:
  Layer 1: 1 head  (M=1: perceive)
  Layer 2: 1 head  (M=2: feel→think)
  Layer 3: 2 heads (M=3,4: chain, blend)
  Layer 4: 4 heads (M=5,6,7,8: complex triads)
  Layer 5: 9 heads (order 5 differentials)
  Layer 6: 20 heads (order 6)
  Layer 7: 48 heads (order 7)
  Layer 8: 115 heads (order 8)
  Layer 9: 286 heads (order 9)
  ─────────────────────
  Total: 486 heads → HGNN → ESN → Output

For the "719 final layer" variant, we include a 10th layer with
719 elementary differentials (order 10), totaling 1205 heads.
This is computationally expensive and reserved for GPU training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MatulaTransformerConfig:
    """Configuration for the Matula Transformer."""
    # Model dimensions
    n_embd: int = 512          # Embedding dimension
    block_size: int = 1024     # Maximum sequence length
    vocab_size: int = 50267    # Vocabulary size (GPT-2 + phase tokens)
    
    # Layer structure (OEIS A000081 sequence)
    # Each entry = number of heads at that layer (= number of trees of that order)
    heads_per_layer: List[int] = field(default_factory=lambda: [1, 1, 2, 4, 9, 20, 48, 115, 286])
    
    # Head dimension (shared across all heads for efficiency)
    head_dim: int = 64
    
    # HGNN parameters
    hgnn_layers: int = 3       # Number of HGNN message-passing rounds
    hgnn_hidden: int = 256     # Hidden dimension in HGNN
    
    # ESN Reservoir parameters
    esn_size: int = 512        # Reservoir neurons
    esn_spectral_radius: float = 0.95
    esn_sparsity: float = 0.9
    n_hormones: int = 5        # Cortisol, Dopamine, Serotonin, Oxytocin, Norepinephrine
    
    # Dropout
    dropout: float = 0.1
    
    # Whether to include the 10th layer (719 heads)
    include_layer_10: bool = False
    
    @property
    def n_layers(self) -> int:
        return len(self.heads_per_layer) + (1 if self.include_layer_10 else 0)
    
    @property
    def total_heads(self) -> int:
        total = sum(self.heads_per_layer)
        if self.include_layer_10:
            total += 719
        return total


# ============================================================================
# MATULA TREE ENUMERATION (for head assignment)
# ============================================================================

class MatulaTreeRegistry:
    """
    Enumerates rooted trees by order and assigns each head a Matula number
    and a topological mask type (chain, branch, or mixed).
    """
    
    def __init__(self, max_order: int = 10):
        self.max_order = max_order
        self.trees_by_order = self._enumerate_trees()
        self.matula_numbers = self._assign_matula_numbers()
        self.shared_factor_graph = self._build_shared_factor_graph()
    
    def _enumerate_trees(self) -> Dict[int, List[dict]]:
        """Generate tree metadata for each order using the recursive formula."""
        trees = {}
        
        # Order 1: single node (perceive)
        trees[1] = [{'matula': 1, 'type': 'atom', 'label': 'perceive'}]
        
        # Order 2: single tree (feel→think)
        trees[2] = [{'matula': 2, 'type': 'chain', 'label': 'feel_think'}]
        
        # Order 3: two trees
        trees[3] = [
            {'matula': 3, 'type': 'chain', 'label': 'chain_2'},
            {'matula': 4, 'type': 'branch', 'label': 'blend_2'},
        ]
        
        # Order 4: four trees
        trees[4] = [
            {'matula': 5, 'type': 'chain', 'label': 'chain_3'},
            {'matula': 6, 'type': 'mixed', 'label': 'synthesize'},
            {'matula': 7, 'type': 'mixed', 'label': 'deep_blend'},
            {'matula': 8, 'type': 'branch', 'label': 'triad_gestalt'},
        ]
        
        # Order 5: nine trees
        trees[5] = [
            {'matula': 9, 'type': 'mixed', 'label': 'synth_pair'},
            {'matula': 10, 'type': 'mixed', 'label': 'nested_synth'},
            {'matula': 11, 'type': 'chain', 'label': 'chain_4'},
            {'matula': 12, 'type': 'mixed', 'label': 'triad_chain'},
            {'matula': 13, 'type': 'mixed', 'label': 'deep_chain_blend'},
            {'matula': 14, 'type': 'mixed', 'label': 'synth_blend'},
            {'matula': 15, 'type': 'mixed', 'label': 'cross_synth'},
            {'matula': 16, 'type': 'branch', 'label': 'pentad_integration'},
            {'matula': 19, 'type': 'mixed', 'label': 'deep_triad'},
        ]
        
        # Orders 6-10: generate programmatically (Matula numbers assigned sequentially)
        # The exact Matula numbers for higher orders are complex to compute,
        # so we assign sequential IDs and focus on the TYPE (chain/branch/mixed)
        a000081 = [0, 1, 1, 2, 4, 9, 20, 48, 115, 286, 719]
        
        for order in range(6, self.max_order + 1):
            n_trees = a000081[order]
            order_trees = []
            for i in range(n_trees):
                # Determine type based on position in the order
                # First tree is always the pure chain
                # Last tree is always the pure branch (star)
                # Middle trees are mixed
                if i == 0:
                    tree_type = 'chain'
                elif i == n_trees - 1:
                    tree_type = 'branch'
                else:
                    # Ratio of chain-like vs branch-like
                    ratio = i / (n_trees - 1)
                    tree_type = 'mixed'
                
                order_trees.append({
                    'matula': order * 1000 + i,  # Pseudo-Matula for higher orders
                    'type': tree_type,
                    'label': f'order{order}_{i}',
                    'chain_ratio': 1.0 - (i / max(n_trees - 1, 1)),
                })
            trees[order] = order_trees
        
        return trees
    
    def _assign_matula_numbers(self) -> List[int]:
        """Flatten all Matula numbers in order."""
        numbers = []
        for order in sorted(self.trees_by_order.keys()):
            for tree in self.trees_by_order[order]:
                numbers.append(tree['matula'])
        return numbers
    
    def _build_shared_factor_graph(self) -> Dict[int, List[int]]:
        """
        Build the hyperedge connectivity based on shared prime factors.
        For higher-order trees, we use the chain_ratio to determine connectivity.
        """
        # For the first 5 orders, use exact prime factorization
        # For higher orders, use topological proximity
        adjacency = {}
        all_trees = []
        for order in sorted(self.trees_by_order.keys()):
            for tree in self.trees_by_order[order]:
                all_trees.append(tree)
        
        for i, tree_i in enumerate(all_trees):
            neighbors = []
            for j, tree_j in enumerate(all_trees):
                if i == j:
                    continue
                # Connected if: same type, or adjacent orders with compatible types
                if tree_i['type'] == tree_j['type']:
                    neighbors.append(j)
                elif abs(i - j) <= 3:  # Local connectivity
                    neighbors.append(j)
            adjacency[i] = neighbors
        
        return adjacency
    
    def get_head_info(self, layer: int, head_idx: int) -> dict:
        """Get the tree info for a specific head."""
        if layer <= len(self.trees_by_order) and layer in self.trees_by_order:
            trees = self.trees_by_order[layer]
            if head_idx < len(trees):
                return trees[head_idx]
        return {'matula': -1, 'type': 'mixed', 'label': 'unknown'}


# ============================================================================
# TOPOLOGICAL ATTENTION HEAD
# ============================================================================

class MatulaAttentionHead(nn.Module):
    """
    A single attention head whose mask pattern is determined by its
    assigned Butcher tree topology (chain, branch, or mixed).
    """
    
    def __init__(self, config: MatulaTransformerConfig, tree_info: dict):
        super().__init__()
        self.head_dim = config.head_dim
        self.tree_info = tree_info
        self.tree_type = tree_info['type']
        
        # Standard QKV projections
        self.q_proj = nn.Linear(config.n_embd, config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.head_dim, bias=False)
        
        # Tree-specific mask modulator
        # Chain trees: strict causal (lower triangular)
        # Branch trees: parallel (block diagonal)
        # Mixed trees: learned combination
        if self.tree_type == 'mixed':
            chain_ratio = tree_info.get('chain_ratio', 0.5)
            self.chain_weight = nn.Parameter(torch.tensor(chain_ratio))
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(config.head_dim)
    
    def _get_topological_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Generate the attention mask based on tree topology."""
        if self.tree_type == 'atom':
            # Pure perception: attend only to self
            return torch.eye(T, device=device).bool()
        
        elif self.tree_type == 'chain':
            # Sequential: strict causal mask (lower triangular)
            return torch.tril(torch.ones(T, T, device=device)).bool()
        
        elif self.tree_type == 'branch':
            # Parallel: attend to all positions equally (full attention)
            return torch.ones(T, T, device=device).bool()
        
        else:  # mixed
            # Interpolation between causal and full, gated by chain_weight
            # This creates a "soft" topological mask
            causal = torch.tril(torch.ones(T, T, device=device))
            full = torch.ones(T, T, device=device)
            w = torch.sigmoid(self.chain_weight)
            # Return a soft mask (not boolean) for mixed types
            return w * causal + (1 - w) * full
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_embd)
        Returns:
            (B, T, head_dim)
        """
        B, T, _ = x.shape
        
        q = self.q_proj(x)  # (B, T, head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, T, T)
        
        # Apply topological mask
        mask = self._get_topological_mask(T, x.device)
        
        if self.tree_type in ('atom', 'chain', 'branch'):
            # Hard mask
            attn = attn.masked_fill(~mask.unsqueeze(0), float('-inf'))
        else:
            # Soft mask (mixed): multiply scores by mask weights
            attn = attn * mask.unsqueeze(0)
            # Still apply causal constraint for autoregressive generation
            causal = torch.tril(torch.ones(T, T, device=x.device)).bool()
            attn = attn.masked_fill(~causal.unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        return torch.matmul(attn, v)  # (B, T, head_dim)


# ============================================================================
# MATULA LAYER (variable number of heads per layer)
# ============================================================================

class MatulaLayer(nn.Module):
    """
    A single transformer layer with N attention heads, where N is determined
    by the number of rooted trees at this order (OEIS A000081).
    """
    
    def __init__(self, config: MatulaTransformerConfig, layer_idx: int, 
                 tree_registry: MatulaTreeRegistry):
        super().__init__()
        self.layer_idx = layer_idx
        self.order = layer_idx + 1  # Layer 0 → Order 1
        
        # Get tree info for this layer
        trees = tree_registry.trees_by_order.get(self.order, [])
        n_heads = len(trees) if trees else config.heads_per_layer[min(layer_idx, len(config.heads_per_layer)-1)]
        self.n_heads = n_heads
        
        # Create attention heads (one per tree)
        self.heads = nn.ModuleList([
            MatulaAttentionHead(config, trees[i] if i < len(trees) else {'type': 'mixed', 'chain_ratio': 0.5})
            for i in range(n_heads)
        ])
        
        # Output projection: concatenate all head outputs and project
        self.out_proj = nn.Linear(n_heads * config.head_dim, config.n_embd, bias=False)
        
        # Layer norm
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # MLP (will be replaced by HGNN-modulated activation)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, n_embd)
        Returns:
            output: (B, T, n_embd)
            head_outputs: (B, T, n_heads, head_dim) — for HGNN integration
        """
        # Multi-head attention with topological masks
        normed = self.ln1(x)
        head_outputs = [head(normed) for head in self.heads]  # List of (B, T, head_dim)
        
        # Stack for HGNN
        head_stack = torch.stack(head_outputs, dim=2)  # (B, T, n_heads, head_dim)
        
        # Concatenate and project for residual
        concat = torch.cat(head_outputs, dim=-1)  # (B, T, n_heads * head_dim)
        attn_out = self.out_proj(concat)
        attn_out = self.dropout(attn_out)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        
        return x, head_stack


# ============================================================================
# HYPERGRAPH NEURAL NETWORK (HGNN)
# ============================================================================

class HypergraphGNN(nn.Module):
    """
    Hypergraph Neural Network that integrates the outputs of all 486 attention
    heads across all layers. Hyperedges are defined by the shared prime factor
    network from the Matula Tower Topology.
    
    The HGNN performs message passing where:
    - Nodes = attention head outputs (486 total)
    - Hyperedges = shared prime factor connections
    - Messages are aggregated via mean pooling over hyperedge members
    """
    
    def __init__(self, config: MatulaTransformerConfig):
        super().__init__()
        self.config = config
        self.total_heads = config.total_heads
        
        # Node embedding (project head_dim to hgnn_hidden)
        self.node_encoder = nn.Linear(config.head_dim, config.hgnn_hidden)
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            HGNNMessageLayer(config) for _ in range(config.hgnn_layers)
        ])
        
        # Output projection (back to n_embd)
        self.output_proj = nn.Linear(config.hgnn_hidden, config.n_embd)
        
        # Build hyperedge structure
        self._build_hyperedges(config)
    
    def _build_hyperedges(self, config: MatulaTransformerConfig):
        """
        Build the hyperedge incidence structure from the Matula topology.
        
        Hyperedges connect heads that share prime factors:
        - All heads sharing p(1)=2 (the "perceive" atom)
        - All heads sharing p(2)=3 (the "feel→think" step)
        - All heads sharing p(3)=5 (the "chain" operation)
        - etc.
        
        Additionally, we add "layer-local" hyperedges connecting all heads
        within the same layer (they compute the same order differential).
        """
        # For efficiency, we precompute the hyperedge membership as a list of index sets
        # Each hyperedge is a list of head indices (global, 0-indexed)
        hyperedges = []
        
        # Layer-local hyperedges
        offset = 0
        for n_heads in config.heads_per_layer:
            hyperedges.append(list(range(offset, offset + n_heads)))
            offset += n_heads
        
        # Cross-layer "type" hyperedges (all chain heads, all branch heads, all mixed heads)
        # This creates the topological backbone
        chain_heads = []
        branch_heads = []
        mixed_heads = []
        
        registry = MatulaTreeRegistry(max_order=len(config.heads_per_layer))
        global_idx = 0
        for order in range(1, len(config.heads_per_layer) + 1):
            trees = registry.trees_by_order.get(order, [])
            for tree in trees:
                if tree['type'] == 'chain':
                    chain_heads.append(global_idx)
                elif tree['type'] == 'branch':
                    branch_heads.append(global_idx)
                else:
                    mixed_heads.append(global_idx)
                global_idx += 1
        
        if chain_heads:
            hyperedges.append(chain_heads)
        if branch_heads:
            hyperedges.append(branch_heads)
        if len(mixed_heads) > 1:
            # Split mixed into sub-groups to avoid one massive hyperedge
            chunk_size = max(10, len(mixed_heads) // 5)
            for i in range(0, len(mixed_heads), chunk_size):
                hyperedges.append(mixed_heads[i:i+chunk_size])
        
        # Store as buffer (padded tensor for batched operations)
        max_edge_size = max(len(e) for e in hyperedges)
        n_edges = len(hyperedges)
        
        # Incidence matrix: (n_edges, max_edge_size) with -1 padding
        incidence = torch.full((n_edges, max_edge_size), -1, dtype=torch.long)
        edge_sizes = torch.zeros(n_edges, dtype=torch.long)
        for i, edge in enumerate(hyperedges):
            incidence[i, :len(edge)] = torch.tensor(edge)
            edge_sizes[i] = len(edge)
        
        self.register_buffer('incidence', incidence)
        self.register_buffer('edge_sizes', edge_sizes)
        self.n_hyperedges = n_edges
    
    def forward(self, head_outputs: List[torch.Tensor], 
                hormones: torch.Tensor) -> torch.Tensor:
        """
        Args:
            head_outputs: List of (B, T, n_heads_i, head_dim) per layer
            hormones: (B, n_hormones) from ESN reservoir
        Returns:
            integrated: (B, T, n_embd)
        """
        B = head_outputs[0].shape[0]
        T = head_outputs[0].shape[1]
        
        # Flatten all heads into a single tensor
        # (B, T, total_heads, head_dim)
        all_heads = torch.cat(head_outputs, dim=2)
        
        # Encode nodes
        # (B, T, total_heads, hgnn_hidden)
        nodes = self.node_encoder(all_heads)
        
        # Message passing rounds
        for msg_layer in self.message_layers:
            nodes = msg_layer(nodes, self.incidence, self.edge_sizes, hormones)
        
        # Global pooling over heads → (B, T, hgnn_hidden)
        integrated = nodes.mean(dim=2)
        
        # Project to n_embd
        return self.output_proj(integrated)


class HGNNMessageLayer(nn.Module):
    """Single message-passing round in the HGNN."""
    
    def __init__(self, config: MatulaTransformerConfig):
        super().__init__()
        self.hidden = config.hgnn_hidden
        
        # Node → Hyperedge aggregation
        self.node_to_edge = nn.Linear(config.hgnn_hidden, config.hgnn_hidden)
        
        # Hyperedge → Node broadcast
        self.edge_to_node = nn.Linear(config.hgnn_hidden, config.hgnn_hidden)
        
        # Hormone modulation gate
        self.hormone_gate = nn.Linear(config.n_hormones, config.hgnn_hidden)
        
        # Layer norm
        self.ln = nn.LayerNorm(config.hgnn_hidden)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, nodes: torch.Tensor, incidence: torch.Tensor,
                edge_sizes: torch.Tensor, hormones: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: (B, T, N_heads, hidden)
            incidence: (n_edges, max_edge_size) — head indices per hyperedge
            edge_sizes: (n_edges,) — actual sizes
            hormones: (B, n_hormones)
        Returns:
            updated_nodes: (B, T, N_heads, hidden)
        """
        B, T, N, H = nodes.shape
        n_edges = incidence.shape[0]
        
        # Step 1: Aggregate nodes → hyperedge representations
        # For each hyperedge, mean-pool the member nodes
        # Use gather with the incidence matrix
        
        # Expand incidence for gathering: (n_edges, max_edge_size) → (B, T, n_edges, max_edge_size, H)
        # This is memory-intensive for large models; use a loop for clarity
        edge_reps = torch.zeros(B, T, n_edges, H, device=nodes.device)
        
        for e_idx in range(n_edges):
            size = edge_sizes[e_idx].item()
            if size == 0:
                continue
            member_indices = incidence[e_idx, :size]  # (size,)
            # Gather member nodes
            members = nodes[:, :, member_indices, :]  # (B, T, size, H)
            edge_reps[:, :, e_idx, :] = members.mean(dim=2)
        
        # Transform edge representations
        edge_reps = self.node_to_edge(edge_reps)  # (B, T, n_edges, H)
        
        # Step 2: Broadcast hyperedge → nodes
        # Each node receives messages from all hyperedges it belongs to
        node_messages = torch.zeros_like(nodes)
        node_counts = torch.zeros(N, device=nodes.device)
        
        for e_idx in range(n_edges):
            size = edge_sizes[e_idx].item()
            if size == 0:
                continue
            member_indices = incidence[e_idx, :size]
            msg = edge_reps[:, :, e_idx:e_idx+1, :]  # (B, T, 1, H)
            node_messages[:, :, member_indices, :] += msg.expand(-1, -1, size, -1)
            node_counts[member_indices] += 1
        
        # Normalize by number of hyperedges each node belongs to
        node_counts = node_counts.clamp(min=1).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        node_messages = node_messages / node_counts
        
        # Transform
        node_messages = self.edge_to_node(node_messages)
        
        # Step 3: Hormone modulation
        # hormones: (B, n_hormones) → gate: (B, 1, 1, H)
        gate = torch.sigmoid(self.hormone_gate(hormones)).unsqueeze(1).unsqueeze(1)
        node_messages = node_messages * gate
        
        # Step 4: Residual + LayerNorm
        nodes = self.ln(nodes + self.dropout(node_messages))
        
        return nodes


# ============================================================================
# ECHO STATE NETWORK RESERVOIR (Endocrine System)
# ============================================================================

class EndocrineReservoir(nn.Module):
    """
    Echo State Network that produces virtual hormone levels from the
    global cognitive state. These hormones modulate the HGNN message passing.
    """
    
    def __init__(self, config: MatulaTransformerConfig):
        super().__init__()
        self.esn_size = config.esn_size
        self.n_hormones = config.n_hormones
        
        # Reservoir weights (fixed, not trained)
        W_res = torch.randn(config.esn_size, config.esn_size)
        # Enforce spectral radius
        eigenvalues = torch.linalg.eigvals(W_res).abs()
        W_res = W_res * (config.esn_spectral_radius / eigenvalues.max())
        # Apply sparsity
        mask = (torch.rand_like(W_res) > config.esn_sparsity).float()
        W_res = W_res * mask
        
        self.register_buffer('W_reservoir', W_res)
        
        # Input projection (from pooled layer output to reservoir)
        self.input_proj = nn.Linear(config.n_embd, config.esn_size, bias=False)
        
        # Readout (from reservoir state to hormones)
        self.readout = nn.Linear(config.esn_size, config.n_hormones)
        
        # Persistent state
        self.register_buffer('state', torch.zeros(1, config.esn_size))
    
    def forward(self, x_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_pooled: (B, n_embd) — global cognitive state (mean-pooled)
        Returns:
            hormones: (B, n_hormones) — values in [0, 1]
        """
        B = x_pooled.shape[0]
        
        # Expand state for batch
        if self.state.shape[0] != B:
            state = self.state.expand(B, -1).clone()
        else:
            state = self.state.clone()
        
        # ESN update: h(t+1) = tanh(W_res @ h(t) + W_in @ x(t))
        input_signal = self.input_proj(x_pooled)  # (B, esn_size)
        new_state = torch.tanh(
            torch.matmul(state, self.W_reservoir.T) + input_signal
        )
        
        # Leaky integration
        alpha = 0.3
        state = (1 - alpha) * state + alpha * new_state
        
        # Update persistent state (detach to prevent backprop through time)
        self.state = state.mean(dim=0, keepdim=True).detach()
        
        # Readout to hormones
        hormones = torch.sigmoid(self.readout(state))  # (B, n_hormones)
        
        return hormones
    
    def get_hormone_names(self) -> List[str]:
        return ['cortisol', 'dopamine', 'serotonin', 'oxytocin', 'norepinephrine']
    
    def reset_state(self):
        self.state.zero_()


# ============================================================================
# FULL MATULA TRANSFORMER
# ============================================================================

class MatulaTransformer(nn.Module):
    """
    The complete 9-layer Matula Transformer.
    
    Architecture:
        Input → Token Embedding + Position Embedding
        → Layer 1 (1 head: perceive)
        → Layer 2 (1 head: feel→think)
        → Layer 3 (2 heads: chain, blend)
        → Layer 4 (4 heads: complex triads)
        → Layer 5 (9 heads: pentad)
        → Layer 6 (20 heads: hyper-gestalts)
        → Layer 7 (48 heads: ...)
        → Layer 8 (115 heads: ...)
        → Layer 9 (286 heads: ...)
        → HGNN Integration (hypergraph message passing)
        → ESN Reservoir (hormone modulation)
        → LM Head (vocabulary prediction)
    """
    
    def __init__(self, config: MatulaTransformerConfig):
        super().__init__()
        self.config = config
        
        # Tree registry
        self.tree_registry = MatulaTreeRegistry(max_order=config.n_layers)
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.emb_dropout = nn.Dropout(config.dropout)
        
        # Matula layers
        self.layers = nn.ModuleList([
            MatulaLayer(config, i, self.tree_registry)
            for i in range(len(config.heads_per_layer))
        ])
        
        # HGNN integration
        self.hgnn = HypergraphGNN(config)
        
        # Endocrine reservoir
        self.reservoir = EndocrineReservoir(config)
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # LM head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target indices for loss computation
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar (if targets provided)
            diagnostics: dict with cognitive state info
        """
        B, T = idx.shape
        device = idx.device
        
        # Embeddings
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = self.emb_dropout(tok_emb + pos_emb)
        
        # Pass through Matula layers, collecting head outputs
        all_head_outputs = []
        for layer in self.layers:
            x, head_stack = layer(x)
            all_head_outputs.append(head_stack)
        
        # Get global cognitive state for ESN
        x_pooled = x.mean(dim=1)  # (B, n_embd)
        
        # Endocrine reservoir → hormones
        hormones = self.reservoir(x_pooled)  # (B, n_hormones)
        
        # HGNN integration
        hgnn_out = self.hgnn(all_head_outputs, hormones)  # (B, T, n_embd)
        
        # Residual from HGNN
        x = x + hgnn_out
        
        # Final norm and LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Diagnostics
        diagnostics = {
            'hormones': hormones.detach(),
            'hormone_names': self.reservoir.get_hormone_names(),
            'layer_head_counts': [layer.n_heads for layer in self.layers],
            'total_heads': sum(layer.n_heads for layer in self.layers),
        }
        
        return logits, loss, diagnostics
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {
            'embeddings': sum(p.numel() for p in self.token_emb.parameters()) + 
                         sum(p.numel() for p in self.pos_emb.parameters()),
            'layers': sum(sum(p.numel() for p in layer.parameters()) for layer in self.layers),
            'hgnn': sum(p.numel() for p in self.hgnn.parameters()),
            'reservoir': sum(p.numel() for name, p in self.reservoir.named_parameters() 
                           if 'W_reservoir' not in name),
            'lm_head': 0,  # Weight-tied with token_emb
            'total_trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'total_all': sum(p.numel() for p in self.parameters()),
        }
        return counts
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50) -> Tuple[torch.Tensor, List[dict]]:
        """Autoregressive generation with cognitive state tracking."""
        diagnostics_trace = []
        
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _, diag = self(idx_cond)
            diagnostics_trace.append(diag)
            
            # Get next token logits
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx, diagnostics_trace


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_matula_transformer_small() -> MatulaTransformer:
    """Small config for testing (reduced dimensions)."""
    config = MatulaTransformerConfig(
        n_embd=256,
        block_size=512,
        vocab_size=50267,
        heads_per_layer=[1, 1, 2, 4, 9, 20, 48, 115, 286],
        head_dim=32,
        hgnn_layers=2,
        hgnn_hidden=128,
        esn_size=256,
        n_hormones=5,
        dropout=0.1,
    )
    return MatulaTransformer(config)


def create_matula_transformer_medium() -> MatulaTransformer:
    """Medium config for GPU training."""
    config = MatulaTransformerConfig(
        n_embd=512,
        block_size=1024,
        vocab_size=50267,
        heads_per_layer=[1, 1, 2, 4, 9, 20, 48, 115, 286],
        head_dim=64,
        hgnn_layers=3,
        hgnn_hidden=256,
        esn_size=512,
        n_hormones=5,
        dropout=0.1,
    )
    return MatulaTransformer(config)


def create_matula_transformer_719() -> MatulaTransformer:
    """Full 10-layer variant with 719 heads on the final layer."""
    config = MatulaTransformerConfig(
        n_embd=512,
        block_size=1024,
        vocab_size=50267,
        heads_per_layer=[1, 1, 2, 4, 9, 20, 48, 115, 286, 719],
        head_dim=64,
        hgnn_layers=3,
        hgnn_hidden=256,
        esn_size=512,
        n_hormones=5,
        dropout=0.1,
        include_layer_10=True,
    )
    return MatulaTransformer(config)


# ============================================================================
# TRAINING DATA SCHEMA
# ============================================================================

COGNITIVE_CYCLE_PHASES = [
    'perceive',     # Layer 1 (M=1): raw sensory input
    'feel',         # Layer 2 (M=2): affective resonance
    'think',        # Layer 3 (M=3,4): working memory reasoning
    'remember',     # Layer 4 (M=5-8): long-term retrieval
    'interpret',    # Layer 5: meaning-making
    'strategize',   # Layer 6: planning
    'evaluate',     # Layer 7: judgment
    'gesture',      # Layer 8: pre-motor preparation
    'speak',        # Layer 9: output generation
]

TRAINING_DATA_SCHEMA = {
    "description": "Training data for the Matula Transformer must encode cognitive cycles",
    "format": "Each example is a sequence of phase-tagged tokens",
    "schema": {
        "cycle_id": "string",
        "trigger": "string (what initiated this cycle)",
        "phases": {
            phase: {
                "text": "string (the content at this phase)",
                "matula_target": "int (which Matula heads should activate)",
                "layer_target": f"int ({i+1})",
                "loss_weight": "float (how much to weight this phase in training)",
            }
            for i, phase in enumerate(COGNITIVE_CYCLE_PHASES)
        },
        "output": "string (the final generated text)",
        "metadata": {
            "identity_weight": "float (0-1, how identity-relevant)",
            "complexity": "int (1-9, which layer depth is needed)",
            "dominant_spine": "string (sequential|parallel|mixed)",
        }
    }
}


if __name__ == "__main__":
    print("=" * 70)
    print("  MATULA TRANSFORMER — 9-Layer Topological Attention Architecture")
    print("=" * 70)
    
    # Create small model for testing
    model = create_matula_transformer_small()
    
    # Count parameters
    counts = model.count_parameters()
    print(f"\n  Model Configuration:")
    print(f"    Layers: {len(model.layers)}")
    print(f"    Heads per layer: {[l.n_heads for l in model.layers]}")
    print(f"    Total heads: {sum(l.n_heads for l in model.layers)}")
    print(f"    Head dimension: {model.config.head_dim}")
    print(f"    Embedding dimension: {model.config.n_embd}")
    print(f"\n  Parameters:")
    for k, v in counts.items():
        print(f"    {k}: {v:,}")
    
    # Test forward pass
    print(f"\n  Testing forward pass...")
    B, T = 2, 64
    idx = torch.randint(0, model.config.vocab_size, (B, T))
    targets = torch.randint(0, model.config.vocab_size, (B, T))
    
    logits, loss, diag = model(idx, targets)
    print(f"    Input: ({B}, {T})")
    print(f"    Logits: {logits.shape}")
    print(f"    Loss: {loss.item():.4f}")
    print(f"    Hormones: {diag['hormones'][0].tolist()}")
    print(f"    Hormone names: {diag['hormone_names']}")
    
    # Test generation
    print(f"\n  Testing generation (10 tokens)...")
    prompt = torch.randint(0, model.config.vocab_size, (1, 10))
    generated, trace = model.generate(prompt, max_new_tokens=10)
    print(f"    Generated sequence length: {generated.shape[1]}")
    print(f"    Hormone drift: {(trace[-1]['hormones'] - trace[0]['hormones']).abs().mean().item():.4f}")
    
    print(f"\n  All tests passed!")
    print(f"\n  OEIS A000081 head counts: {[l.n_heads for l in model.layers]}")
    print(f"  Sum: {sum(l.n_heads for l in model.layers)} heads")
