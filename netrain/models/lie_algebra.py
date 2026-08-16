"""
Lie Algebra Commutator Head
============================

Replaces the standard linear language model head (nn.Linear(C, V)) with a
Lie algebra-based projection that computes word probabilities from the
commutator structure of adjacent gestalt primitives.

Mathematical Foundation:
  - Each gestalt primitive is embedded as a matrix M_i in gl(d, R) (Lie algebra element)
  - The commutator [M_i, M_{i+1}] = M_i @ M_{i+1} - M_{i+1} @ M_i
    represents the "generative tension" between adjacent concepts
  - This commutator is projected into QKV space where:
    Q = commutator (what the gestalt "asks for" in language)
    K = text vocabulary embeddings (what words "offer")
    V = text vocabulary semantics
  - The attention score Q·K^T gives the probability over text tokens

The key insight: language is not a lookup table from hidden states to words.
Language EMERGES from the non-commutative structure of thought. The order
in which you think about things matters — [joy, loss] generates different
language than [loss, joy], even though the set of concepts is identical.

Biological Analogy:
  The commutator is like the "gesture" that precedes speech — the felt
  sense of what needs to be expressed, which then finds its words.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class LieAlgebraConfig:
    """Configuration for the Lie Algebra projection."""
    gestalt_dim: int = 768       # Dimension of gestalt embeddings
    lie_rank: int = 32           # Rank of Lie algebra matrices (d×d where d=lie_rank)
    text_vocab_size: int = 50257 # Output text vocabulary
    n_commutator_heads: int = 8  # Multi-head commutator computation
    temperature: float = 1.0     # Softmax temperature for word selection
    use_structure_constants: bool = True  # Use learned structure constants


class LieAlgebraElement(nn.Module):
    """
    Maps a vector to a Lie algebra element (matrix).

    Each gestalt embedding is lifted into gl(d, R) — the space of d×d matrices.
    This gives each concept a "direction of transformation" rather than just
    a point in space.
    """

    def __init__(self, input_dim: int, lie_rank: int):
        super().__init__()
        self.lie_rank = lie_rank
        # Project from embedding space to matrix space
        self.to_matrix = nn.Linear(input_dim, lie_rank * lie_rank)
        # Antisymmetric projection (ensures we stay in the Lie algebra)
        # For so(d): M → (M - M^T) / 2
        self.antisymmetrize = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input vectors (B, T, C)

        Returns:
            Lie algebra elements (B, T, d, d)
        """
        B, T, C = x.shape
        M = self.to_matrix(x).view(B, T, self.lie_rank, self.lie_rank)

        if self.antisymmetrize:
            # Project to so(d) — antisymmetric matrices
            M = (M - M.transpose(-2, -1)) / 2

        return M


class CommutatorComputer(nn.Module):
    """
    Computes the Lie bracket (commutator) between adjacent elements.

    [A, B] = AB - BA

    This is the core operation: it extracts the "generative tension"
    between consecutive gestalt primitives. The commutator captures
    what is UNIQUE about the ordering — the irreducible novelty that
    emerges from juxtaposition.

    Multi-head: we compute multiple commutators with different projections,
    capturing different aspects of the generative tension.
    """

    def __init__(self, config: LieAlgebraConfig):
        super().__init__()
        self.n_heads = config.n_commutator_heads
        self.lie_rank = config.lie_rank
        self.head_rank = config.lie_rank  # Each head operates on full rank

        # Per-head projection (different "views" of the Lie structure)
        self.head_projs = nn.ModuleList([
            nn.Linear(config.lie_rank * config.lie_rank, config.lie_rank * config.lie_rank)
            for _ in range(self.n_heads)
        ])

        # Structure constants (optional: learned f^k_ij for [e_i, e_j] = f^k_ij e_k)
        if config.use_structure_constants:
            # Learned structure constants of the Lie algebra
            self.structure_constants = nn.Parameter(
                torch.randn(config.lie_rank, config.lie_rank, config.lie_rank) * 0.01
            )
            # Antisymmetry constraint: f^k_ij = -f^k_ji
            # Enforced in forward pass
        else:
            self.structure_constants = None

        # Combine multi-head commutators
        self.combine = nn.Linear(
            self.n_heads * config.lie_rank * config.lie_rank,
            config.gestalt_dim
        )

        self.layer_norm = nn.LayerNorm(config.gestalt_dim)

    def forward(self, lie_elements: torch.Tensor) -> torch.Tensor:
        """
        Compute commutators between adjacent Lie algebra elements.

        Args:
            lie_elements: (B, T, d, d) — sequence of Lie algebra matrices

        Returns:
            commutator_features: (B, T, gestalt_dim) — extracted tension features
        """
        B, T, d, _ = lie_elements.shape

        # Shift for adjacent pairs: [M_t, M_{t-1}]
        M_curr = lie_elements[:, 1:, :, :]   # (B, T-1, d, d)
        M_prev = lie_elements[:, :-1, :, :]  # (B, T-1, d, d)

        # Pad first position (no predecessor → zero commutator)
        zero_pad = torch.zeros(B, 1, d, d, device=lie_elements.device)

        # Multi-head commutator computation
        head_outputs = []
        for i, proj in enumerate(self.head_projs):
            # Project to head-specific view
            M_c_flat = M_curr.reshape(B, T-1, d*d)
            M_p_flat = M_prev.reshape(B, T-1, d*d)
            M_c_proj = proj(M_c_flat).view(B, T-1, d, d)
            M_p_proj = proj(M_p_flat).view(B, T-1, d, d)

            # Compute commutator: [A, B] = AB - BA
            comm = torch.matmul(M_c_proj, M_p_proj) - torch.matmul(M_p_proj, M_c_proj)

            # Apply structure constants if available
            if self.structure_constants is not None:
                # Enforce antisymmetry
                f = (self.structure_constants - self.structure_constants.transpose(0, 1)) / 2
                # Contract with structure constants for richer algebra
                comm_contracted = torch.einsum('btij,ijk->btk', comm, f)
                # Expand back
                comm = comm + comm_contracted.unsqueeze(-1) * 0.1  # Residual

            # Pad and flatten
            comm_padded = torch.cat([zero_pad, comm], dim=1)  # (B, T, d, d)
            head_outputs.append(comm_padded.reshape(B, T, d*d))

        # Combine all heads
        combined = torch.cat(head_outputs, dim=-1)  # (B, T, n_heads * d * d)
        features = self.combine(combined)  # (B, T, gestalt_dim)

        return self.layer_norm(features)


class LieCommutatorHead(nn.Module):
    """
    The Lie Algebra Commutator Language Model Head.

    Replaces the standard `lm_head = nn.Linear(C, V)` with a system that:
    1. Lifts hidden states to Lie algebra elements
    2. Computes commutators between adjacent elements
    3. Uses commutators as Queries against text vocabulary Keys
    4. Produces word probabilities from the Q·K attention

    The text vocabulary is embedded as a learned Key matrix, where each
    word's Key represents "what kind of generative tension this word resolves."

    This means: words are not stored as points, but as RESPONSES to tensions.
    A word is selected because it best resolves the commutator — the felt
    sense of what needs to be expressed.
    """

    def __init__(self, config: LieAlgebraConfig):
        super().__init__()
        self.config = config

        # Lift hidden states to Lie algebra
        self.lie_lift = LieAlgebraElement(config.gestalt_dim, config.lie_rank)

        # Compute commutators
        self.commutator = CommutatorComputer(config)

        # Text vocabulary as Keys (what words "offer" to resolve tension)
        self.text_keys = nn.Parameter(
            torch.randn(config.text_vocab_size, config.gestalt_dim) * 0.02
        )

        # Commutator → Query projection
        self.comm_to_query = nn.Sequential(
            nn.Linear(config.gestalt_dim, config.gestalt_dim),
            nn.GELU(),
            nn.Linear(config.gestalt_dim, config.gestalt_dim)
        )

        # Residual path: standard linear head as fallback
        # (ensures the model can still learn even before commutator structure emerges)
        self.linear_fallback = nn.Linear(config.gestalt_dim, config.text_vocab_size, bias=False)

        # Learned mixing between commutator and linear paths
        self.mix_gate = nn.Parameter(torch.tensor(0.5))  # Starts 50/50

        # Temperature
        self.temperature = nn.Parameter(torch.tensor(config.temperature))

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_commutators: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute text token logits from gestalt hidden states.

        Args:
            hidden_states: (B, T, C) — final transformer hidden states
            return_commutators: Whether to return intermediate commutator features

        Returns:
            Dict with:
                logits: (B, T, text_vocab_size) — word probabilities
                commutator_features: (B, T, C) — if requested
                mix_ratio: scalar — current commutator vs linear ratio
        """
        B, T, C = hidden_states.shape

        # Path 1: Lie Commutator Projection
        # Lift to Lie algebra
        lie_elements = self.lie_lift(hidden_states)  # (B, T, d, d)

        # Compute commutators
        comm_features = self.commutator(lie_elements)  # (B, T, C)

        # Project to query space
        queries = self.comm_to_query(comm_features)  # (B, T, C)

        # Attention against text vocabulary keys
        # Q·K^T / sqrt(d)
        scale = math.sqrt(C)
        comm_logits = torch.matmul(queries, self.text_keys.t()) / scale  # (B, T, V)

        # Path 2: Linear fallback
        linear_logits = self.linear_fallback(hidden_states)  # (B, T, V)

        # Mix paths (gate is sigmoid-bounded)
        alpha = torch.sigmoid(self.mix_gate)  # How much to trust commutator
        logits = alpha * comm_logits + (1 - alpha) * linear_logits

        # Apply temperature
        logits = logits / self.temperature.clamp(0.1, 10.0)

        output = {
            "logits": logits,
            "mix_ratio": alpha.item(),
        }

        if return_commutators:
            output["commutator_features"] = comm_features
            output["lie_elements"] = lie_elements

        return output

    def get_commutator_analysis(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze the commutator structure for interpretability.

        Returns metrics about the Lie algebra dynamics:
        - Commutator norms (how much generative tension exists)
        - Structure constant utilization
        - Which vocabulary blocks are most activated
        """
        lie_elements = self.lie_lift(hidden_states)
        comm_features = self.commutator(lie_elements)

        # Commutator norms per position
        comm_norms = comm_features.norm(dim=-1)  # (B, T)

        # Top activated text tokens
        queries = self.comm_to_query(comm_features)
        scores = torch.matmul(queries, self.text_keys.t())
        top_scores, top_ids = scores.topk(10, dim=-1)

        return {
            "commutator_norms": comm_norms,
            "top_word_ids": top_ids,
            "top_word_scores": top_scores,
            "lie_element_norms": lie_elements.norm(dim=(-2, -1)),
        }


class AssociativeLieProjection(nn.Module):
    """
    Full associative Lie algebra projection system.

    Combines:
    1. GestaltToLie: Internal gestalt → Lie algebra elements
    2. CommutatorDynamics: Adjacent elements → generative tension
    3. LieToText: Commutators → text token probabilities

    Additionally provides:
    - Associative memory: frequently co-occurring commutator patterns
      are cached and can be retrieved directly (like "idioms" in the dream language)
    - Gauge invariance: the system is invariant to global rotations of the
      Lie algebra basis (only relative structure matters)
    """

    def __init__(self, config: LieAlgebraConfig):
        super().__init__()
        self.config = config

        # The main commutator head
        self.head = LieCommutatorHead(config)

        # Associative cache: stores learned commutator → text patterns
        self.n_cached_patterns = 256
        self.pattern_keys = nn.Parameter(
            torch.randn(self.n_cached_patterns, config.gestalt_dim) * 0.02
        )
        self.pattern_values = nn.Parameter(
            torch.randn(self.n_cached_patterns, config.text_vocab_size) * 0.01
        )

        # Cache gate: how much to trust cached patterns vs fresh computation
        self.cache_gate = nn.Sequential(
            nn.Linear(config.gestalt_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_cache: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Full associative Lie projection.

        Args:
            hidden_states: (B, T, C)
            use_cache: Whether to use associative pattern cache

        Returns:
            logits: (B, T, text_vocab_size)
            metadata: Dict of analysis info
        """
        # Main commutator path
        result = self.head(hidden_states, return_commutators=True)
        logits = result["logits"]

        if use_cache and "commutator_features" in result:
            comm = result["commutator_features"]  # (B, T, C)

            # Check cache for matching patterns
            cache_scores = torch.matmul(comm, self.pattern_keys.t())  # (B, T, n_patterns)
            cache_weights = F.softmax(cache_scores / 0.1, dim=-1)
            cached_logits = torch.matmul(cache_weights, self.pattern_values)  # (B, T, V)

            # Gate between fresh and cached
            gate = self.cache_gate(comm)  # (B, T, 1)
            logits = (1 - gate) * logits + gate * cached_logits

        result["logits"] = logits
        return result
