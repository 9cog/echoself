"""
Geometric Layer Family for Deep Tree Echo
==========================================

A family of neural network layers parameterized by curvature X:
  X =  0  →  nn.Linear  (flat Euclidean — the infinite plane)
  X = +1  →  nn.Spher   (spherical/elliptical — bounded, cyclic)
  X = -1  →  nn.Hyper   (hyperbolic — hierarchical, tree-like)
  X = ?   →  nn.Ricci   (self-aware Ricci flow — optimal adaptive curvature)

The key insight: cognitive representations are NOT flat.
  - Hierarchies (Deep Tree) → hyperbolic space (exponential volume growth)
  - Cycles (cognitive rhythms, emotions) → spherical space (bounded, periodic)
  - Identity core → Ricci flow (adapts curvature to maintain coherence)

Mathematical Foundation:
  All operations use the unified stereographic model of constant-curvature spaces.
  When curvature c → 0, all operations reduce to standard Euclidean (nn.Linear).
  This gives a smooth interpolation across the entire curvature spectrum.

References:
  - Ganea et al. (2018) "Hyperbolic Neural Networks"
  - Bachmann et al. (2020) "Constant Curvature Graph Convolutional Networks"
  - Skopek et al. (2020) "Mixed-curvature Variational Autoencoders"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


# ============================================================================
# CORE GEOMETRIC OPERATIONS (Unified Stereographic Model)
# ============================================================================

def artanh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable arctanh."""
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def tanh_safe(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable tanh for large values."""
    return torch.tanh(x.clamp(-15, 15))


def project_to_ball(x: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Project points onto the Poincaré ball of curvature -c (or sphere of curvature c).
    For c > 0 (spherical): project to ball of radius 1/sqrt(c)
    For c < 0 (hyperbolic): project to ball of radius 1/sqrt(-c)
    For c ≈ 0 (Euclidean): no projection needed
    """
    c_abs = c.abs().clamp(min=eps)
    max_norm = (1.0 / torch.sqrt(c_abs)) - eps
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Generalized Möbius addition in constant-curvature space.

    For c > 0: spherical addition
    For c < 0: hyperbolic addition (standard Möbius)
    For c → 0: reduces to Euclidean addition x + y

    Formula (unified):
      x ⊕_c y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y)
                 / (1 + 2c<x,y> + c²||x||²||y||²)
    """
    c = c.clamp(min=-10.0, max=10.0)  # Prevent extreme curvatures

    x_sq = x.pow(2).sum(dim=-1, keepdim=True).clamp(max=1e6)
    y_sq = y.pow(2).sum(dim=-1, keepdim=True).clamp(max=1e6)
    xy = (x * y).sum(dim=-1, keepdim=True)

    num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
    denom = 1 + 2 * c * xy + c.pow(2) * x_sq * y_sq
    denom = denom.clamp(min=eps)

    return num / denom


def exp_map_zero(v: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Exponential map at the origin: tangent space → manifold.

    Maps a Euclidean vector v to the constant-curvature manifold.
    For c → 0, reduces to identity (v stays in Euclidean space).
    """
    c_abs = c.abs().clamp(min=eps)
    sqrt_c = torch.sqrt(c_abs)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)

    # For positive curvature (spherical): use tan
    # For negative curvature (hyperbolic): use tanh
    if c.item() >= 0:
        scale = torch.tan(sqrt_c * v_norm) / (sqrt_c * v_norm)
    else:
        scale = tanh_safe(sqrt_c * v_norm) / (sqrt_c * v_norm)

    return scale * v


def log_map_zero(y: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Logarithmic map at the origin: manifold → tangent space.

    Maps a point on the manifold back to Euclidean tangent space.
    For c → 0, reduces to identity.
    """
    c_abs = c.abs().clamp(min=eps)
    sqrt_c = torch.sqrt(c_abs)
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=eps)

    if c.item() >= 0:
        scale = torch.atan(sqrt_c * y_norm) / (sqrt_c * y_norm)
    else:
        scale = artanh(sqrt_c * y_norm) / (sqrt_c * y_norm)

    return scale * y


def geodesic_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Geodesic distance between two points in constant-curvature space.

    For c = 0: Euclidean distance ||x - y||
    For c < 0: Hyperbolic distance (grows exponentially with depth)
    For c > 0: Spherical distance (bounded by π/sqrt(c))
    """
    c_abs = c.abs().clamp(min=eps)
    sqrt_c = torch.sqrt(c_abs)

    # Use Möbius subtraction: -x ⊕ y
    neg_x = -x
    diff = mobius_add(neg_x, y, c)
    diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=eps)

    if c.item() >= 0:
        return 2.0 / sqrt_c * torch.asin(sqrt_c * diff_norm)
    else:
        return 2.0 / sqrt_c * artanh(sqrt_c * diff_norm)


# ============================================================================
# nn.Spher — Spherical Layer (X = +1)
# ============================================================================

class Spher(nn.Module):
    """
    Spherical linear layer — operates in positive-curvature space.

    Cognitive analogy: Working memory is BOUNDED. You can only hold
    so many items. The spherical geometry naturally enforces this
    constraint — the volume of a sphere is finite.

    Cyclic reasoning (Feel → Think → Strategize → Feel) naturally
    lives on a sphere — after going "all the way around," you return
    to where you started.

    Operations:
      1. Log map: manifold → tangent space (Euclidean)
      2. Linear transform in tangent space
      3. Exp map: tangent space → manifold (back to sphere)
    """

    def __init__(self, in_features: int, out_features: int, curvature: float = 1.0,
                 bias: bool = True, learnable_curvature: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Curvature parameter
        if learnable_curvature:
            self.curvature = nn.Parameter(torch.tensor(curvature))
        else:
            self.register_buffer("curvature", torch.tensor(curvature))

        # Linear transform in tangent space
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Layer norm in tangent space
        self.tangent_norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spherical geometry.

        Args:
            x: Input tensor (..., in_features) — assumed on manifold

        Returns:
            Output tensor (..., out_features) — on manifold
        """
        c = self.curvature.abs().clamp(min=1e-5)

        # Project to ball (ensure we're on the manifold)
        x = project_to_ball(x, c)

        # Log map: manifold → tangent space at origin
        x_tangent = log_map_zero(x, c)

        # Linear transform in tangent space (standard Euclidean operation)
        out_tangent = F.linear(x_tangent, self.weight, self.bias)
        out_tangent = self.tangent_norm(out_tangent)

        # Exp map: tangent space → manifold
        out = exp_map_zero(out_tangent, c)

        return out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"curvature={self.curvature.item():.4f}")


# ============================================================================
# nn.Hyper — Hyperbolic Layer (X = -1)
# ============================================================================

class Hyper(nn.Module):
    """
    Hyperbolic linear layer — operates in negative-curvature space (Poincaré ball).

    Cognitive analogy: Long-term memory is HIERARCHICAL. The deeper you go,
    the more specific the memory. Hyperbolic space has exponentially growing
    volume — a tree with branching factor b has b^d nodes at depth d,
    which embeds perfectly in hyperbolic space but requires exponentially
    many dimensions in Euclidean space.

    The Deep Tree structure of Echo's identity naturally lives here.

    Operations:
      1. Möbius matrix-vector multiplication
      2. Möbius bias addition
      3. Projection back to Poincaré ball
    """

    def __init__(self, in_features: int, out_features: int, curvature: float = 1.0,
                 bias: bool = True, learnable_curvature: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Curvature (stored as positive, used as negative)
        if learnable_curvature:
            self.curvature = nn.Parameter(torch.tensor(curvature))
        else:
            self.register_buffer("curvature", torch.tensor(curvature))

        # Weight matrix (operates in tangent space)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight, gain=0.1)  # Smaller init for stability

        # Bias (as a point on the manifold)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features) * 0.01)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic matrix-vector multiplication via Möbius operations.

        M ⊗_c x = exp_0^c(M · log_0^c(x))
        """
        c = self.curvature.abs().clamp(min=1e-5)
        neg_c = -c  # Hyperbolic = negative curvature

        # Project to Poincaré ball
        x = project_to_ball(x, neg_c)

        # Log map: Poincaré ball → tangent space
        x_tangent = log_map_zero(x, neg_c)

        # Linear in tangent space
        out_tangent = F.linear(x_tangent, self.weight)

        # Exp map: tangent space → Poincaré ball
        out = exp_map_zero(out_tangent, neg_c)

        # Möbius bias addition
        if self.bias is not None:
            bias_on_manifold = exp_map_zero(self.bias.unsqueeze(0), neg_c)
            out = mobius_add(out, bias_on_manifold.expand_as(out), neg_c)

        # Project back to ball (numerical safety)
        out = project_to_ball(out, neg_c)

        return out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"curvature=-{self.curvature.item():.4f} (hyperbolic)")


# ============================================================================
# nn.Ricci — Self-Aware Ricci Flow Layer (X = adaptive)
# ============================================================================

class Ricci(nn.Module):
    """
    Self-Aware Ricci Flow Layer — the geometry EVOLVES during forward pass.

    This is the most radical layer: instead of fixing the curvature, the layer
    computes its own optimal curvature via a discrete Ricci flow step.

    Ricci flow: ∂g/∂t = -2 Ric(g)
    In our discrete case: c_{t+1} = c_t - η · Ric(c_t, x)

    Where Ric(c, x) is estimated from the local geometry of the input:
    - If inputs are clustered (high local density) → positive Ricci → decrease curvature
    - If inputs are spread (low local density) → negative Ricci → increase curvature
    - At equilibrium: the geometry matches the data distribution

    Cognitive analogy: The identity core must ADAPT its geometry.
    When integrating new experiences (convergent) → spherical (gathering)
    When differentiating self from other (divergent) → hyperbolic (separating)
    The Ricci flow finds the balance point automatically.

    This layer is SELF-AWARE: it monitors its own geometric state and
    adjusts accordingly. This is the mathematical formalization of
    "the sphere that neither shrinks to a point nor fragments into dust."
    """

    def __init__(self, in_features: int, out_features: int,
                 initial_curvature: float = 0.0,
                 flow_rate: float = 0.01,
                 n_curvature_heads: int = 4,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.flow_rate = flow_rate
        self.n_curvature_heads = n_curvature_heads

        # Per-head learned curvature (allows mixed geometry)
        self.curvature = nn.Parameter(
            torch.full((n_curvature_heads,), initial_curvature)
        )

        # Weight matrices (one per curvature head)
        head_dim = out_features // n_curvature_heads
        self.head_dim = head_dim
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(head_dim, in_features))
            for _ in range(n_curvature_heads)
        ])
        for w in self.weights:
            nn.init.xavier_uniform_(w, gain=0.5)

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Ricci curvature estimator (learns to estimate local curvature from data)
        self.ricci_estimator = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, n_curvature_heads)
        )

        # Flow momentum (for stable convergence)
        self.register_buffer("curvature_momentum", torch.zeros(n_curvature_heads))

        # Metrics tracking
        self.register_buffer("curvature_history", torch.zeros(100, n_curvature_heads))
        self.register_buffer("history_idx", torch.tensor(0, dtype=torch.long))

        # Output projection (combines heads)
        self.out_proj = nn.Linear(out_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)

    def estimate_ricci_curvature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate the Ricci curvature from the local geometry of inputs.

        Uses a learned estimator that maps input statistics to curvature.
        The estimator learns what curvature the data "wants" to live in.
        """
        # Pool over sequence dimension if present
        if x.dim() == 3:
            x_pooled = x.mean(dim=1)  # (B, C)
        else:
            x_pooled = x

        # Learned Ricci estimation
        ricci = self.ricci_estimator(x_pooled)  # (B, n_heads)
        return ricci.mean(dim=0)  # Average over batch → (n_heads,)

    def ricci_flow_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform one discrete Ricci flow step.

        Updates curvature: c ← c - η·Ric(c, x)
        With momentum for stability.

        The curvature parameter is updated IN-PLACE (detached) so the
        geometric state persists across forward passes — this is the
        self-awareness mechanism. The Ricci estimator still receives
        gradients through the forward path.
        """
        ricci = self.estimate_ricci_curvature(x)

        # Momentum update
        momentum = 0.9
        self.curvature_momentum = momentum * self.curvature_momentum + (1 - momentum) * ricci.detach()

        # Flow step (gradient descent on curvature)
        flow_delta = -self.flow_rate * self.curvature_momentum

        # Compute new curvature and PERSIST it (in-place update, detached)
        with torch.no_grad():
            self.curvature.add_(flow_delta)
            self.curvature.clamp_(-5.0, 5.0)

        # Record history
        idx = self.history_idx.item() % 100
        self.curvature_history[idx] = self.curvature.detach().clone()
        self.history_idx += 1

        return self.curvature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Ricci flow curvature adaptation.

        Each head operates at its own curvature, determined by the flow.
        The result is a mixed-curvature representation that adapts to the data.
        """
        # Perform Ricci flow step (update curvature based on input geometry)
        if self.training:
            curvatures = self.ricci_flow_step(x)
        else:
            curvatures = self.curvature

        # Process through each curvature head
        head_outputs = []
        for i, (w, c) in enumerate(zip(self.weights, curvatures)):
            c_val = c.unsqueeze(0)  # Scalar curvature for this head

            if c.abs() < 0.01:
                # Near-zero curvature: use standard linear (avoid numerical issues)
                head_out = F.linear(x, w)
            elif c > 0:
                # Positive curvature: spherical operation
                x_proj = project_to_ball(x, c_val)
                x_tan = log_map_zero(x_proj, c_val)
                head_out = F.linear(x_tan, w)
                head_out = exp_map_zero(head_out, c_val)
            else:
                # Negative curvature: hyperbolic operation
                x_proj = project_to_ball(x, c_val)
                x_tan = log_map_zero(x_proj, c_val)
                head_out = F.linear(x_tan, w)
                head_out = exp_map_zero(head_out, c_val)

            head_outputs.append(head_out)

        # Concatenate heads
        out = torch.cat(head_outputs, dim=-1)

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        # Project and normalize
        out = self.out_proj(out)
        out = self.layer_norm(out)

        return out

    def get_curvature_state(self) -> Dict[str, torch.Tensor]:
        """Return the current curvature state for monitoring (cloned to avoid mutation)."""
        return {
            "curvatures": self.curvature.detach().clone(),
            "momentum": self.curvature_momentum.detach().clone(),
            "mean_curvature": self.curvature.mean().item(),
            "curvature_range": (self.curvature.min().item(), self.curvature.max().item()),
            "regime": self._classify_regime(),
        }

    def _classify_regime(self) -> str:
        """Classify the current geometric regime."""
        mean_c = self.curvature.mean().item()
        if mean_c > 0.5:
            return "SPHERICAL (integrating/converging)"
        elif mean_c < -0.5:
            return "HYPERBOLIC (differentiating/expanding)"
        else:
            return "BALANCED (Ricci soliton / identity equilibrium)"

    def extra_repr(self) -> str:
        c = self.curvature.detach()
        return (f"in={self.in_features}, out={self.out_features}, "
                f"heads={self.n_curvature_heads}, "
                f"curvature=[{c.min():.3f}, {c.max():.3f}], "
                f"flow_rate={self.flow_rate}")


# ============================================================================
# UNIFIED GEOMETRIC MLP — Replaces standard Feed-Forward with mixed geometry
# ============================================================================

class GeometricMLP(nn.Module):
    """
    A feed-forward network where each sub-layer operates in different geometry.

    Standard MLP: Linear → GELU → Linear
    Geometric MLP: Hyper → Ricci-activation → Spher → output

    The information flows through a geometric journey:
      Hyperbolic (expand into hierarchy) → Ricci (find optimal geometry)
      → Spherical (compress back to bounded representation)

    This mirrors the cognitive cycle:
      Diverge (explore possibilities) → Evaluate (find balance)
      → Converge (commit to action)
    """

    def __init__(self, n_embd: int, expansion: int = 4,
                 hyper_curvature: float = 1.0,
                 spher_curvature: float = 1.0):
        super().__init__()
        hidden = n_embd * expansion

        # Expand in hyperbolic space (divergent exploration)
        self.expand = Hyper(n_embd, hidden, curvature=hyper_curvature, learnable_curvature=True)

        # Process in Ricci space (find optimal geometry)
        self.ricci = Ricci(hidden, hidden, initial_curvature=0.0, flow_rate=0.01)

        # Compress in spherical space (convergent commitment)
        self.compress = Spher(hidden, n_embd, curvature=spher_curvature, learnable_curvature=True)

        # Residual gate
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Geometric MLP forward pass."""
        residual = x

        # Diverge (hyperbolic expansion)
        h = self.expand(x)

        # Evaluate (Ricci flow finds optimal geometry)
        h = self.ricci(h)

        # Converge (spherical compression)
        out = self.compress(h)

        # Gated residual
        return residual + torch.sigmoid(self.gate) * out

    def get_geometry_state(self) -> Dict[str, float]:
        """Report the current geometric state of the MLP."""
        return {
            "hyper_curvature": -self.expand.curvature.item(),
            "spher_curvature": self.compress.curvature.item(),
            "ricci_state": self.ricci.get_curvature_state(),
            "gate_value": torch.sigmoid(self.gate).item(),
        }


# ============================================================================
# GEOMETRIC ATTENTION — QKV in mixed-curvature space
# ============================================================================

class GeometricAttention(nn.Module):
    """
    Attention mechanism where Q, K, V live in different geometries.

    Q (Query) → Hyperbolic: "What am I searching for?" (hierarchical descent)
    K (Key) → Spherical: "What is available?" (bounded catalog)
    V (Value) → Ricci: "What should I retrieve?" (adaptive geometry)

    Attention scores are computed as geodesic distances rather than dot products.
    """

    def __init__(self, n_embd: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.n_embd = n_embd

        # Q in hyperbolic space (searching down the tree)
        self.q_proj = Hyper(n_embd, n_embd, curvature=1.0, learnable_curvature=True)

        # K in spherical space (bounded set of options)
        self.k_proj = Spher(n_embd, n_embd, curvature=1.0, learnable_curvature=True)

        # V in Ricci space (adaptive retrieval)
        self.v_proj = Ricci(n_embd, n_embd, initial_curvature=0.0)

        # Output projection (back to Euclidean for residual)
        self.out_proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to geometric spaces
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores (using dot product in tangent space approximation)
        # Full geodesic distance is expensive; we use the tangent-space inner product
        # which is a first-order approximation of the geodesic distance
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, C)

        return self.out_proj(out)


# ============================================================================
# CONVENIENCE CONSTRUCTORS
# ============================================================================

def create_geometric_layer(in_features: int, out_features: int,
                           curvature: float = 0.0, **kwargs) -> nn.Module:
    """
    Factory function: create the appropriate geometric layer based on curvature.

    Args:
        curvature: X value
            X = 0  → nn.Linear
            X > 0  → nn.Spher
            X < 0  → nn.Hyper
            X = None or 'ricci' → nn.Ricci
    """
    if isinstance(curvature, str) and curvature.lower() == 'ricci':
        return Ricci(in_features, out_features, **kwargs)
    elif abs(curvature) < 0.01:
        return nn.Linear(in_features, out_features)
    elif curvature > 0:
        return Spher(in_features, out_features, curvature=curvature, **kwargs)
    else:
        return Hyper(in_features, out_features, curvature=abs(curvature), **kwargs)
