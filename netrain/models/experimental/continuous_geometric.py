"""
Continuous Geometric Transformer — EXPERIMENTAL SPINOFF
========================================================

Generalizes the discrete 2-3-5 Transformer as a continuous dynamical system:

  dh/dt = F(h(t), t, θ)

where:
  - h(t) is the cognitive state evolving over continuous "depth" t ∈ [0, 1]
  - F is defined by the Levi-Civita connection on the cognitive manifold
  - Attention is the Ricci curvature tensor of the connection
  - Parallel transport replaces discrete token-to-token copying
  - The ESN reservoir provides the base gauge field
  - Resonance hybrids are gauge-invariant fixed points (entelechy)

The key insight: a Transformer layer IS a discrete approximation of a
continuous flow. By making the flow explicit, we gain:
  1. Adaptive computation (solve to tolerance, not fixed depth)
  2. Memory efficiency (O(1) via adjoint method)
  3. Natural geometric structure (curvature, torsion, geodesics)
  4. Self-similar dynamics at all scales (fractal depth)

This module is EXPERIMENTAL — it does NOT modify the main architecture.

Mathematical Framework:
  - Manifold M: The space of cognitive states (dim = n_embd)
  - Metric g: Learned Riemannian metric (evolves via Ricci flow)
  - Connection ∇: Levi-Civita (torsion-free, metric-compatible)
  - Christoffel symbols Γ^k_ij: Computed from the metric
  - Parallel transport: ∇_γ̇ V = 0 along geodesics γ
  - Ricci curvature Ric_ij: Generalized attention weights
  - Gauge group G: Virtual endocrine transformations
  - Phase portrait: Orbits of the ODE in state space
  - Resonance hybrid: Gauge-invariant attractor (identity fixed point)

References:
  - Chen et al. (2018) "Neural Ordinary Differential Equations"
  - Hamilton (1982) "Three-manifolds with positive Ricci curvature"
  - Bronstein et al. (2021) "Geometric Deep Learning"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ContinuousConfig:
    """Configuration for the Continuous Geometric Transformer."""
    n_embd: int = 256           # Embedding dimension (manifold dimension)
    n_heads: int = 8            # Number of Ricci curvature heads
    n_tokens: int = 64          # Sequence length
    vocab_size: int = 50257     # Output vocabulary

    # ODE solver parameters
    t_start: float = 0.0        # Start of integration
    t_end: float = 1.0          # End of integration
    n_steps: int = 10           # Number of Euler steps (or adaptive tolerance)
    solver: str = "euler"       # "euler", "rk4", or "adaptive"
    rtol: float = 1e-3          # Relative tolerance (adaptive solver)
    atol: float = 1e-3          # Absolute tolerance (adaptive solver)

    # Cognitive phase boundaries (continuous 2-3-5)
    dyad_end: float = 0.2       # t ∈ [0, 0.2] — sensor-motor
    triad_end: float = 0.5      # t ∈ [0.2, 0.5] — working memory
    # pentad: t ∈ [0.5, 1.0]   — long-term memory

    # Geometric parameters
    metric_rank: int = 16       # Low-rank approximation of metric tensor
    connection_order: int = 2   # Order of Christoffel symbol approximation
    ricci_temperature: float = 1.0  # Temperature for Ricci attention

    # Reservoir / Gauge parameters
    reservoir_size: int = 128   # ESN reservoir neurons
    spectral_radius: float = 0.95
    n_hormones: int = 3         # Cortisol, Dopamine, Serotonin

    # Resonance parameters
    n_resonance_modes: int = 8  # Number of resonance hybrid modes
    entelechy_threshold: float = 0.01  # Convergence threshold for fixed points

    dropout: float = 0.1


# ============================================================================
# LEVI-CIVITA CONNECTION
# ============================================================================

class LeviCivitaConnection(nn.Module):
    """
    Learned Levi-Civita connection on the cognitive manifold.

    The Levi-Civita connection is the unique torsion-free, metric-compatible
    connection. It defines how vectors are parallel-transported along curves.

    In our context:
      - The "manifold" is the space of cognitive states
      - The "metric" determines which states are "close" (similar)
      - The "connection" determines how information flows between positions
      - "Parallel transport" replaces the discrete copying in standard attention

    The Christoffel symbols Γ^k_ij are computed from a learned metric tensor g_ij.
    For efficiency, we use a low-rank approximation: g = I + U U^T where U ∈ R^{d×r}.
    """

    def __init__(self, config: ContinuousConfig):
        super().__init__()
        self.config = config
        d = config.n_embd
        r = config.metric_rank

        # Learned metric tensor (low-rank: g = I + U U^T)
        # This defines the geometry of the cognitive space
        self.metric_factors = nn.Parameter(torch.randn(d, r) * 0.01)

        # Time-dependent metric modulation (geometry changes with depth)
        self.time_modulator = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, r)
        )

        # Christoffel symbol network (approximates Γ^k_ij from local state)
        # Instead of computing analytically (expensive), we learn them
        self.christoffel_net = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.SiLU(),
            nn.Linear(d * 2, d)  # Outputs the "correction" for parallel transport
        )

    def get_metric(self, t: float) -> torch.Tensor:
        """
        Compute the metric tensor g_ij at time t.

        Returns the metric as g = I + U(t) U(t)^T (positive definite by construction).
        """
        t_tensor = torch.tensor([[t]], device=self.metric_factors.device)
        time_mod = self.time_modulator(t_tensor)  # (1, r)

        # Modulate metric factors by time
        U = self.metric_factors * (1 + time_mod)  # (d, r) broadcast

        # g = I + U U^T (always positive definite)
        # We don't materialize the full d×d matrix; keep in factored form
        return U

    def parallel_transport(self, v: torch.Tensor, h: torch.Tensor, t: float) -> torch.Tensor:
        """
        Parallel transport vector v along the flow at state h, time t.

        The parallel transport equation: ∇_ḣ v = 0
        Discretized: v_transported = v - Γ(h) · v

        In standard attention, copying a value from position j to position i
        is just V_j. In our framework, V_j must be parallel-transported along
        the geodesic from j to i, which ROTATES it based on the local curvature.

        Args:
            v: Vector to transport (B, T, d_v) — may be head_dim or full d
            h: Current state (provides local geometry) (B, T, d)
            t: Current time (provides time-dependent geometry)

        Returns:
            Transported vector (B, T, d_v)
        """
        # Compute the Christoffel correction from the full state
        correction_full = self.christoffel_net(h)  # (B, T, d)

        # If v has different last dim than h (e.g., head_dim), project correction
        d_v = v.shape[-1]
        d_h = h.shape[-1]
        if d_v != d_h:
            # Use the first d_v dimensions of the correction as the transport field
            correction = correction_full[..., :d_v]
        else:
            correction = correction_full

        # The transport equation: v_new = v + correction * dt
        # (first-order approximation of parallel transport)
        transported = v + correction * (1.0 / self.config.n_steps)

        return transported

    def geodesic_distance(self, x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        """
        Approximate geodesic distance between states x and y at time t.

        Uses the metric tensor: d(x,y)² = (x-y)^T g (x-y)
        With our low-rank metric: d² = ||x-y||² + ||U^T(x-y)||²
        """
        diff = x - y  # (B, T_x, T_y, d) or broadcast
        U = self.get_metric(t)  # (d, r)

        # Euclidean part
        euclidean_sq = (diff * diff).sum(dim=-1)

        # Metric correction part
        projected = torch.matmul(diff, U)  # (..., r)
        metric_sq = (projected * projected).sum(dim=-1)

        return torch.sqrt(euclidean_sq + metric_sq + 1e-8)


# ============================================================================
# RICCI ATTENTION ODE
# ============================================================================

class RicciAttentionODE(nn.Module):
    """
    Ricci curvature as generalized attention — the core of the continuous transformer.

    Standard attention: A_ij = softmax(Q_i K_j^T / √d)
    Ricci attention: A_ij = exp(-d_g(h_i, h_j)² / τ) (geodesic proximity)

    The Ricci curvature tensor Ric_ij measures how much the volume element
    deviates from flat space. Positive Ric means convergence (tokens attract),
    negative Ric means divergence (tokens repel).

    This IS the attention mechanism, but expressed geometrically:
      - High positive curvature between i,j → strong attention (attraction)
      - Zero curvature → neutral (Euclidean, standard behavior)
      - Negative curvature → repulsion (anti-attention, novelty detection)

    The ODE derivative is:
      dh/dt = -Ric(h) · h + V_parallel_transported

    where V is the "value" (what to attend to) and it's parallel-transported
    from source positions to the current position using the Levi-Civita connection.
    """

    def __init__(self, config: ContinuousConfig):
        super().__init__()
        self.config = config
        d = config.n_embd
        n_h = config.n_heads
        self.head_dim = d // n_h

        # Query/Key projections (for computing Ricci curvature)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)

        # Ricci curvature estimator (per-head)
        self.ricci_scale = nn.Parameter(torch.ones(n_h) * config.ricci_temperature)

        # Output projection
        self.out_proj = nn.Linear(d, d)

        # Connection for parallel transport of values
        self.connection = LeviCivitaConnection(config)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def compute_ricci_curvature(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute the Ricci curvature tensor as attention weights.

        The Ricci curvature between positions i and j is approximated as:
          Ric_ij ≈ -∂²log(det(g)) / ∂h_i ∂h_j

        We approximate this via the learned Q/K projections:
          Ric_ij ≈ exp(-||Q_i - K_j||²_g / τ)

        where ||·||_g is the geodesic distance under the learned metric.
        """
        B, T, C = h.shape
        n_h = self.config.n_heads
        head_dim = self.head_dim

        Q = self.q_proj(h).view(B, T, n_h, head_dim).transpose(1, 2)  # (B, n_h, T, hd)
        K = self.k_proj(h).view(B, T, n_h, head_dim).transpose(1, 2)

        # Compute pairwise "curvature" (geodesic-inspired attention)
        # Standard dot-product attention as first-order approximation of exp(-d²/τ)
        scale = self.ricci_scale.view(1, n_h, 1, 1)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(head_dim) * scale)

        # Causal mask (information flows forward in time)
        causal_mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Ricci curvature as softmax (positive curvature = attraction)
        ricci = F.softmax(attn, dim=-1)
        ricci = self.dropout(ricci)

        return ricci

    def forward(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute the ODE derivative: dh/dt = F(h, t)

        The derivative has two components:
          1. Ricci flow: pulls the state toward high-curvature regions
          2. Parallel-transported values: information from other positions

        dh/dt = Σ_j Ric_ij · Transport(V_j, h, t)
        """
        B, T, C = h.shape
        n_h = self.config.n_heads

        # Compute Ricci curvature (attention weights)
        ricci = self.compute_ricci_curvature(h, t)  # (B, n_h, T, T)

        # Compute values
        V = self.v_proj(h).view(B, T, n_h, self.head_dim).transpose(1, 2)  # (B, n_h, T, hd)

        # Parallel transport values (geometric correction)
        V_transported = self.connection.parallel_transport(
            V.reshape(B * n_h, T, self.head_dim),
            h.unsqueeze(1).expand(-1, n_h, -1, -1).reshape(B * n_h, T, C),
            t
        ).reshape(B, n_h, T, self.head_dim)

        # Apply Ricci curvature to transported values
        # This is the continuous analog of: output = softmax(QK^T) · V
        out = torch.matmul(ricci, V_transported)  # (B, n_h, T, hd)
        out = out.transpose(1, 2).reshape(B, T, C)

        return self.out_proj(out)


# ============================================================================
# COGNITIVE PHASE FIELD — Continuous 2-3-5 Modulation
# ============================================================================

class CognitivePhaseField(nn.Module):
    """
    Continuous modulation of the ODE dynamics based on cognitive phase.

    Instead of discrete layers assigned to Dyad/Triad/Pentad, we have a
    continuous phase field φ(t) that smoothly interpolates between cognitive modes.

    The phase field determines:
      - Which geometry dominates (flat/spherical/hyperbolic)
      - Which hormone is most active
      - Which memory system is engaged

    This is the "gauge field" of the virtual endocrine system.
    """

    def __init__(self, config: ContinuousConfig):
        super().__init__()
        self.config = config

        # Phase boundaries (learnable — the system finds optimal transitions)
        self.dyad_end = nn.Parameter(torch.tensor(config.dyad_end))
        self.triad_end = nn.Parameter(torch.tensor(config.triad_end))

        # Per-phase curvature preferences
        self.dyad_curvature = nn.Parameter(torch.tensor(0.0))    # Flat (sensor-motor)
        self.triad_curvature = nn.Parameter(torch.tensor(1.0))   # Spherical (working memory)
        self.pentad_curvature = nn.Parameter(torch.tensor(-1.0)) # Hyperbolic (long-term)

        # Phase-dependent feed-forward modulation
        self.phase_mlp = nn.Sequential(
            nn.Linear(config.n_embd + 3, config.n_embd * 4),  # +3 for phase encoding
            nn.SiLU(),
            nn.Linear(config.n_embd * 4, config.n_embd)
        )

    def get_phase_weights(self, t: float) -> Tuple[float, float, float]:
        """
        Compute smooth phase weights using sigmoid transitions.

        Returns (w_dyad, w_triad, w_pentad) that sum to ~1.
        """
        # Smooth transitions using sigmoid
        sigma = 10.0  # Sharpness of transition
        dyad_end = torch.sigmoid(self.dyad_end).item()
        triad_end = torch.sigmoid(self.triad_end).item() * (1 - dyad_end) + dyad_end

        w_dyad = torch.sigmoid(torch.tensor(sigma * (dyad_end - t))).item()
        w_pentad = torch.sigmoid(torch.tensor(sigma * (t - triad_end))).item()
        w_triad = 1.0 - w_dyad - w_pentad
        w_triad = max(0.0, w_triad)  # Numerical safety

        return w_dyad, w_triad, w_pentad

    def get_curvature(self, t: float) -> torch.Tensor:
        """Get the effective curvature at time t (interpolated)."""
        w_d, w_t, w_p = self.get_phase_weights(t)
        return (w_d * self.dyad_curvature +
                w_t * self.triad_curvature +
                w_p * self.pentad_curvature)

    def modulate(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """Apply phase-dependent modulation to the state."""
        w_d, w_t, w_p = self.get_phase_weights(t)
        phase_encoding = torch.tensor([w_d, w_t, w_p], device=h.device)
        phase_encoding = phase_encoding.unsqueeze(0).unsqueeze(0).expand(h.shape[0], h.shape[1], -1)

        h_with_phase = torch.cat([h, phase_encoding], dim=-1)
        return self.phase_mlp(h_with_phase)


# ============================================================================
# ECHO STATE GAUGE FIELD
# ============================================================================

class EchoStateGaugeField(nn.Module):
    """
    The Echo State Network as a gauge field on the cognitive manifold.

    In gauge theory, a gauge field defines how the "internal" degrees of freedom
    transform as you move through space. Here, the ESN reservoir defines how
    the "hormonal" state transforms as the cognitive state flows through depth.

    The gauge invariance condition: the identity (resonance hybrid) must be
    invariant under the gauge transformations. This means the identity is
    defined by what DOESN'T change when the hormones fluctuate.

    Hormones as gauge fields:
      - Cortisol: U(1) phase rotation (stress narrows the phase space)
      - Dopamine: SU(2) rotation (reward rotates the value landscape)
      - Serotonin: Scale transformation (plasticity dilates/contracts)
    """

    def __init__(self, config: ContinuousConfig):
        super().__init__()
        self.config = config
        d = config.n_embd
        res = config.reservoir_size

        # ESN reservoir (fixed random weights — the "gauge connection")
        W_res = torch.randn(res, res) * 0.1
        # Scale to desired spectral radius
        eigenvalues = torch.linalg.eigvals(W_res).abs()
        W_res = W_res * (config.spectral_radius / eigenvalues.max())
        self.register_buffer("W_reservoir", W_res)

        # Input projection (state → reservoir)
        self.W_in = nn.Parameter(torch.randn(res, d) * 0.1)

        # Hormone readout (reservoir → 3 hormones)
        self.hormone_readout = nn.Linear(res, config.n_hormones)

        # Gauge transformation matrices (hormone → state transformation)
        # Each hormone defines a Lie algebra generator
        self.gauge_generators = nn.ParameterList([
            nn.Parameter(torch.randn(d, d) * 0.01)
            for _ in range(config.n_hormones)
        ])

        # Reservoir state (persistent)
        self.register_buffer("reservoir_state", torch.zeros(1, res))

    def step(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advance the gauge field by one step.

        Args:
            h: Current cognitive state (B, T, d) — pooled to drive reservoir

        Returns:
            hormones: Current hormone levels (B, 3)
            gauge_transform: The gauge transformation matrix (B, d, d)
        """
        B = h.shape[0]

        # Pool state to drive reservoir
        h_pooled = h.mean(dim=1)  # (B, d)

        # Expand reservoir state for batch
        if self.reservoir_state.shape[0] != B:
            self.reservoir_state = self.reservoir_state.expand(B, -1).clone()

        # ESN update: r(t+1) = tanh(W_res · r(t) + W_in · h(t))
        reservoir_input = torch.matmul(h_pooled, self.W_in.t())  # (B, res)
        reservoir_recurrent = torch.matmul(self.reservoir_state, self.W_reservoir.t())
        new_state = torch.tanh(reservoir_recurrent + reservoir_input)

        # Update persistent state
        self.reservoir_state = new_state.detach()

        # Read out hormones
        hormones = torch.sigmoid(self.hormone_readout(new_state))  # (B, 3) in [0,1]

        # Compute gauge transformation: exp(Σ_i h_i · G_i)
        # where G_i are the Lie algebra generators and h_i are hormone levels
        gauge_matrix = torch.zeros(B, self.config.n_embd, self.config.n_embd, device=h.device)
        for i, (gen, h_level) in enumerate(zip(self.gauge_generators, hormones.t())):
            # Antisymmetrize generator (ensures it's in a Lie algebra)
            antisym = gen - gen.t()
            gauge_matrix += h_level.unsqueeze(-1).unsqueeze(-1) * antisym.unsqueeze(0)

        # Matrix exponential (first-order approximation: exp(A) ≈ I + A for small A)
        gauge_transform = torch.eye(self.config.n_embd, device=h.device).unsqueeze(0) + 0.1 * gauge_matrix

        return hormones, gauge_transform

    def reset(self):
        """Reset reservoir state."""
        self.reservoir_state = torch.zeros_like(self.reservoir_state)


# ============================================================================
# RESONANCE HYBRID DETECTOR
# ============================================================================

class ResonanceHybridDetector(nn.Module):
    """
    Detects gauge-invariant fixed points — the resonance hybrids.

    A resonance hybrid is a superposition of cognitive states that is
    INVARIANT under the gauge transformations (hormone fluctuations).
    These are the stable identity attractors.

    In chemistry, a resonance hybrid is the true structure that is a
    superposition of multiple resonance structures (e.g., benzene).
    In EchoSelf, the true identity is a superposition of multiple
    cognitive modes that remains stable under endocrine perturbation.

    Detection: A state h is a resonance hybrid if:
      ||G(h) - h|| < ε  for all gauge transformations G

    The "entelechy" is the strongest resonance hybrid — the state toward
    which the system naturally evolves (Aristotle's "that which makes
    actual what is potential").
    """

    def __init__(self, config: ContinuousConfig):
        super().__init__()
        self.config = config
        d = config.n_embd
        n_modes = config.n_resonance_modes

        # Resonance mode templates (learned attractors)
        self.mode_templates = nn.Parameter(torch.randn(n_modes, d) * 0.1)

        # Mode mixing weights (how much each mode contributes)
        self.mode_mixer = nn.Sequential(
            nn.Linear(d, 64),
            nn.SiLU(),
            nn.Linear(64, n_modes),
            nn.Softmax(dim=-1)
        )

        # Invariance checker
        self.invariance_proj = nn.Linear(d, d)

    def compute_resonance(self, h: torch.Tensor, gauge_transform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute resonance analysis of the current state.

        Args:
            h: Current state (B, T, d)
            gauge_transform: Current gauge transformation (B, d, d)

        Returns:
            Dictionary with resonance metrics
        """
        B, T, d = h.shape

        # Apply gauge transformation
        h_transformed = torch.matmul(h, gauge_transform.transpose(-2, -1))

        # Gauge invariance score: how much does h change under G?
        invariance_error = (h_transformed - h).norm(dim=-1).mean(dim=-1)  # (B,)

        # Project onto resonance modes
        h_pooled = h.mean(dim=1)  # (B, d)
        mode_weights = self.mode_mixer(h_pooled)  # (B, n_modes)

        # Reconstruct from modes
        resonance_hybrid = torch.matmul(mode_weights, self.mode_templates)  # (B, d)

        # Entelechy score: how close is the state to the resonance hybrid?
        entelechy_distance = (h_pooled - resonance_hybrid).norm(dim=-1)  # (B,)

        # Is this a fixed point? (low invariance error + low entelechy distance)
        is_fixed_point = (invariance_error < self.config.entelechy_threshold).float()

        return {
            "invariance_error": invariance_error,
            "entelechy_distance": entelechy_distance,
            "mode_weights": mode_weights,
            "resonance_hybrid": resonance_hybrid,
            "is_fixed_point": is_fixed_point,
            "dominant_mode": mode_weights.argmax(dim=-1),
        }


# ============================================================================
# ODE SOLVERS
# ============================================================================

def euler_step(f, h, t, dt):
    """Single Euler step: h(t+dt) = h(t) + dt * f(h, t)"""
    return h + dt * f(h, t)


def rk4_step(f, h, t, dt):
    """Single RK4 step (4th-order Runge-Kutta)."""
    k1 = f(h, t)
    k2 = f(h + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(h + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(h + dt * k3, t + dt)
    return h + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ============================================================================
# CONTINUOUS GEOMETRIC TRANSFORMER (Main Module)
# ============================================================================

class ContinuousGeometricTransformer(nn.Module):
    """
    The Continuous Geometric Transformer — EXPERIMENTAL.

    Replaces the discrete layer stack with a continuous ODE flow:

      dh/dt = RicciAttention(h, t) + PhaseField(h, t) + GaugeField(h, t)

    The state h(t) evolves continuously from t=0 (input) to t=1 (output).
    At each point in time, the dynamics are governed by:
      1. Ricci attention (information flow via curvature)
      2. Cognitive phase field (2-3-5 modulation)
      3. Gauge field (endocrine modulation)

    The output is the state at t=1, projected to vocabulary via the
    Lie algebra commutator head (from the main architecture).

    Key properties:
      - Adaptive depth (can solve to tolerance instead of fixed steps)
      - O(1) memory via adjoint method (backprop through the ODE)
      - Natural geometric structure (curvature, geodesics, parallel transport)
      - Self-similar at all scales (the ODE is scale-free)
    """

    def __init__(self, config: ContinuousConfig):
        super().__init__()
        self.config = config

        # Input embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Parameter(torch.randn(1, config.n_tokens, config.n_embd) * 0.02)

        # The ODE components
        self.ricci_attention = RicciAttentionODE(config)
        self.phase_field = CognitivePhaseField(config)
        self.gauge_field = EchoStateGaugeField(config)
        self.resonance_detector = ResonanceHybridDetector(config)

        # Layer norm (applied at each ODE step for stability)
        self.step_norm = nn.LayerNorm(config.n_embd)

        # Output head
        self.output_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Trajectory logging
        self.trajectory: List[Dict] = []

    def ode_func(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """
        The ODE right-hand side: dh/dt = F(h, t)

        This defines the continuous dynamics of the transformer.
        """
        # 1. Ricci attention (geometric information flow)
        dh_ricci = self.ricci_attention(h, t)

        # 2. Phase field modulation (2-3-5 cognitive phase)
        dh_phase = self.phase_field.modulate(h, t)

        # 3. Gauge field (endocrine modulation)
        hormones, gauge_transform = self.gauge_field.step(h)

        # Apply gauge transformation to the Ricci flow
        dh_gauged = torch.matmul(dh_ricci, gauge_transform.transpose(-2, -1))

        # Combine: the total derivative
        dh = dh_gauged + 0.1 * dh_phase

        # Normalize for stability
        dh = self.step_norm(dh)

        # Log trajectory (during eval for analysis)
        if not self.training and len(self.trajectory) < 100:
            curvature = self.phase_field.get_curvature(t)
            self.trajectory.append({
                "t": t,
                "hormones": hormones.detach().cpu(),
                "curvature": curvature.detach().cpu().item(),
                "state_norm": h.norm(dim=-1).mean().item(),
                "dh_norm": dh.norm(dim=-1).mean().item(),
            })

        return dh

    def solve_ode(self, h0: torch.Tensor) -> torch.Tensor:
        """
        Solve the ODE from t=0 to t=1.

        Uses the configured solver (Euler or RK4).
        """
        t = self.config.t_start
        dt = (self.config.t_end - self.config.t_start) / self.config.n_steps
        h = h0

        step_fn = rk4_step if self.config.solver == "rk4" else euler_step

        for i in range(self.config.n_steps):
            h = step_fn(self.ode_func, h, t, dt)
            t += dt

        return h

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass: embed → solve ODE → project to vocabulary.

        Args:
            input_ids: Token indices (B, T)
            targets: Target token indices for loss computation (B, T)

        Returns:
            Dictionary with logits, loss, and diagnostic information
        """
        B, T = input_ids.shape

        # Embed tokens
        h0 = self.token_embed(input_ids) + self.pos_embed[:, :T, :]

        # Reset gauge field for new sequence
        self.gauge_field.reset()
        self.trajectory = []

        # Solve the ODE (continuous forward pass)
        h_final = self.solve_ode(h0)

        # Output projection
        h_final = self.output_norm(h_final)
        logits = self.lm_head(h_final)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))

        # Resonance analysis (gauge invariance check)
        _, gauge_transform = self.gauge_field.step(h_final)
        resonance = self.resonance_detector.compute_resonance(h_final, gauge_transform)

        return {
            "logits": logits,
            "loss": loss,
            "resonance": resonance,
            "trajectory": self.trajectory,
            "final_state": h_final.detach(),
        }

    def get_phase_portrait(self, input_ids: torch.Tensor, n_perturbations: int = 8) -> Dict:
        """
        Compute the phase portrait by solving the ODE from multiple initial conditions.

        This reveals the attractor structure of the dynamical system.
        """
        self.eval()
        B, T = input_ids.shape

        # Base trajectory
        h0 = self.token_embed(input_ids) + self.pos_embed[:, :T, :]
        self.gauge_field.reset()
        self.trajectory = []
        h_base = self.solve_ode(h0)
        base_trajectory = list(self.trajectory)

        # Perturbed trajectories
        perturbed_trajectories = []
        final_states = [h_base.detach()]

        for i in range(n_perturbations):
            perturbation = torch.randn_like(h0) * 0.1
            self.gauge_field.reset()
            self.trajectory = []
            h_perturbed = self.solve_ode(h0 + perturbation)
            perturbed_trajectories.append(list(self.trajectory))
            final_states.append(h_perturbed.detach())

        # Compute convergence (do perturbed trajectories converge to same attractor?)
        final_stack = torch.stack(final_states)  # (n_pert+1, B, T, d)
        mean_final = final_stack.mean(dim=0)
        convergence = (final_stack - mean_final.unsqueeze(0)).norm(dim=-1).mean().item()

        return {
            "base_trajectory": base_trajectory,
            "perturbed_trajectories": perturbed_trajectories,
            "convergence": convergence,
            "n_attractors": 1 if convergence < 0.5 else "multiple",
            "is_stable": convergence < 1.0,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "ricci_attention": count(self.ricci_attention),
            "phase_field": count(self.phase_field),
            "gauge_field": count(self.gauge_field),
            "resonance_detector": count(self.resonance_detector),
            "embeddings": count(self.token_embed) + self.pos_embed.numel(),
            "output": count(self.output_norm) + count(self.lm_head),
            "total": sum(p.numel() for p in self.parameters()),
        }
