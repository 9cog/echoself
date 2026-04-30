"""
Cognitive Grip Monitor
======================

Aggregates signals from all subsystems into a 7-dimensional *grip tensor*
(one entry per persona dimension) plus two scalar summary indices:

  learning_capacity_index      — how much headroom the current topology has
  inference_effectiveness_index — how well that capacity is being exploited

Signals consumed
----------------
* train_loss / val_loss and their moving windows   (MetricsTracker)
* per-block gradient-norm vector                   (model.get_topology_state)
* IntrospectionNode depth metrics                  (IntrospectionNode)
* EchobeatNode stream-coupling norms               (EchobeatNode)
* MembraneNode permeability profile                (MembraneNode)
* OpponentProcess balances                         (RelevanceRealizationEngine)

Computed quantities
-------------------
* gradient_SNR          = mean(grad_norms) / (std(grad_norms) + ε)
* lyapunov_index        = rolling mean of IntrospectionNode divergence @ depth-2
* capacity_effectiveness_ratio = exp(val_loss) / exp(min_val_loss_seen)
* grip[d] for each persona dimension d
* grip_velocity         = Δgrip / Δsteps  (gates topology mutations)
"""

from __future__ import annotations

import math
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Persona dimensions (must match nanecho_config.json order)
# ---------------------------------------------------------------------------
PERSONA_DIMS = [
    "cognitive",
    "introspective",
    "adaptive",
    "recursive",
    "synergistic",
    "holographic",
    "neural_symbolic",
]  # 7 dimensions

# Per-dimension weights for the grip formula components.
# α  = capacity/effectiveness contribution
# β  = lyapunov contribution
# γ  = phase-readiness contribution
_DIM_ALPHA = {
    "cognitive":      0.40,
    "introspective":  0.20,
    "adaptive":       0.35,
    "recursive":      0.25,
    "synergistic":    0.30,
    "holographic":    0.20,
    "neural_symbolic": 0.30,
}
_DIM_BETA = {
    "cognitive":      0.30,
    "introspective":  0.50,
    "adaptive":       0.25,
    "recursive":      0.40,
    "synergistic":    0.20,
    "holographic":    0.30,
    "neural_symbolic": 0.25,
}
_DIM_GAMMA = {
    "cognitive":      0.30,
    "introspective":  0.30,
    "adaptive":       0.40,
    "recursive":      0.35,
    "synergistic":    0.50,
    "holographic":    0.50,
    "neural_symbolic": 0.45,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class CognitiveGripMonitor:
    """Instrument panel synthesising all training signals into grip tensors.

    Parameters
    ----------
    window_size : int
        Rolling window length for all moving averages (default 100).
    lyapunov_window : int
        Dedicated window for the Lyapunov rolling mean (default 50).
    """

    def __init__(self, window_size: int = 100, lyapunov_window: int = 50) -> None:
        self.window_size = window_size
        self.lyapunov_window = lyapunov_window

        # Loss history
        self._train_losses: deque = deque(maxlen=window_size)
        self._val_losses: deque = deque(maxlen=window_size)
        self._min_val_loss: float = float("inf")

        # Gradient norms per block
        self._grad_norms: deque = deque(maxlen=window_size)

        # IntrospectionNode divergence at depth-2
        self._lyapunov_vals: deque = deque(maxlen=lyapunov_window)

        # Grip history (for velocity computation)
        self._grip_history: deque = deque(maxlen=window_size)
        self._step_history: deque = deque(maxlen=window_size)
        self._global_step: int = 0

        # Phase readiness scores (set externally by PhaseSequencer)
        self.phase_readiness: Dict[str, float] = {}
        self.current_phase: str = "basic_awareness"

        # OpponentProcess balances (set externally)
        self.opponent_balances: Dict[str, float] = {}

        # Latest grip vector
        self.grip: np.ndarray = np.full(len(PERSONA_DIMS), 0.5)

        # Cached secondary indices
        self.gradient_SNR: float = 0.0
        self.lyapunov_index: float = 0.0
        self.capacity_effectiveness_ratio: float = 1.0
        self.learning_capacity_index: float = 0.5
        self.inference_effectiveness_index: float = 0.5

    # ------------------------------------------------------------------
    # Recording methods (called by the trainer)
    # ------------------------------------------------------------------

    def record_train_loss(self, loss: float, step: int) -> None:
        self._train_losses.append(loss)
        self._global_step = step

    def record_val_loss(self, loss: float) -> None:
        self._val_losses.append(loss)
        if loss < self._min_val_loss:
            self._min_val_loss = loss

    def record_grad_norms(self, grad_norms: List[float]) -> None:
        """Accept a list of per-block gradient norms."""
        if grad_norms:
            self._grad_norms.append(grad_norms)

    def record_introspection(self, depth_metrics: List[Dict[str, float]]) -> None:
        """Accept IntrospectionNode metrics list indexed by depth."""
        if len(depth_metrics) >= 3:
            self._lyapunov_vals.append(float(depth_metrics[2].get("divergence", 0.0)))
        elif len(depth_metrics) >= 1:
            self._lyapunov_vals.append(float(depth_metrics[-1].get("divergence", 0.0)))

    def record(
        self,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        grad_norms: Optional[List[float]] = None,
        introspection_depths: Optional[List[Dict[str, float]]] = None,
        step: Optional[int] = None,
    ) -> np.ndarray:
        """Convenience: record a batch of signals and return updated grip."""
        if step is not None:
            self._global_step = step
        if train_loss is not None:
            self.record_train_loss(train_loss, self._global_step)
        if val_loss is not None:
            self.record_val_loss(val_loss)
        if grad_norms is not None:
            self.record_grad_norms(grad_norms)
        if introspection_depths is not None:
            self.record_introspection(introspection_depths)
        return self.compute_grip()

    def record_inference(self, val_metrics: Dict[str, float]) -> None:
        """Record inference-time metrics (called from evaluate())."""
        if "val_loss" in val_metrics:
            self.record_val_loss(val_metrics["val_loss"])

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    def _compute_gradient_SNR(self) -> float:
        if not self._grad_norms:
            return 1.0
        flat = [n for row in self._grad_norms for n in row if n is not None]
        if len(flat) < 2:
            return 1.0
        return float(np.mean(flat)) / (float(np.std(flat)) + 1e-8)

    def _compute_lyapunov_index(self) -> float:
        if not self._lyapunov_vals:
            return 0.5  # neutral
        return float(np.mean(list(self._lyapunov_vals)))

    def _compute_capacity_effectiveness_ratio(self) -> float:
        if not self._val_losses:
            return 1.0
        current_val = float(np.mean(list(self._val_losses)[-20:]))
        if self._min_val_loss == float("inf") or self._min_val_loss <= 0:
            return 1.0
        # ratio > 1 ⇒ model hasn't yet extracted its full capacity
        return math.exp(current_val) / (math.exp(self._min_val_loss) + 1e-8)

    def _compute_grip_vector(
        self,
        gradient_SNR: float,
        lyapunov_index: float,
        capacity_effectiveness_ratio: float,
    ) -> np.ndarray:
        """Compute the 7-D grip tensor."""
        # capacity_log term: positive when capacity > effectiveness
        cap_log = math.log(max(capacity_effectiveness_ratio, 1e-4))

        readiness = self.phase_readiness.get(self.current_phase, 0.5)

        grip = np.zeros(len(PERSONA_DIMS))
        for i, dim in enumerate(PERSONA_DIMS):
            # Pull opponent-process balance for this dimension
            opp_balance = self.opponent_balances.get(dim, 0.5)
            # Rescale coefficients by opponent balance so grip is reflexive
            a = _DIM_ALPHA[dim] * (1.0 + 0.4 * (opp_balance - 0.5))
            b = _DIM_BETA[dim] * (1.0 + 0.4 * (0.5 - opp_balance))
            g = _DIM_GAMMA[dim]
            # Bounded lyapunov contribution: prefer intermediate (0.3–0.7)
            lya_term = -abs(lyapunov_index - 0.5)  # 0 at edges, -0.5 at centre
            raw = a * cap_log + b * lya_term + g * (readiness - 0.5)
            grip[i] = _sigmoid(raw)
        return grip

    def compute_grip(self) -> np.ndarray:
        """Recompute and store all derived quantities; return grip vector."""
        self.gradient_SNR = self._compute_gradient_SNR()
        self.lyapunov_index = self._compute_lyapunov_index()
        self.capacity_effectiveness_ratio = self._compute_capacity_effectiveness_ratio()

        self.grip = self._compute_grip_vector(
            self.gradient_SNR,
            self.lyapunov_index,
            self.capacity_effectiveness_ratio,
        )

        # Scalar summaries
        self.learning_capacity_index = float(np.mean(self.grip[[0, 2, 4]]))  # cog/adaptive/synerg
        self.inference_effectiveness_index = float(np.mean(self.grip[[1, 3, 5]]))  # intro/recur/holo

        # Record for velocity computation
        self._grip_history.append(self.grip.copy())
        self._step_history.append(self._global_step)

        return self.grip

    def grip_velocity(self) -> float:
        """Return the mean rate of change of the grip vector (Δgrip/Δsteps).

        Positive = improving grip, negative = degrading, near-zero = stable.
        """
        if len(self._grip_history) < 2:
            return 0.0
        recent = list(self._grip_history)
        steps = list(self._step_history)
        g_new = np.mean(recent[-1])
        g_old = np.mean(recent[max(0, len(recent) - 10)])
        s_new = steps[-1]
        s_old = steps[max(0, len(steps) - 10)]
        delta_s = s_new - s_old
        if delta_s == 0:
            return 0.0
        return float((g_new - g_old) / delta_s)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "grip": self.grip.tolist(),
            "grip_mean": float(np.mean(self.grip)),
            "grip_velocity": self.grip_velocity(),
            "gradient_SNR": self.gradient_SNR,
            "lyapunov_index": self.lyapunov_index,
            "capacity_effectiveness_ratio": self.capacity_effectiveness_ratio,
            "learning_capacity_index": self.learning_capacity_index,
            "inference_effectiveness_index": self.inference_effectiveness_index,
            "current_phase": self.current_phase,
            "phase_readiness": dict(self.phase_readiness),
        }
