"""Dynamic topology & model-size adaptation (Phase 4).

The reservoir's readout proposes which persona dimensions, attention heads,
and layers to grow or prune based on their measured grip contribution, and a
grip-saturation curve selects the smallest model size that saturates persona
grip — rather than fixing topology/size a priori.

Two capabilities
----------------
1. ``TopologyAdvisor`` — given per-dimension grip scores (from
   ``Introspection.evaluate_echo_self_quality``), propose per-dimension
   weight deltas (grow strong dimensions, decay weak ones). These deltas are
   consumed by the Phase-3 orchestrator as its ``dimension_weights`` output.

2. ``ModelSizeSelector`` — fit an elastic (saturating-exponential) grip curve
   to (model_size, grip) observations and return the smallest size whose
   predicted grip is within ``tolerance`` of the asymptote, so the
   orchestrator can warm-start the smallest sufficient configuration.

Both are pure, dependency-light, and checkpoint-serializable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TopologyProposal:
    """Per-dimension grow/prune proposal derived from grip contribution."""

    dimension_weights: Dict[str, float]
    grow: List[str] = field(default_factory=list)
    prune: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TopologyAdvisor:
    """Propose persona-dimension weight adjustments from grip contributions.

    Parameters
    ----------
    grow_threshold:
        Dimensions with grip >= this are reinforced (grown).
    prune_threshold:
        Dimensions with grip <= this are decayed (pruned).
    learning_rate:
        Step size for weight updates (0, 1].
    """

    def __init__(
        self,
        grow_threshold: float = 0.5,
        prune_threshold: float = 0.15,
        learning_rate: float = 0.2,
    ) -> None:
        self.grow_threshold = grow_threshold
        self.prune_threshold = prune_threshold
        self.learning_rate = learning_rate

    def propose(
        self,
        current_weights: Dict[str, float],
        grip_scores: Dict[str, float],
    ) -> TopologyProposal:
        """Compute new dimension weights from measured grip per dimension.

        ``grip_scores`` maps dimension name -> contribution in [0, 1]
        (e.g. per-dimension persona coverage). Dimensions above
        ``grow_threshold`` are reinforced; below ``prune_threshold`` are
        decayed; the rest are unchanged. Weights are renormalized to sum to 1.
        """
        if not current_weights:
            return TopologyProposal(dimension_weights={})
        new_weights: Dict[str, float] = {}
        grow: List[str] = []
        prune: List[str] = []
        for dim, w in current_weights.items():
            grip = grip_scores.get(dim, 0.0)
            if grip >= self.grow_threshold:
                new_weights[dim] = w * (1.0 + self.learning_rate)
                grow.append(dim)
            elif grip <= self.prune_threshold:
                new_weights[dim] = w * (1.0 - self.learning_rate)
                prune.append(dim)
            else:
                new_weights[dim] = w
        total = sum(new_weights.values()) or 1.0
        new_weights = {k: v / total for k, v in new_weights.items()}
        return TopologyProposal(dimension_weights=new_weights, grow=grow, prune=prune)


@dataclass
class SizeObservation:
    n_layer: int
    n_embd: int
    grip: float

    @property
    def param_proxy(self) -> float:
        """Rough parameter-count proxy used as the size axis."""
        return float(self.n_layer * self.n_embd * self.n_embd)


class ModelSizeSelector:
    """Select the smallest model size that saturates persona grip.

    Fits a saturating-exponential grip curve ``grip(s) = a * (1 - exp(-s/b))``
    to (size, grip) observations and returns the smallest size whose predicted
    grip is within ``tolerance`` of the asymptote ``a``.
    """

    def __init__(self, tolerance: float = 0.02) -> None:
        self.tolerance = tolerance
        self.observations: List[SizeObservation] = []

    def record(self, n_layer: int, n_embd: int, grip: float) -> None:
        self.observations.append(SizeObservation(n_layer=n_layer, n_embd=n_embd, grip=grip))

    def _fit(self) -> Optional[Tuple[float, float]]:
        """Least-squares fit of a saturating exponential; returns (a, b)."""
        if len(self.observations) < 3:
            return None
        s = np.array([o.param_proxy for o in self.observations])
        g = np.array([o.grip for o in self.observations])
        a0 = float(g.max()) if g.max() > 0 else 1.0
        b0 = float(np.median(s)) if len(s) else 1.0
        # Coarse grid search over (a, b) — robust and dependency-free.
        best = (a0, b0)
        best_err = float("inf")
        for a in np.linspace(max(a0 * 0.5, 1e-3), a0 * 1.5 + 1e-3, 25):
            for b in np.linspace(max(b0 * 0.1, 1.0), b0 * 3.0 + 1.0, 25):
                pred = a * (1.0 - np.exp(-s / b))
                err = float(np.mean((pred - g) ** 2))
                if err < best_err:
                    best_err = err
                    best = (float(a), float(b))
        return best

    def recommended_size(self, candidate_sizes: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
        """Return the smallest (n_layer, n_embd) candidate saturating grip.

        Falls back to the smallest candidate when there is too little data to
        fit a curve (safe default: start small and grow).
        """
        if not candidate_sizes:
            raise ValueError("candidate_sizes must be non-empty")
        ordered = sorted(candidate_sizes, key=lambda c: c[0] * c[1] * c[1])
        fit = self._fit()
        if fit is None:
            return ordered[0]
        a, b = fit
        if a <= 0:
            return ordered[0]
        for n_layer, n_embd in ordered:
            s = float(n_layer * n_embd * n_embd)
            pred = a * (1.0 - math.exp(-s / b))
            if pred >= a * (1.0 - self.tolerance):
                return (n_layer, n_embd)
        return ordered[-1]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "observations": [asdict(o) for o in self.observations],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.tolerance = float(state.get("tolerance", self.tolerance))
        self.observations = [
            SizeObservation(**o) for o in state.get("observations", [])
        ]
