"""ESN orchestration of training hyperparameters (Phase 3).

The ``ReservoirOrchestrator`` is the conductor of the training loop: an
Echo State Network whose ridge readout maps observed training state
(reservoir statistics + persona grip) to per-interval hyperparameter
decisions, replacing fixed schedules with persona-driven control.

Design
------
- **Observations**: reservoir state statistics (mean |state|, entropy
  ratio, spectral radius), recent val loss, connection ratio, and a scalar
  persona grip score (mean persona-dimension coverage from introspection).
- **ESN**: a small ``EchoReservoir`` (numpy, from ``dte_nodes``) integrates
  the observation stream over time — this is the reservoir that "watches"
  training.
- **Ridge readout**: a ``CognitiveReadout`` maps ESN states to decisions.
  It is re-fit online every decision interval from (state, decision) →
  (grip improvement) pairs — the ridge between reservoir and transformer.
- **Decisions** (all bounded for safety):
    - ``lr_scale``         multiplies the scheduled learning rate
    - ``connection_growth_rate``  overrides the model's growth rate
    - ``recursion_depth``  clamps recursive reasoning depth
    - ``dimension_weights`` re-weights persona dimensions

The orchestrator only acts when ``reservoir_mode == 'orchestrated'``; in
``shadow`` mode it observes and records but returns no-op decisions, and in
``off`` mode it is never constructed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path

import numpy as np

# The dte_nodes package uses bare ``dte_nodes.`` intra-package imports, so the
# NanEcho directory must be importable before importing the node modules.
_NANECHO_DIR = Path(__file__).resolve().parent
if str(_NANECHO_DIR) not in sys.path:
    sys.path.insert(0, str(_NANECHO_DIR))

from dte_nodes.echo_reservoir import EchoReservoir
from dte_nodes.cognitive_readout import CognitiveReadout


@dataclass
class OrchestratorDecision:
    """Bounded hyperparameter decisions for one interval."""

    lr_scale: float = 1.0
    connection_growth_rate: float = 0.05
    recursion_depth: int = 5
    dimension_weights: Optional[Dict[str, float]] = None

    def clamp(self) -> "OrchestratorDecision":
        self.lr_scale = float(min(max(self.lr_scale, 0.25), 4.0))
        self.connection_growth_rate = float(
            min(max(self.connection_growth_rate, 0.0), 0.25)
        )
        self.recursion_depth = int(min(max(self.recursion_depth, 1), 14))
        if self.dimension_weights is not None:
            total = sum(max(v, 0.0) for v in self.dimension_weights.values()) or 1.0
            self.dimension_weights = {
                k: max(v, 0.0) / total for k, v in self.dimension_weights.items()
            }
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Observation vector layout (fixed order).
OBS_KEYS = (
    "state_mean_abs",
    "state_entropy_ratio",
    "spectral_radius",
    "val_loss",
    "connection_ratio",
    "persona_grip",
)
OBS_DIM = len(OBS_KEYS)

# Decision vector layout for the ridge readout.
DECISION_DIM = 4  # lr_scale, connection_growth_rate, recursion_depth, grip_bias


class ReservoirOrchestrator:
    """Drives training hyperparameters from reservoir state + persona grip.

    Parameters
    ----------
    mode:
        ``shadow`` (observe only) or ``orchestrated`` (act on decisions).
    reservoir_units:
        Size of the orchestrator's internal ESN.
    decision_interval:
        How often (in eval intervals) to re-fit the ridge readout.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        mode: str = "orchestrated",
        reservoir_units: int = 64,
        decision_interval: int = 1,
        seed: int = 42,
    ) -> None:
        if mode not in ("shadow", "orchestrated"):
            raise ValueError(f"orchestrator mode must be shadow|orchestrated, got {mode!r}")
        self.mode = mode
        self.decision_interval = max(1, decision_interval)
        self.reservoir = EchoReservoir(
            units=reservoir_units,
            spectral_radius=0.95,
            input_scaling=0.1,
            seed=seed,
            name="OrchestratorESN",
        )
        self.readout = CognitiveReadout(
            output_dim=DECISION_DIM, ridge=1e-4, mode="offline", name="OrchestratorRidge"
        )

        # History for online ridge re-fit: (state, decision, reward).
        self._states: List[np.ndarray] = []
        self._decisions: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._last_grip: Optional[float] = None
        self._interval_count = 0
        self.history: List[Dict[str, Any]] = []

    # -- observation ----------------------------------------------------

    def observe(
        self,
        reservoir_stats: Optional[Dict[str, float]] = None,
        val_loss: float = 0.0,
        connection_ratio: float = 0.0,
        persona_grip: float = 0.0,
    ) -> np.ndarray:
        """Fold one training observation into the orchestrator ESN."""
        stats = reservoir_stats or {}
        vec = np.array(
            [
                stats.get("state_mean_abs", 0.0),
                stats.get("state_entropy_ratio", 0.0),
                stats.get("spectral_radius", 0.95),
                # Squash loss into a bounded signal.
                math.tanh(val_loss / 10.0),
                connection_ratio,
                persona_grip,
            ],
            dtype=float,
        )
        state = self.reservoir.step(vec)
        self._current_state = state
        self._current_obs = vec
        return state

    # -- decision ---------------------------------------------------------

    def _readout_vec(self) -> np.ndarray:
        state = getattr(self, "_current_state", None)
        if state is None:
            state = np.zeros(self.reservoir.units)
        if not self.readout.initialized:
            self.readout.initialize(state)
        return self.readout.step(state)

    def decide(self) -> OrchestratorDecision:
        """Produce bounded hyperparameter decisions from the current state."""
        raw = self._readout_vec()
        # Map raw readout outputs through squashing functions to bounded ranges.
        lr_scale = 1.0 + math.tanh(float(raw[0]))
        growth = 0.05 * (1.0 + math.tanh(float(raw[1])))
        recursion = int(round(5 + 4 * math.tanh(float(raw[2]))))
        grip_bias = float(raw[3])

        decision = OrchestratorDecision(
            lr_scale=lr_scale,
            connection_growth_rate=growth,
            recursion_depth=recursion,
        ).clamp()

        # Record for the online re-fit and history.
        self._states.append(getattr(self, "_current_state", np.zeros(self.reservoir.units)).copy())
        self._decisions.append(np.array([lr_scale, growth, recursion, grip_bias]))
        self.history.append(
            {
                "interval": self._interval_count,
                "observation": getattr(self, "_current_obs", np.zeros(OBS_DIM)).tolist(),
                "decision": decision.to_dict(),
                "mode": self.mode,
            }
        )

        if self.mode == "shadow":
            # Observation-only: never alter the schedule.
            return OrchestratorDecision()
        return decision

    # -- learning ---------------------------------------------------------

    def report_grip(self, persona_grip: float) -> None:
        """Provide the grip outcome used to reward the previous decision."""
        if self._last_grip is not None:
            reward = persona_grip - self._last_grip
            self._rewards.append(reward)
        self._last_grip = persona_grip
        self._interval_count += 1
        if self._interval_count % self.decision_interval == 0:
            self._refit()

    def _refit(self) -> None:
        """Re-fit the ridge readout from reward-weighted (state, decision) pairs.

        States with positive grip improvement get their decisions reinforced;
        the readout is fit to reproduce reward-scaled decisions, so better
        decisions become more likely. This is the ridge regression sitting
        between the reservoir and the transformer's update rule.
        """
        n = min(len(self._states), len(self._decisions), len(self._rewards))
        if n < 2:
            return
        X = np.stack(self._states[-n:])
        D = np.stack(self._decisions[-n:])
        R = np.array(self._rewards[-n:])
        # Shift rewards to non-negative weights.
        w = R - R.min() + 1e-3
        Y = D * w[:, None]
        try:
            self.readout.fit(X, Y)
        except Exception:
            # A singular system should never crash training.
            pass

    # -- persistence ------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Serializable orchestrator state for checkpoint persistence."""
        return {
            "mode": self.mode,
            "interval_count": self._interval_count,
            "last_grip": self._last_grip,
            "reservoir": {
                "units": self.reservoir.units,
                "spectral_radius": self.reservoir.spectral_radius,
                "Win": self.reservoir.Win.tolist() if self.reservoir.Win is not None else None,
                "W": self.reservoir.W.tolist() if self.reservoir.W is not None else None,
            },
            "readout": {
                "Wout": self.readout.Wout.tolist() if self.readout.Wout is not None else None,
                "bias": self.readout.bias.tolist() if self.readout.bias is not None else None,
            },
            "history": self.history[-100:],  # keep tail for analysis
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore orchestrator state from a checkpoint."""
        self.mode = state.get("mode", self.mode)
        self._interval_count = int(state.get("interval_count", 0))
        self._last_grip = state.get("last_grip")
        res = state.get("reservoir", {})
        if res.get("W") is not None and res.get("Win") is not None:
            self.reservoir.Win = np.array(res["Win"])
            self.reservoir.W = np.array(res["W"])
            self.reservoir.input_dim = self.reservoir.Win.shape[1]
            self.reservoir.initialized = True
            self.reservoir.reset()
        ro = state.get("readout", {})
        if ro.get("Wout") is not None:
            self.readout.Wout = np.array(ro["Wout"])
            self.readout.bias = np.array(ro.get("bias", []))
            self.readout.input_dim = self.readout.Wout.shape[0]
            self.readout.initialized = True
        self.history = list(state.get("history", []))
