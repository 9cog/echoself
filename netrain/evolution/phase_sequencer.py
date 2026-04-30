"""
Adaptive Phase Sequencer
========================

Replaces the fixed start_ratio / end_ratio phase clock in nanecho_config.json
with **multi-criteria transition gates** and **hysteresis bands**.

Phase progression
-----------------
  basic_awareness  →  persona_dimensions  →  hypergraph_patterns
    →  recursive_reasoning  →  adaptive_mastery

Forward gate for each transition: AND of all sub-conditions.
Backward (hysteresis): if any grip dimension < 0.35 for > patience_steps,
    regress to previous phase.

Topology targets
----------------
Each phase has a target topology dict that TopologyEvolutionController reads:
  spectral_radius, tree_depth, echo_depth, memory_size, permeability,
  active_echo_layers (number of transformer blocks with echo activated).

LR multiplier
-------------
  lr_mult = base_mult[phase] * (1.0 + 0.3 * (1 - capacity_effectiveness_ratio))
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from netrain.evolution.cognitive_grip_monitor import CognitiveGripMonitor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

PHASES: List[str] = [
    "basic_awareness",
    "persona_dimensions",
    "hypergraph_patterns",
    "recursive_reasoning",
    "adaptive_mastery",
]

# Topology target for each phase
PHASE_TOPOLOGY_TARGETS: Dict[str, Dict[str, Any]] = {
    "basic_awareness": {
        "spectral_radius": 0.95,
        "tree_depth":      1,
        "echo_depth":      1,
        "memory_size":     128,
        "permeability":    0.8,
        "active_echo_layers": 2,
    },
    "persona_dimensions": {
        "spectral_radius": 0.93,
        "tree_depth":      2,
        "echo_depth":      2,
        "memory_size":     256,
        "permeability":    0.7,
        "active_echo_layers": 4,
    },
    "hypergraph_patterns": {
        "spectral_radius": 0.90,
        "tree_depth":      4,
        "echo_depth":      3,
        "memory_size":     512,
        "permeability":    0.6,
        "active_echo_layers": 6,
    },
    "recursive_reasoning": {
        "spectral_radius": 0.88,
        "tree_depth":      9,
        "echo_depth":      5,
        "memory_size":     512,
        "permeability":    0.5,
        "active_echo_layers": 8,
    },
    "adaptive_mastery": {
        "spectral_radius": 0.85,
        "tree_depth":      9,
        "echo_depth":      7,
        "memory_size":     1024,
        "permeability":    0.5,
        "active_echo_layers": 12,
    },
}

# Base LR multipliers per phase
PHASE_BASE_LR_MULT: Dict[str, float] = {
    "basic_awareness":    1.20,
    "persona_dimensions": 1.00,
    "hypergraph_patterns": 0.90,
    "recursive_reasoning": 0.80,
    "adaptive_mastery":   0.70,
}

# Hysteresis: grip dimension floor
GRIP_REGRESSION_THRESHOLD: float = 0.35
GRIP_REGRESSION_PATIENCE: int = 200  # steps below threshold before regression


class AdaptivePhaseSequencer:
    """Multi-criteria adaptive phase sequencer with hysteresis.

    Parameters
    ----------
    config : dict
        ``adaptive_phases`` section of nanecho_config.json (may be empty;
        defaults are hardcoded above and match the plan exactly).
    patience_steps : int
        Steps of degraded grip before a backward phase regression is allowed.
    transition_freeze_steps : int
        Number of steps topology mutations are frozen around a phase change.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        patience_steps: int = GRIP_REGRESSION_PATIENCE,
        transition_freeze_steps: int = 150,
    ) -> None:
        cfg = config or {}
        self.patience_steps = int(cfg.get("patience_steps", patience_steps))
        self.transition_freeze_steps = int(
            cfg.get("transition_freeze_steps", transition_freeze_steps)
        )

        # Merge any overrides from config
        self._topology_targets: Dict[str, Dict[str, Any]] = {}
        for phase in PHASES:
            self._topology_targets[phase] = dict(PHASE_TOPOLOGY_TARGETS[phase])
            if "topology_targets" in cfg and phase in cfg["topology_targets"]:
                self._topology_targets[phase].update(cfg["topology_targets"][phase])

        self._base_lr_mult: Dict[str, float] = dict(PHASE_BASE_LR_MULT)
        if "base_lr_multipliers" in cfg:
            self._base_lr_mult.update(cfg["base_lr_multipliers"])

        # Runtime state
        self._phase_idx: int = 0
        self._steps_below_grip_threshold: int = 0
        self._in_transition: bool = False
        self._transition_steps_remaining: int = 0
        self._re_entry_flag: bool = False
        self._re_entry_steps_remaining: int = 0
        self._global_step: int = 0

        # Phase readiness cache (populated in step())
        self._phase_readiness: Dict[str, float] = {p: 0.0 for p in PHASES}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> str:
        return PHASES[self._phase_idx]

    @property
    def in_transition(self) -> bool:
        return self._in_transition

    @property
    def topology_target(self) -> Dict[str, Any]:
        return self._topology_targets[self.current_phase]

    # ------------------------------------------------------------------
    # Forward gate evaluation
    # ------------------------------------------------------------------

    def _eval_forward_gate(
        self,
        next_phase: str,
        grip_monitor: "CognitiveGripMonitor",
    ) -> Tuple[bool, float]:
        """Evaluate whether the forward gate to next_phase is satisfied.

        Returns (gate_open: bool, readiness_score: float in [0,1]).
        """
        val_loss = (
            float(np.mean(list(grip_monitor._val_losses)[-20:]))
            if grip_monitor._val_losses
            else float("inf")
        )
        lyapunov = grip_monitor.lyapunov_index
        grad_SNR = grip_monitor.gradient_SNR
        cer = grip_monitor.capacity_effectiveness_ratio
        grip = grip_monitor.grip

        conditions: List[bool] = []

        if next_phase == "persona_dimensions":
            conditions = [
                val_loss < 3.0,
                float(np.mean(grip)) > 0.7,
                grad_SNR > 0.4,
            ]
        elif next_phase == "hypergraph_patterns":
            conditions = [
                val_loss < 2.5,
                lyapunov < 0.8,  # echo state property proxy
                cer < 1.20,      # membrane stability proxy
            ]
        elif next_phase == "recursive_reasoning":
            conditions = [
                val_loss < 2.0,
                lyapunov < 0.75,
                lyapunov > 0.25,  # bounded (not frozen)
            ]
        elif next_phase == "adaptive_mastery":
            conditions = [
                val_loss < 1.6,
                float(np.min(grip)) > 0.6,
            ]
        else:
            # basic_awareness has no prerequisite
            conditions = [True]

        readiness = float(sum(conditions)) / max(len(conditions), 1)
        return (all(conditions), readiness)

    # ------------------------------------------------------------------
    # Step (called every training step from EchobeatTrainer)
    # ------------------------------------------------------------------

    def step(
        self,
        grip_monitor: "CognitiveGripMonitor",
        global_step: int,
        controller=None,   # TopologyEvolutionController (to flip transition flag)
    ) -> Dict[str, Any]:
        """Evaluate transitions and hysteresis; return a status dict."""
        self._global_step = global_step

        # Decrement transition freeze
        if self._in_transition:
            self._transition_steps_remaining -= 1
            if self._transition_steps_remaining <= 0:
                self._in_transition = False
                if controller is not None:
                    controller.set_phase_transition(False)

        # Decrement re-entry boost
        if self._re_entry_flag:
            self._re_entry_steps_remaining -= 1
            if self._re_entry_steps_remaining <= 0:
                self._re_entry_flag = False

        # Compute phase readiness for all phases
        for i, phase in enumerate(PHASES):
            if i <= self._phase_idx:
                self._phase_readiness[phase] = 1.0
            else:
                _, r = self._eval_forward_gate(phase, grip_monitor)
                self._phase_readiness[phase] = r

        # Push readiness into grip_monitor
        grip_monitor.phase_readiness = dict(self._phase_readiness)
        grip_monitor.current_phase = self.current_phase

        # ── Forward transition ─────────────────────────────────────────
        if not self._in_transition and self._phase_idx < len(PHASES) - 1:
            next_phase = PHASES[self._phase_idx + 1]
            gate_open, _ = self._eval_forward_gate(next_phase, grip_monitor)
            if gate_open:
                self._advance_phase(controller)
                return self._status("advanced_to", PHASES[self._phase_idx])

        # ── Backward regression (hysteresis) ─────────────────────────
        if not self._in_transition and self._phase_idx > 0:
            grip = grip_monitor.grip
            if float(np.min(grip)) < GRIP_REGRESSION_THRESHOLD:
                self._steps_below_grip_threshold += 1
            else:
                self._steps_below_grip_threshold = 0

            if self._steps_below_grip_threshold >= self.patience_steps:
                self._steps_below_grip_threshold = 0
                self._regress_phase(controller)
                return self._status("regressed_to", PHASES[self._phase_idx])

        return self._status("stable", PHASES[self._phase_idx])

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _advance_phase(self, controller) -> None:
        old = PHASES[self._phase_idx]
        self._phase_idx = min(self._phase_idx + 1, len(PHASES) - 1)
        new = PHASES[self._phase_idx]
        logger.info("Phase sequencer: %s → %s", old, new)
        self._start_transition(controller)

    def _regress_phase(self, controller) -> None:
        old = PHASES[self._phase_idx]
        self._phase_idx = max(self._phase_idx - 1, 0)
        new = PHASES[self._phase_idx]
        logger.warning("Phase sequencer regression: %s → %s", old, new)
        self._re_entry_flag = True
        self._re_entry_steps_remaining = self.patience_steps
        self._start_transition(controller)

    def _start_transition(self, controller) -> None:
        self._in_transition = True
        self._transition_steps_remaining = self.transition_freeze_steps
        if controller is not None:
            controller.set_phase_transition(True)
            controller.set_current_phase(self.current_phase)

    def _status(self, event: str, phase: str) -> Dict[str, Any]:
        return {
            "event": event,
            "phase": phase,
            "phase_idx": self._phase_idx,
            "in_transition": self._in_transition,
            "re_entry": self._re_entry_flag,
            "readiness": dict(self._phase_readiness),
        }

    # ------------------------------------------------------------------
    # LR multiplier
    # ------------------------------------------------------------------

    def lr_multiplier(self, capacity_effectiveness_ratio: float) -> float:
        """Compute adaptive LR multiplier for the current phase.

        lr_mult = base_mult * (1.0 + 0.3 * (1 - capacity_effectiveness_ratio))
        Plus a +0.2 bonus during re-entry after regression.
        """
        base = self._base_lr_mult.get(self.current_phase, 1.0)
        adaptive = base * (1.0 + 0.3 * (1.0 - capacity_effectiveness_ratio))
        if self._re_entry_flag:
            adaptive += 0.2
        return max(0.1, min(2.5, adaptive))

    # ------------------------------------------------------------------
    # Serialisation helpers (for checkpointing)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "phase_idx": self._phase_idx,
            "steps_below_grip": self._steps_below_grip_threshold,
            "in_transition": self._in_transition,
            "transition_steps_remaining": self._transition_steps_remaining,
            "re_entry_flag": self._re_entry_flag,
            "re_entry_steps_remaining": self._re_entry_steps_remaining,
            "global_step": self._global_step,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self._phase_idx = int(d.get("phase_idx", 0))
        self._steps_below_grip_threshold = int(d.get("steps_below_grip", 0))
        self._in_transition = bool(d.get("in_transition", False))
        self._transition_steps_remaining = int(d.get("transition_steps_remaining", 0))
        self._re_entry_flag = bool(d.get("re_entry_flag", False))
        self._re_entry_steps_remaining = int(d.get("re_entry_steps_remaining", 0))
        self._global_step = int(d.get("global_step", 0))
