"""
Topology Evolution Controller
==============================

Converts CognitiveGrip signals into structural changes at two timescales:

Fast gear  (runs every echobeat_period steps)
  - EchoReservoir spectral-radius targeting via entropy feedback
  - fast/slow leak-rate ratio adjustment via lyapunov_index
  - MembraneNode permeability annealing toward phase setpoint

Slow gear  (runs every topology_epoch_beats * echobeat_period steps)
  - Tree depth ladder following OEIS A000081 values [1,1,2,4,9,20,48]
  - Echo layer activation schedule (in rooted-tree enumeration order)
  - Echo depth per-block increment
  - Memory bank doubling when capacity_effectiveness_ratio > 1.15

Mutation gate
  mutate = (|grip_velocity| > θ_mut)
           AND (steps_since_mutation > τ_cooldown)
           AND (NOT in_phase_transition)
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from netrain.evolution.cognitive_grip_monitor import CognitiveGripMonitor

logger = logging.getLogger(__name__)

# OEIS A000081 sequence: rooted tree counts — natural topology ladder
OEIS_A000081 = [1, 1, 2, 4, 9, 20, 48, 115, 286]


class TopologyEvolutionController:
    """Differential topology mutation controller.

    Parameters
    ----------
    config : dict
        ``topology_evolution`` section of nanecho_config.json.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.echobeat_period: int = config.get("echobeat_period", 9)
        self.topology_epoch_beats: int = config.get("topology_epoch_beats", 100)
        self.mutation_cooldown_steps: int = config.get("mutation_cooldown_steps", 500)
        self.grip_velocity_threshold: float = config.get("grip_velocity_threshold", 0.02)
        self.spectral_radius_range: tuple = tuple(
            config.get("spectral_radius_range", [0.80, 0.98])
        )
        self.oeis_ladder: List[int] = config.get(
            "oeis_a000081_ladder", OEIS_A000081[:7]
        )
        self.entropy_target_by_phase: Dict[str, float] = config.get(
            "entropy_target_by_phase",
            {
                "basic_awareness":    0.80,
                "persona_dimensions": 0.75,
                "hypergraph_patterns": 0.70,
                "recursive_reasoning": 0.65,
                "adaptive_mastery":   0.60,
            },
        )
        gate_cfg = config.get("mutation_gate", {})
        self.min_grip_velocity: float = gate_cfg.get("min_grip_velocity", 0.01)
        self.max_grip_velocity: float = gate_cfg.get("max_grip_velocity", 0.30)

        # Internal state
        self._steps_since_fast_mutation: int = 0
        self._steps_since_slow_mutation: int = 0
        self._in_phase_transition: bool = False
        self._fast_step_counter: int = 0
        self._slow_beat_counter: int = 0

        # Current topology shadow (updated by model instrumentation)
        self._current_tree_depth_idx: int = 0   # index into OEIS ladder
        self._current_phase: str = "basic_awareness"

        # Track current spectral radius for delta computation
        self._current_sr: float = 0.95

    # ------------------------------------------------------------------
    # Phase transition flag (set by AdaptivePhaseSequencer)
    # ------------------------------------------------------------------

    def set_phase_transition(self, in_transition: bool) -> None:
        self._in_phase_transition = in_transition

    def set_current_phase(self, phase: str) -> None:
        self._current_phase = phase

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mutation_gate_open(self, grip_velocity: float) -> bool:
        if self._in_phase_transition:
            return False
        vel = abs(grip_velocity)
        if vel < self.min_grip_velocity:
            return False
        if vel > self.max_grip_velocity:
            return False
        return True

    @staticmethod
    def _spectral_radius_target(
        current_entropy: float,
        entropy_target: float,
        k: float = 5.0,
    ) -> float:
        """Compute desired spectral radius from entropy deviation."""
        # SR increases when entropy is below target (need more chaos)
        # SR decreases when entropy is above target (need more stability)
        delta = entropy_target - current_entropy
        return 0.5 + 0.5 * math.tanh(k * delta)

    # ------------------------------------------------------------------
    # Fast-gear update (every echobeat_period steps)
    # ------------------------------------------------------------------

    def fast_update(
        self,
        reservoir,           # EchoReservoir instance (may be None)
        membrane,            # MembraneNode instance (may be None)
        grip_monitor: "CognitiveGripMonitor",
        phase_topology_target: Dict[str, Any],
        step: int,
    ) -> Dict[str, Any]:
        """Apply fast-gear adaptations.

        Returns a dict describing what (if anything) was changed.
        """
        self._fast_step_counter += 1
        if self._fast_step_counter % self.echobeat_period != 0:
            return {}

        self._steps_since_fast_mutation += 1
        changes: Dict[str, Any] = {}

        # ── Spectral radius targeting ──────────────────────────────────
        if reservoir is not None:
            entropy_target = self.entropy_target_by_phase.get(self._current_phase, 0.70)
            lyapunov = grip_monitor.lyapunov_index
            # Use lyapunov as a proxy for current entropy (both in [0,1])
            current_entropy = lyapunov
            sr_target = self._spectral_radius_target(current_entropy, entropy_target)
            sr_target = max(self.spectral_radius_range[0],
                            min(self.spectral_radius_range[1], sr_target))

            if abs(sr_target - self._current_sr) > 0.005:
                old_sr = reservoir.spectral_radius
                reservoir.adapt_spectral_radius(sr_target)
                self._current_sr = sr_target
                changes["spectral_radius"] = {"from": old_sr, "to": sr_target}
                logger.debug("Fast gear: SR %s → %s", old_sr, sr_target)

            # ── Leak-rate adjustment based on lyapunov ────────────────
            # Too chaotic (lyapunov > 0.7) → slow down fast pool
            # Too frozen  (lyapunov < 0.3) → speed up fast pool
            if lyapunov > 0.7 and reservoir.leak_rate_fast > 0.5:
                reservoir.leak_rate_fast = max(0.5, reservoir.leak_rate_fast - 0.02)
                changes["leak_rate_fast"] = reservoir.leak_rate_fast
            elif lyapunov < 0.3 and reservoir.leak_rate_fast < 0.95:
                reservoir.leak_rate_fast = min(0.95, reservoir.leak_rate_fast + 0.02)
                changes["leak_rate_fast"] = reservoir.leak_rate_fast

        # ── Membrane permeability annealing ──────────────────────────
        if membrane is not None:
            target_perm = phase_topology_target.get("permeability", 0.5)
            current_perm = membrane.get_permeability_profile()
            mean_perm = sum(current_perm) / len(current_perm) if current_perm else 0.5
            if abs(mean_perm - target_perm) > 0.03:
                # Nudge bias terms toward target
                direction = 1.0 if target_perm > mean_perm else -1.0
                if hasattr(membrane, "bias_gate"):
                    for bias in membrane.bias_gate:
                        bias += direction * 0.05
                    changes["membrane_permeability"] = {
                        "mean_was": mean_perm, "target": target_perm
                    }

        return changes

    # ------------------------------------------------------------------
    # Slow-gear update (every topology_epoch_beats echobeats)
    # ------------------------------------------------------------------

    def slow_update(
        self,
        model,              # DeepTreeEchoTransformer instance (may be None)
        grip_monitor: "CognitiveGripMonitor",
        phase_topology_target: Dict[str, Any],
        step: int,
    ) -> Dict[str, Any]:
        """Apply slow-gear topology mutations.

        Returns a dict describing what (if anything) was changed.
        """
        self._slow_beat_counter += 1
        if self._slow_beat_counter % self.topology_epoch_beats != 0:
            return {}

        grip_vel = grip_monitor.grip_velocity()
        if not self._mutation_gate_open(grip_vel):
            logger.debug(
                "Slow gear: mutation gate closed (vel=%.4f, in_transition=%s)",
                grip_vel, self._in_phase_transition,
            )
            return {}

        cooldown_ok = (self._steps_since_slow_mutation >= self.mutation_cooldown_steps)
        if not cooldown_ok:
            logger.debug(
                "Slow gear: cooldown not expired (%d/%d)",
                self._steps_since_slow_mutation, self.mutation_cooldown_steps,
            )
            return {}

        if model is None:
            return {}

        changes: Dict[str, Any] = {}
        gradient_SNR = grip_monitor.gradient_SNR
        lyapunov = grip_monitor.lyapunov_index
        cer = grip_monitor.capacity_effectiveness_ratio

        # ── Tree depth (OEIS A000081 ladder) ─────────────────────────
        target_depth = phase_topology_target.get("tree_depth", 1)
        current_depth = getattr(model, "_current_tree_depth", 1)
        snr_ok = gradient_SNR > 0.4
        vel_positive = grip_vel > 0

        if target_depth > current_depth and snr_ok and vel_positive:
            new_depth = min(target_depth, current_depth + 1)
            try:
                model.set_tree_depth(new_depth)
                model._current_tree_depth = new_depth
                changes["tree_depth"] = {"from": current_depth, "to": new_depth}
                logger.info("Slow gear: tree_depth %d → %d", current_depth, new_depth)
                self._steps_since_slow_mutation = 0
            except Exception as exc:  # pragma: no cover
                logger.warning("set_tree_depth failed: %s", exc)

        elif target_depth < current_depth:
            new_depth = max(target_depth, current_depth - 1)
            try:
                model.set_tree_depth(new_depth)
                model._current_tree_depth = new_depth
                changes["tree_depth"] = {"from": current_depth, "to": new_depth}
                logger.info("Slow gear: tree_depth %d → %d (regression)", current_depth, new_depth)
                self._steps_since_slow_mutation = 0
            except Exception as exc:  # pragma: no cover
                logger.warning("set_tree_depth failed: %s", exc)

        # ── Echo depth per block ──────────────────────────────────────
        target_echo_depth = phase_topology_target.get("echo_depth", 1)
        if hasattr(model, "blocks"):
            for idx, block in enumerate(model.blocks):
                if block.echo_layer is None:
                    continue
                cur_ed = block.echo_layer.echo_depth
                if (
                    target_echo_depth > cur_ed
                    and gradient_SNR > 0.3
                    and lyapunov < 0.75
                ):
                    try:
                        model.set_echo_depth(idx, cur_ed + 1)
                        changes.setdefault("echo_depth", {})[idx] = cur_ed + 1
                        logger.info("Slow gear: block %d echo_depth %d → %d",
                                    idx, cur_ed, cur_ed + 1)
                        self._steps_since_slow_mutation = 0
                        break  # one block per slow tick
                    except Exception as exc:  # pragma: no cover
                        logger.warning("set_echo_depth failed for block %d: %s", idx, exc)

        # ── Memory bank growth ────────────────────────────────────────
        if cer > 1.15 and hasattr(model, "memory_bank") and model.memory_bank is not None:
            target_mem = phase_topology_target.get("memory_size", 512)
            current_mem = model.memory_bank.memory_size
            if target_mem > current_mem:
                new_mem = min(target_mem, current_mem * 2)
                try:
                    model.set_memory_size(new_mem)
                    changes["memory_size"] = {"from": current_mem, "to": new_mem}
                    logger.info("Slow gear: memory_size %d → %d", current_mem, new_mem)
                    self._steps_since_slow_mutation = 0
                except Exception as exc:  # pragma: no cover
                    logger.warning("set_memory_size failed: %s", exc)

        # ── Activate dormant echo layers ──────────────────────────────
        if hasattr(model, "blocks"):
            activated = getattr(model, "_activated_echo_layers", set())
            for idx, block in enumerate(model.blocks):
                if idx in activated:
                    continue
                if block.echo_layer is not None:
                    continue  # already active
                # Activate when SNR is good and we're past early training
                if gradient_SNR > 0.5 and lyapunov > 0.35:
                    target_active_count = phase_topology_target.get(
                        "active_echo_layers", len(model.blocks)
                    )
                    if len(activated) < target_active_count:
                        try:
                            model.activate_echo_layer(idx)
                            activated.add(idx)
                            model._activated_echo_layers = activated
                            changes["activated_echo_layer"] = idx
                            logger.info("Slow gear: activated echo layer on block %d", idx)
                            self._steps_since_slow_mutation = 0
                            break
                        except Exception as exc:  # pragma: no cover
                            logger.warning("activate_echo_layer failed for block %d: %s", idx, exc)

        self._steps_since_slow_mutation += 1
        return changes

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def maybe_mutate(
        self,
        model,
        reservoir,
        membrane,
        grip_monitor: "CognitiveGripMonitor",
        phase_topology_target: Dict[str, Any],
        step: int,
    ) -> Dict[str, Any]:
        """Run both gears and return a combined change-log dict."""
        fast_changes = self.fast_update(
            reservoir, membrane, grip_monitor, phase_topology_target, step
        )
        slow_changes = self.slow_update(
            model, grip_monitor, phase_topology_target, step
        )
        all_changes: Dict[str, Any] = {}
        all_changes.update(fast_changes)
        all_changes.update(slow_changes)
        return all_changes
