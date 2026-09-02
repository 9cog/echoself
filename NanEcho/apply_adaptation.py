#!/usr/bin/env python3
"""
NanEcho Config Rewriter — Apply Adaptation Proposals
=====================================================
Reads evaluation analysis outputs (automation_analysis.json, next_cycle_trigger.json,
and optionally eval_history/ reports from jsonl_eval.py) and rewrites
``nanecho_config.json`` so the *next* training cycle picks up improved
hyper-parameters automatically.

Autogenesis-informed safety features (from dte-ksm-evo-autogenesis):
  - **Delta Clamp**: No single parameter may change by more than 20% per cycle.
  - **Adaptation History**: Every cycle is logged to ``adaptation_history.jsonl``
    for cross-cycle trend analysis.
  - **Regression Detection**: If fidelity *dropped* after the last cycle's
    adaptations, revert ``adapted_config.json`` to its prior state.
  - **Coherence Halt**: If fidelity stays below 0.20 for 2+ consecutive cycles,
    write ``halt_adaptation.flag`` so the workflow stops triggering new cycles.
  - **Spectral Radius Monitoring**: If ESN spectral radius drifts outside the
    [0.85, 0.95] optimal band, adjust it back.

Usage
-----
  python apply_adaptation.py \\
      --analysis automation_analysis.json \\
      --trigger  next_cycle_trigger.json \\
      --config   ../nanecho_config.json \\
      --eval-report ../.training-progress/eval_history/ \\
      --output   adapted_config.json \\
      --history-file ../.training-progress/adaptation_history.jsonl
"""

import argparse
import copy
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning *None* on any error."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"⚠️  Could not load {path}: {exc}")
        return None


def _latest_eval_report(eval_dir: str) -> Optional[Dict[str, Any]]:
    """Find and load the most-recent ``eval_*.json`` in *eval_dir*."""
    if not eval_dir or not os.path.isdir(eval_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(eval_dir, "eval_*.json")))
    if not candidates:
        return None
    return _load_json(candidates[-1])


# ---------------------------------------------------------------------------
# Delta Clamp — Autogenesis safety constraint
# ---------------------------------------------------------------------------

# Maximum fractional change allowed per parameter per cycle.
# From dte-ksm-evo-autogenesis: "No single modification may alter a core
# parameter (e.g., spectral radius, learning rate) by more than 20%."
DELTA_CLAMP = 0.20

# Absolute hard cap for max_iters to prevent unbounded growth.
MAX_ITERS_CAP = 200000

# Parameters exempt from delta clamping (flags, weights that are rebalanced).
_CLAMP_EXEMPT = {"persona_weight", "dimension_weights"}


def _clamp_value(original: float, proposed: float, param_name: str) -> Tuple[float, bool]:
    """Clamp *proposed* so it differs from *original* by at most DELTA_CLAMP (20%).

    Returns (clamped_value, was_clamped).
    Skips clamping for parameters in ``_CLAMP_EXEMPT`` or when original is 0.
    """
    if param_name in _CLAMP_EXEMPT or original == 0:
        return proposed, False

    max_allowed = original * (1 + DELTA_CLAMP)
    min_allowed = original * (1 - DELTA_CLAMP)

    if proposed > max_allowed:
        return max_allowed, True
    if proposed < min_allowed:
        return min_allowed, True
    return proposed, False


def _clamp_int(original: int, proposed: int, param_name: str) -> Tuple[int, bool]:
    """Integer variant of _clamp_value — rounds to nearest int."""
    clamped, was_clamped = _clamp_value(float(original), float(proposed), param_name)
    return int(round(clamped)), was_clamped


def _enforce_embd_head_divisibility(
    model: Dict[str, Any],
    changes: List[str],
) -> None:
    """Ensure n_embd is divisible by n_head (transformer attention constraint).

    After delta clamping, n_embd and n_head may be independently rounded to
    incompatible values.  We fix this by rounding n_embd down to the nearest
    multiple of n_head.  See nanoGPT/model.py:33 —
    ``assert config.n_embd % config.n_head == 0``.
    """
    n_embd = model.get("n_embd")
    n_head = model.get("n_head")
    if n_embd is None or n_head is None or n_head == 0:
        return
    if n_embd % n_head != 0:
        adjusted = (n_embd // n_head) * n_head
        if adjusted == 0:
            adjusted = n_head  # minimum 1 head-dim
        model["n_embd"] = adjusted
        changes.append(
            f"n_embd {n_embd} → {adjusted} (rounded down for n_embd % n_head == 0)"
        )


# ---------------------------------------------------------------------------
# Adaptation History — cross-cycle trend tracking
# ---------------------------------------------------------------------------

def _load_history(history_path: str) -> List[Dict[str, Any]]:
    """Load adaptation history from a JSONL file."""
    entries: List[Dict[str, Any]] = []
    if not history_path or not os.path.isfile(history_path):
        return entries
    try:
        with open(history_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as exc:
        print(f"⚠️  Could not load history from {history_path}: {exc}")
    return entries


def _append_history(
    history_path: str,
    entry: Dict[str, Any],
) -> None:
    """Append a single entry to the adaptation history JSONL file."""
    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    print(f"📜 Adaptation history appended to: {history_path}")


# ---------------------------------------------------------------------------
# Regression Detection — revert if fidelity dropped after last adaptation
# ---------------------------------------------------------------------------

# Fidelity threshold below which we consider the system in "coherence crisis".
COHERENCE_CRISIS_THRESHOLD = 0.20
# Number of consecutive crisis cycles before halting.
COHERENCE_HALT_CONSECUTIVE = 2


def _check_regression(
    history: List[Dict[str, Any]],
    current_fidelity: float,
) -> bool:
    """Return True if the current fidelity is *worse* than the previous cycle's.

    This indicates the last adaptation made things worse and should be reverted.
    Only triggers if the previous cycle actually applied changes.
    """
    if not history:
        return False

    last = history[-1]
    prev_fidelity = last.get("fidelity_after", last.get("fidelity_before", 1.0))
    prev_changes = last.get("total_changes", 0)

    if prev_changes > 0 and current_fidelity < prev_fidelity:
        drop = prev_fidelity - current_fidelity
        print(f"📉 Regression detected: fidelity dropped {prev_fidelity:.3f} → {current_fidelity:.3f} "
              f"(Δ = -{drop:.3f}) after {prev_changes} adaptation(s) last cycle")
        return True
    return False


def _revert_adapted_config(output_path: str, config_path: str, history: List[Dict[str, Any]]) -> bool:
    """Revert adapted_config.json and nanecho_config.json to pre-regression state.

    Returns True if reverted, False if no prior state available.
    """
    if len(history) < 1:
        return False

    last = history[-1]
    prev_delta = last.get("delta_snapshot")
    if prev_delta and isinstance(prev_delta, dict):
        # Revert to the delta that existed *before* last cycle's changes.
        # If the last cycle was the first, revert to empty (no adaptations).
        if len(history) >= 2:
            second_last = history[-2]
            revert_to = second_last.get("delta_snapshot", {})
        else:
            revert_to = {}

        if os.path.isfile(output_path):
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(revert_to, fh, indent=2)
            print(f"⏪ Reverted adapted_config.json to pre-regression state")

        # Also restore the base nanecho_config.json if a snapshot was saved
        config_snapshot = last.get("config_snapshot")
        if config_snapshot and isinstance(config_snapshot, dict) and len(history) >= 2:
            # Restore the config that existed before the regressed cycle
            second_last = history[-2]
            prev_config = second_last.get("config_snapshot")
            if prev_config and isinstance(prev_config, dict):
                with open(config_path, "w", encoding="utf-8") as fh:
                    json.dump(prev_config, fh, indent=2)
                print(f"⏪ Restored nanecho_config.json to pre-regression state")
        elif not config_snapshot:
            # No config snapshot in history — at minimum, re-apply the reverted
            # delta on top of the original base config to keep them consistent
            print(f"⚠️  No config_snapshot in history — base config may be stale")

        return True
    return False


# ---------------------------------------------------------------------------
# Coherence Halt — emergency stop for sustained fidelity crisis
# ---------------------------------------------------------------------------

def _check_coherence_halt(
    history: List[Dict[str, Any]],
    current_fidelity: float,
    halt_flag_path: str,
) -> bool:
    """Check if fidelity has been in crisis for too many consecutive cycles.

    If so, write a halt flag file that the workflow can check before triggering
    the next cycle.  Returns True if halted.
    """
    # Count consecutive crisis cycles (including current)
    consecutive = 0
    if current_fidelity < COHERENCE_CRISIS_THRESHOLD:
        consecutive = 1
        for entry in reversed(history):
            f = entry.get("fidelity_after", entry.get("fidelity_before", 1.0))
            if f < COHERENCE_CRISIS_THRESHOLD:
                consecutive += 1
            else:
                break

    if consecutive >= COHERENCE_HALT_CONSECUTIVE:
        os.makedirs(os.path.dirname(halt_flag_path) or ".", exist_ok=True)
        with open(halt_flag_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "halted_at": datetime.now(timezone.utc).isoformat(),
                "reason": f"Fidelity below {COHERENCE_CRISIS_THRESHOLD} for "
                          f"{consecutive} consecutive cycles",
                "consecutive_crisis_cycles": consecutive,
                "last_fidelity": current_fidelity,
            }, indent=2))
        print(f"🛑 COHERENCE HALT: Fidelity < {COHERENCE_CRISIS_THRESHOLD} for "
              f"{consecutive} consecutive cycles. Wrote halt flag to {halt_flag_path}")
        return True

    # Clear halt flag if we recovered
    if os.path.isfile(halt_flag_path) and current_fidelity >= COHERENCE_CRISIS_THRESHOLD:
        os.remove(halt_flag_path)
        print(f"✅ Coherence recovered (fidelity={current_fidelity:.3f}). Cleared halt flag.")

    return False


# ---------------------------------------------------------------------------
# Adaptation rules
# ---------------------------------------------------------------------------

def _apply_fidelity_rules(
    config: Dict[str, Any],
    overall_fidelity: float,
    max_iters_original: int,
    changes: List[str],
) -> None:
    """Bump model / training parameters when persona fidelity is low.

    *max_iters_original* is the snapshot taken **before** any rules run.
    All multiplications are based on this original value to avoid compounding.
    Delta clamping is applied to every numeric change.
    """

    model = config.setdefault("model", {})
    training = config.setdefault("training", {})
    echo_self = config.setdefault("echo_self", {})
    data = config.setdefault("data", {})

    if overall_fidelity < 0.40:
        # Critical — aggressive parameter increase (clamped)
        raw_max = max(50000, int(max_iters_original * 1.5))
        new_max, clamped = _clamp_int(max_iters_original, raw_max, "max_iters")
        new_max = min(new_max, MAX_ITERS_CAP)
        if new_max != training.get("max_iters"):
            training["max_iters"] = new_max
            training["lr_decay_iters"] = new_max
            suffix = " [clamped]" if clamped else ""
            changes.append(f"max_iters → {new_max} (fidelity < 0.40, critical){suffix}")

        if data.get("persona_weight") != 0.95:
            data["persona_weight"] = 0.95
            changes.append("persona_weight → 0.95 (critical fidelity)")

        target_depth = 7
        current_depth = echo_self.get("max_recursion_depth", 5)
        if current_depth < target_depth:
            new_depth, clamped = _clamp_int(current_depth, target_depth, "max_recursion_depth")
            echo_self["max_recursion_depth"] = new_depth
            suffix = " [clamped]" if clamped else ""
            changes.append(f"echo_depth (max_recursion_depth) → {new_depth}{suffix}")

        current_lr = training.get("learning_rate", 1e-4)
        target_lr = 6e-5
        if current_lr != target_lr:
            new_lr, clamped = _clamp_value(current_lr, target_lr, "learning_rate")
            training["learning_rate"] = new_lr
            suffix = " [clamped]" if clamped else ""
            changes.append(f"learning_rate → {new_lr} (critical fidelity){suffix}")

        current_embd = model.get("n_embd", 768)
        if current_embd < 1024:
            new_embd, clamped = _clamp_int(current_embd, 1024, "n_embd")
            model["n_embd"] = new_embd
            suffix = " [clamped]" if clamped else ""
            changes.append(f"n_embd → {new_embd} (critical fidelity){suffix}")

        current_head = model.get("n_head", 12)
        if current_head < 16:
            new_head, clamped = _clamp_int(current_head, 16, "n_head")
            model["n_head"] = new_head
            suffix = " [clamped]" if clamped else ""
            changes.append(f"n_head → {new_head} (critical fidelity){suffix}")

        # Enforce n_embd % n_head == 0 after independent clamping
        _enforce_embd_head_divisibility(model, changes)

    elif overall_fidelity < 0.65:
        # Low — moderate increase (clamped)
        raw_max = int(max_iters_original * 1.5)
        new_max, clamped = _clamp_int(max_iters_original, raw_max, "max_iters")
        new_max = min(new_max, MAX_ITERS_CAP)
        if new_max != training.get("max_iters"):
            training["max_iters"] = new_max
            training["lr_decay_iters"] = new_max
            suffix = " [clamped]" if clamped else ""
            changes.append(f"max_iters → {new_max} (fidelity < 0.65){suffix}")

        current_lr = training.get("learning_rate", 1e-4)
        target_lr = 6e-5
        if current_lr != target_lr:
            new_lr, clamped = _clamp_value(current_lr, target_lr, "learning_rate")
            training["learning_rate"] = new_lr
            suffix = " [clamped]" if clamped else ""
            changes.append(f"learning_rate → {new_lr} (low fidelity){suffix}")

    elif overall_fidelity < 0.70 and max_iters_original >= 50000:
        # Borderline after substantial training — bump model capacity (clamped)
        current_embd = model.get("n_embd", 768)
        if current_embd < 1024:
            new_embd, clamped = _clamp_int(current_embd, 1024, "n_embd")
            model["n_embd"] = new_embd
            suffix = " [clamped]" if clamped else ""
            changes.append(f"n_embd {current_embd} → {new_embd} (fidelity < 0.70 after ≥50k iters){suffix}")

        current_head = model.get("n_head", 12)
        if current_head < 16:
            new_head, clamped = _clamp_int(current_head, 16, "n_head")
            model["n_head"] = new_head
            suffix = " [clamped]" if clamped else ""
            changes.append(f"n_head {current_head} → {new_head} (fidelity < 0.70 after ≥50k iters){suffix}")

        # Enforce n_embd % n_head == 0 after independent clamping
        _enforce_embd_head_divisibility(model, changes)


def _apply_eval_report_rules(
    config: Dict[str, Any],
    eval_report: Dict[str, Any],
    depth_original: int,
    changes: List[str],
) -> None:
    """Apply rules derived from the jsonl_eval.py report."""

    echo_self = config.setdefault("echo_self", {})
    model = config.setdefault("model", {})

    dimension_coverage = eval_report.get("dimension_coverage", 1.0)
    keyword_coverage = eval_report.get("keyword_coverage", 1.0)
    avg_latency = eval_report.get("avg_latency_ms", 0)

    # Rebalance dimension weights when coverage is low
    if dimension_coverage < 0.50:
        weights = echo_self.get("dimension_weights", {})
        if weights:
            avg_w = sum(weights.values()) / len(weights) if weights else 0.125
            rebalanced = {k: max(v, avg_w) for k, v in weights.items()}
            # Normalize so total weight is preserved at 1.0
            total = sum(rebalanced.values())
            if total > 0:
                rebalanced = {k: round(v / total, 4) for k, v in rebalanced.items()}
            if rebalanced != weights:
                echo_self["dimension_weights"] = rebalanced
                changes.append(
                    f"Rebalanced dimension_weights (dimension_coverage={dimension_coverage:.2f} < 0.50)"
                )

    # Double echo depth when keyword coverage is very low (clamped)
    # Skip if a higher-priority fidelity rule already changed max_recursion_depth
    if keyword_coverage < 0.40:
        current_depth = echo_self.get("max_recursion_depth", 5)
        if current_depth == depth_original:
            raw_depth = min(current_depth * 2, 14)
            new_depth, clamped = _clamp_int(current_depth, raw_depth, "max_recursion_depth")
            if new_depth != current_depth:
                echo_self["max_recursion_depth"] = new_depth
                suffix = " [clamped]" if clamped else ""
                changes.append(
                    f"max_recursion_depth {current_depth} → {new_depth} "
                    f"(keyword_coverage={keyword_coverage:.2f} < 0.40){suffix}"
                )

    # Reduce model size for inference when latency is too high (clamped)
    if avg_latency > 5000:
        current_layer = model.get("n_layer", 12)
        if current_layer > 6:
            new_layer, clamped = _clamp_int(current_layer, 6, "n_layer")
            model["n_layer"] = new_layer
            suffix = " [clamped]" if clamped else ""
            changes.append(
                f"n_layer → {new_layer} (avg_latency={avg_latency:.0f}ms > 5000ms){suffix}"
            )


def _apply_trigger_rules(
    config: Dict[str, Any],
    trigger: Dict[str, Any],
    max_iters_original: int,
    lr_original: float,
    changes: List[str],
) -> None:
    """Apply hyper-parameter hints from ``next_cycle_trigger.json``.

    *max_iters_original* and *lr_original* are pre-mutation snapshots.
    If a higher-priority rule already changed a parameter, that hint is skipped.
    Delta clamping is applied.
    """

    training = config.setdefault("training", {})
    hp = trigger.get("hyperparameter_adjustments", {})

    # Only apply LR hint if no higher-priority rule already changed it
    lr_hint = hp.get("learning_rate")
    if lr_hint == "increase" and training.get("learning_rate") == lr_original:
        raw_lr = min(lr_original * 1.5, 3e-4)
        new_lr, clamped = _clamp_value(lr_original, raw_lr, "learning_rate")
        if new_lr != lr_original:
            training["learning_rate"] = new_lr
            suffix = " [clamped]" if clamped else ""
            changes.append(f"learning_rate {lr_original} → {new_lr} (trigger hint: increase){suffix}")
    elif lr_hint == "decrease" and training.get("learning_rate") == lr_original:
        raw_lr = max(lr_original * 0.5, 1e-6)
        new_lr, clamped = _clamp_value(lr_original, raw_lr, "learning_rate")
        if new_lr != lr_original:
            training["learning_rate"] = new_lr
            suffix = " [clamped]" if clamped else ""
            changes.append(f"learning_rate {lr_original} → {new_lr} (trigger hint: decrease){suffix}")

    iters_hint = hp.get("max_iters")
    # Only apply if no higher-priority rule already changed max_iters
    if iters_hint == "increase" and training.get("max_iters") == max_iters_original:
        raw_val = int(max_iters_original * 1.25)
        new_val, clamped = _clamp_int(max_iters_original, raw_val, "max_iters")
        new_val = min(new_val, MAX_ITERS_CAP)
        if new_val != max_iters_original:
            training["max_iters"] = new_val
            training["lr_decay_iters"] = new_val
            suffix = " [clamped]" if clamped else ""
            changes.append(f"max_iters {max_iters_original} → {new_val} (trigger hint: increase){suffix}")


def _apply_recommendation_rules(
    config: Dict[str, Any],
    recommendations: Dict[str, Any],
    max_iters_original: int,
    changes: List[str],
) -> None:
    """Translate high-level recommendation strings into concrete config tweaks.

    *max_iters_original* is the pre-mutation snapshot to avoid compounding.
    If a higher-priority rule already changed max_iters, this rule is skipped.
    Delta clamping is applied.
    """

    training = config.setdefault("training", {})
    echo_self = config.setdefault("echo_self", {})

    hp_adjustments = recommendations.get("hyperparameter_adjustments", [])

    # Only apply if no higher-priority rule already changed max_iters
    if "extend_training_duration" in hp_adjustments and training.get("max_iters") == max_iters_original:
        raw_val = int(max_iters_original * 1.25)
        new_val, clamped = _clamp_int(max_iters_original, raw_val, "max_iters")
        new_val = min(new_val, MAX_ITERS_CAP)
        if new_val != max_iters_original:
            training["max_iters"] = new_val
            training["lr_decay_iters"] = new_val
            suffix = " [clamped]" if clamped else ""
            changes.append(f"max_iters {max_iters_original} → {new_val} (recommendation: extend_training_duration){suffix}")

    if "balance_persona_weights" in hp_adjustments:
        weights = echo_self.get("dimension_weights", {})
        if weights:
            uniform = round(1.0 / len(weights), 4)
            balanced = {k: uniform for k in weights}
            if balanced != weights:
                echo_self["dimension_weights"] = balanced
                changes.append("Balanced dimension_weights uniformly (recommendation: balance_persona_weights)")


# ---------------------------------------------------------------------------
# Spectral Radius Monitoring — from DTE Autonomy Evolution Report
# ---------------------------------------------------------------------------

# Optimal spectral radius band for ESN reservoir (from evolution report).
SPECTRAL_RADIUS_MIN = 0.85
SPECTRAL_RADIUS_MAX = 0.95
SPECTRAL_RADIUS_TARGET = 0.95


def _apply_spectral_radius_rules(
    config: Dict[str, Any],
    eval_report: Optional[Dict[str, Any]],
    changes: List[str],
) -> None:
    """Monitor ESN spectral radius and adjust if outside optimal band [0.85, 0.95].

    Reads spectral_radius from the eval report's ESN telemetry (if available)
    or from the config's echo_self section.
    """
    echo_self = config.setdefault("echo_self", {})

    # Try to get spectral_radius from eval report ESN telemetry
    current_sr = None
    if eval_report:
        esn_telemetry = eval_report.get("esn_telemetry", {})
        current_sr = esn_telemetry.get("spectral_radius")
        if current_sr is None:
            current_sr = eval_report.get("spectral_radius")

    # Fallback: read from config
    if current_sr is None:
        current_sr = echo_self.get("spectral_radius")

    if current_sr is None:
        return  # No spectral radius data available

    current_sr = float(current_sr)

    if current_sr < SPECTRAL_RADIUS_MIN:
        new_sr = SPECTRAL_RADIUS_TARGET
        new_sr, clamped = _clamp_value(current_sr, new_sr, "spectral_radius")
        echo_self["spectral_radius"] = round(new_sr, 4)
        suffix = " [clamped]" if clamped else ""
        changes.append(
            f"spectral_radius {current_sr:.4f} → {new_sr:.4f} "
            f"(below optimal band [{SPECTRAL_RADIUS_MIN}, {SPECTRAL_RADIUS_MAX}]){suffix}"
        )
    elif current_sr > SPECTRAL_RADIUS_MAX:
        new_sr = SPECTRAL_RADIUS_TARGET
        new_sr, clamped = _clamp_value(current_sr, new_sr, "spectral_radius")
        echo_self["spectral_radius"] = round(new_sr, 4)
        suffix = " [clamped]" if clamped else ""
        changes.append(
            f"spectral_radius {current_sr:.4f} → {new_sr:.4f} "
            f"(above optimal band [{SPECTRAL_RADIUS_MIN}, {SPECTRAL_RADIUS_MAX}]){suffix}"
        )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def apply_adaptation(
    analysis_path: str,
    trigger_path: str,
    config_path: str,
    eval_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    history_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Read evaluation artefacts, rewrite *config_path*, and return a summary.

    Parameters
    ----------
    analysis_path : str
        Path to ``automation_analysis.json`` (output of automation_integration.py).
    trigger_path : str
        Path to ``next_cycle_trigger.json`` (output of analyze_training_triggers.py).
    config_path : str
        Path to the current ``nanecho_config.json``.
    eval_dir : str, optional
        Directory containing ``eval_*.json`` reports from ``jsonl_eval.py``.
    output_path : str, optional
        If given, write the adapted config here *instead of* overwriting *config_path*.
    history_path : str, optional
        Path to ``adaptation_history.jsonl`` for cross-cycle trend tracking.

    Returns
    -------
    dict
        Summary with keys ``changes``, ``config_before``, ``config_after``.
    """

    print("🔄 NanEcho Config Rewriter — Applying adaptation proposals")
    print("=" * 60)

    # --- Load inputs --------------------------------------------------------
    analysis = _load_json(analysis_path) or {}
    trigger = _load_json(trigger_path) or {}
    config = _load_json(config_path)
    if config is None:
        print(f"❌ Cannot proceed without a valid config at {config_path}")
        sys.exit(1)

    eval_report = _latest_eval_report(eval_dir) if eval_dir else None

    config_before = copy.deepcopy(config)
    changes: List[str] = []

    # --- Extract key metrics (snapshot BEFORE any rules run) -----------------
    overall_fidelity = analysis.get("overall_fidelity", 1.0)
    max_iters_current = config.get("training", {}).get("max_iters", 50000)
    lr_current = config.get("training", {}).get("learning_rate", 1e-4)
    depth_current = config.get("echo_self", {}).get("max_recursion_depth", 5)
    recommendations = analysis.get("recommendations", {})

    print(f"📊 Overall fidelity : {overall_fidelity:.3f}")
    print(f"📊 Current max_iters: {max_iters_current}")
    print(f"📊 Delta clamp      : ±{DELTA_CLAMP:.0%} per parameter per cycle")

    # --- Load adaptation history for regression/halt checks -----------------
    dest = output_path or config_path
    halt_flag_path = os.path.join(os.path.dirname(dest), "halt_adaptation.flag")
    history = _load_history(history_path) if history_path else []

    # --- Regression detection -----------------------------------------------
    if _check_regression(history, overall_fidelity):
        reverted = _revert_adapted_config(dest, config_path, history)
        if reverted:
            # Log the revert in history
            revert_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": len(history) + 1,
                "action": "revert",
                "reason": "Fidelity regression detected after previous adaptation",
                "fidelity_before": overall_fidelity,
                "fidelity_after": overall_fidelity,
                "total_changes": 0,
                "changes_applied": ["REVERTED to pre-regression state"],
                "delta_snapshot": _load_json(dest) if os.path.isfile(dest) else {},
            }
            if history_path:
                _append_history(history_path, revert_entry)

            # Still check coherence halt even on revert
            _check_coherence_halt(history, overall_fidelity, halt_flag_path)

            summary = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "revert",
                "overall_fidelity": overall_fidelity,
                "changes_applied": ["REVERTED to pre-regression state"],
                "total_changes": 0,
            }
            print("\n⏪ Adaptation reverted due to regression. No new changes applied.")
            return summary

    # --- Coherence halt check -----------------------------------------------
    halted = _check_coherence_halt(history, overall_fidelity, halt_flag_path)
    if halted:
        halt_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": len(history) + 1,
            "action": "halt",
            "reason": f"Coherence halt — fidelity < {COHERENCE_CRISIS_THRESHOLD} "
                      f"for {COHERENCE_HALT_CONSECUTIVE}+ consecutive cycles",
            "fidelity_before": overall_fidelity,
            "fidelity_after": overall_fidelity,
            "total_changes": 0,
            "changes_applied": ["HALTED — awaiting human oversight"],
            "delta_snapshot": _load_json(dest) if os.path.isfile(dest) else {},
        }
        if history_path:
            _append_history(history_path, halt_entry)

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "halt",
            "overall_fidelity": overall_fidelity,
            "changes_applied": ["HALTED — awaiting human oversight"],
            "total_changes": 0,
        }
        print("\n🛑 Adaptation halted. No changes applied.")
        return summary

    # --- Apply rules in priority order --------------------------------------
    # Snapshot max_iters and learning_rate BEFORE any mutations so all rules
    # compare against the original values, preventing compounding and ensuring
    # higher-priority rules take precedence.
    _apply_fidelity_rules(config, overall_fidelity, max_iters_current, changes)
    _apply_recommendation_rules(config, recommendations, max_iters_current, changes)
    _apply_trigger_rules(config, trigger, max_iters_current, lr_current, changes)
    if eval_report:
        _apply_eval_report_rules(config, eval_report, depth_current, changes)
        _apply_spectral_radius_rules(config, eval_report, changes)
    else:
        # Still check spectral radius from config even without eval report
        _apply_spectral_radius_rules(config, None, changes)

    # --- Build delta of only changed keys ------------------------------------
    # We write ONLY the keys that were actually modified to adapted_config.json.
    # Writing the full base config would leak full-training values (n_layer=12,
    # max_iters=50000, etc.) into CI/scheduled runs that expect smaller defaults.
    delta: Dict[str, Any] = {}
    for section in ("model", "training", "echo_self", "data"):
        before_section = config_before.get(section, {})
        after_section = config.get(section, {})
        changed_keys = {
            k: v for k, v in after_section.items()
            if before_section.get(k) != v
        }
        if changed_keys:
            delta[section] = changed_keys

    if changes and delta:
        # Merge current delta INTO any existing adapted_config.json so that
        # prior cycle's adaptations (e.g. n_embd=1024) are preserved even
        # when the current cycle only changes different keys (e.g. max_iters).
        existing_delta = _load_json(dest) if os.path.isfile(dest) else None
        if existing_delta and isinstance(existing_delta, dict):
            for section, keys in delta.items():
                if section in existing_delta:
                    existing_delta[section].update(keys)
                else:
                    existing_delta[section] = keys
            merged = existing_delta
        else:
            merged = delta

        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2)
        print(f"\n💾 Adapted config (cumulative delta) written to: {dest}")

        # Also update the source nanecho_config.json with the full config
        # so it reflects the latest adapted state for training scripts
        # that read it directly.
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
        print(f"💾 Updated source config: {config_path}")
    else:
        print(f"\n⏭️  No changes — skipping write to {dest}")

    # --- Summary ------------------------------------------------------------
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analysis_source": analysis_path,
        "trigger_source": trigger_path,
        "eval_report_used": bool(eval_report),
        "overall_fidelity": overall_fidelity,
        "changes_applied": changes,
        "changes_delta": delta,
        "total_changes": len(changes),
        "config_output": dest if changes else None,
        "delta_clamp": DELTA_CLAMP,
    }

    if changes:
        print(f"\n✅ {len(changes)} adaptation(s) applied:")
        for i, c in enumerate(changes, 1):
            print(f"   {i}. {c}")
    else:
        print("\n✅ No adaptations needed — config is already optimal for current metrics.")

    # Write summary alongside the adapted config
    summary_path = dest.replace(".json", "_summary.json") if dest.endswith(".json") else dest + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"📝 Adaptation summary written to: {summary_path}")

    # --- Append to adaptation history ----------------------------------------
    if history_path:
        history_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": len(history) + 1,
            "action": "adapt" if changes else "no_change",
            "fidelity_before": overall_fidelity,
            "fidelity_after": overall_fidelity,  # Updated after next cycle's eval
            "total_changes": len(changes),
            "changes_applied": changes,
            "delta_snapshot": _load_json(dest) if changes and os.path.isfile(dest) else {},
            "config_snapshot": config_before,  # Full base config for revert support
            "delta_clamp": DELTA_CLAMP,
            "max_iters_before": max_iters_current,
            "lr_before": lr_current,
        }
        _append_history(history_path, history_entry)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply persona metric evaluation results back into training config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--analysis", required=True,
        help="Path to automation_analysis.json",
    )
    parser.add_argument(
        "--trigger", required=True,
        help="Path to next_cycle_trigger.json",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to nanecho_config.json (read/write)",
    )
    parser.add_argument(
        "--eval-report", default=None,
        help="Directory containing eval_*.json reports from jsonl_eval.py",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write adapted config to this path instead of overwriting --config",
    )
    parser.add_argument(
        "--history-file", default=None,
        help="Path to adaptation_history.jsonl for cross-cycle trend tracking",
    )
    args = parser.parse_args()

    apply_adaptation(
        analysis_path=args.analysis,
        trigger_path=args.trigger,
        config_path=args.config,
        eval_dir=args.eval_report,
        output_path=args.output,
        history_path=args.history_file,
    )


if __name__ == "__main__":
    main()
