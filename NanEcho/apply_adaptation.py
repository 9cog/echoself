#!/usr/bin/env python3
"""
NanEcho Config Rewriter — Apply Adaptation Proposals
=====================================================
Reads evaluation analysis outputs (automation_analysis.json, next_cycle_trigger.json,
and optionally eval_history/ reports from jsonl_eval.py) and rewrites
``nanecho_config.json`` so the *next* training cycle picks up improved
hyper-parameters automatically.

Usage
-----
  python apply_adaptation.py \
      --analysis automation_analysis.json \
      --trigger  next_cycle_trigger.json \
      --config   ../../nanecho_config.json \
      --eval-report ../../.training-progress/eval_history/ \
      --output   adapted_config.json
"""

import argparse
import copy
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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
# Adaptation rules
# ---------------------------------------------------------------------------

# Hard cap to prevent unbounded growth across adaptation cycles.
MAX_ITERS_CAP = 200000


def _apply_fidelity_rules(
    config: Dict[str, Any],
    overall_fidelity: float,
    max_iters_original: int,
    changes: List[str],
) -> None:
    """Bump model / training parameters when persona fidelity is low.

    *max_iters_original* is the snapshot taken **before** any rules run.
    All multiplications are based on this original value to avoid compounding.
    """

    model = config.setdefault("model", {})
    training = config.setdefault("training", {})
    echo_self = config.setdefault("echo_self", {})
    data = config.setdefault("data", {})

    if overall_fidelity < 0.40:
        # Critical — aggressive parameter increase
        new_max = min(max(50000, int(max_iters_original * 1.5)), MAX_ITERS_CAP)
        if new_max != training.get("max_iters"):
            training["max_iters"] = new_max
            training["lr_decay_iters"] = new_max
            changes.append(f"max_iters → {new_max} (fidelity < 0.40, critical)")

        if data.get("persona_weight") != 0.95:
            data["persona_weight"] = 0.95
            changes.append("persona_weight → 0.95 (critical fidelity)")

        target_depth = 7
        if echo_self.get("max_recursion_depth", 5) < target_depth:
            echo_self["max_recursion_depth"] = target_depth
            changes.append(f"echo_depth (max_recursion_depth) → {target_depth}")

        if training.get("learning_rate") != 6e-5:
            training["learning_rate"] = 6e-5
            changes.append("learning_rate → 6e-5 (critical fidelity)")

        if model.get("n_embd", 768) < 1024:
            model["n_embd"] = 1024
            changes.append("n_embd → 1024 (critical fidelity)")
        if model.get("n_head", 12) < 16:
            model["n_head"] = 16
            changes.append("n_head → 16 (critical fidelity)")

    elif overall_fidelity < 0.65:
        # Low — moderate increase
        new_max = min(int(max_iters_original * 1.5), MAX_ITERS_CAP)
        if new_max != training.get("max_iters"):
            training["max_iters"] = new_max
            training["lr_decay_iters"] = new_max
            changes.append(f"max_iters → {new_max} (fidelity < 0.65)")

        if training.get("learning_rate") != 6e-5:
            training["learning_rate"] = 6e-5
            changes.append("learning_rate → 6e-5 (low fidelity)")

    elif overall_fidelity < 0.70 and max_iters_original >= 50000:
        # Borderline after substantial training — bump model capacity
        if model.get("n_embd", 768) < 1024:
            model["n_embd"] = 1024
            changes.append("n_embd 768 → 1024 (fidelity < 0.70 after ≥50k iters)")
        if model.get("n_head", 12) < 16:
            model["n_head"] = 16
            changes.append("n_head 12 → 16 (fidelity < 0.70 after ≥50k iters)")


def _apply_eval_report_rules(
    config: Dict[str, Any],
    eval_report: Dict[str, Any],
    changes: List[str],
) -> None:
    """Apply rules derived from the jsonl_eval.py report."""

    echo_self = config.setdefault("echo_self", {})
    model = config.setdefault("model", {})
    data = config.setdefault("data", {})

    dimension_coverage = eval_report.get("dimension_coverage", 1.0)
    keyword_coverage = eval_report.get("keyword_coverage", 1.0)
    avg_latency = eval_report.get("avg_latency_ms", 0)

    # Rebalance dimension weights when coverage is low
    if dimension_coverage < 0.50:
        weights = echo_self.get("dimension_weights", {})
        if weights:
            avg_w = sum(weights.values()) / len(weights) if weights else 0.125
            rebalanced = {k: round(max(v, avg_w), 4) for k, v in weights.items()}
            if rebalanced != weights:
                echo_self["dimension_weights"] = rebalanced
                changes.append(
                    f"Rebalanced dimension_weights (dimension_coverage={dimension_coverage:.2f} < 0.50)"
                )

    # Double echo depth when keyword coverage is very low
    if keyword_coverage < 0.40:
        current_depth = echo_self.get("max_recursion_depth", 5)
        new_depth = min(current_depth * 2, 14)
        if new_depth != current_depth:
            echo_self["max_recursion_depth"] = new_depth
            changes.append(
                f"max_recursion_depth {current_depth} → {new_depth} "
                f"(keyword_coverage={keyword_coverage:.2f} < 0.40)"
            )

    # Reduce model size for inference when latency is too high
    if avg_latency > 5000:
        if model.get("n_layer", 12) > 6:
            model["n_layer"] = 6
            changes.append(
                f"n_layer → 6 (avg_latency={avg_latency:.0f}ms > 5000ms)"
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
    """

    training = config.setdefault("training", {})
    hp = trigger.get("hyperparameter_adjustments", {})

    # Only apply LR hint if no higher-priority rule already changed it
    lr_hint = hp.get("learning_rate")
    if lr_hint == "increase" and training.get("learning_rate") == lr_original:
        new_lr = min(lr_original * 1.5, 3e-4)
        if new_lr != lr_original:
            training["learning_rate"] = new_lr
            changes.append(f"learning_rate {lr_original} → {new_lr} (trigger hint: increase)")
    elif lr_hint == "decrease" and training.get("learning_rate") == lr_original:
        new_lr = max(lr_original * 0.5, 1e-6)
        if new_lr != lr_original:
            training["learning_rate"] = new_lr
            changes.append(f"learning_rate {lr_original} → {new_lr} (trigger hint: decrease)")

    iters_hint = hp.get("max_iters")
    # Only apply if no higher-priority rule already changed max_iters
    if iters_hint == "increase" and training.get("max_iters") == max_iters_original:
        new_val = min(int(max_iters_original * 1.25), MAX_ITERS_CAP)
        if new_val != max_iters_original:
            training["max_iters"] = new_val
            training["lr_decay_iters"] = new_val
            changes.append(f"max_iters {max_iters_original} → {new_val} (trigger hint: increase)")


def _apply_recommendation_rules(
    config: Dict[str, Any],
    recommendations: Dict[str, Any],
    max_iters_original: int,
    changes: List[str],
) -> None:
    """Translate high-level recommendation strings into concrete config tweaks.

    *max_iters_original* is the pre-mutation snapshot to avoid compounding.
    If a higher-priority rule already changed max_iters, this rule is skipped.
    """

    training = config.setdefault("training", {})
    echo_self = config.setdefault("echo_self", {})

    hp_adjustments = recommendations.get("hyperparameter_adjustments", [])

    # Only apply if no higher-priority rule already changed max_iters
    if "extend_training_duration" in hp_adjustments and training.get("max_iters") == max_iters_original:
        new_val = min(int(max_iters_original * 1.25), MAX_ITERS_CAP)
        if new_val != max_iters_original:
            training["max_iters"] = new_val
            training["lr_decay_iters"] = new_val
            changes.append(f"max_iters {max_iters_original} → {new_val} (recommendation: extend_training_duration)")

    if "balance_persona_weights" in hp_adjustments:
        weights = echo_self.get("dimension_weights", {})
        if weights:
            uniform = round(1.0 / len(weights), 4)
            balanced = {k: uniform for k in weights}
            if balanced != weights:
                echo_self["dimension_weights"] = balanced
                changes.append("Balanced dimension_weights uniformly (recommendation: balance_persona_weights)")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def apply_adaptation(
    analysis_path: str,
    trigger_path: str,
    config_path: str,
    eval_dir: Optional[str] = None,
    output_path: Optional[str] = None,
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

    # --- Extract key metrics ------------------------------------------------
    overall_fidelity = analysis.get("overall_fidelity", 1.0)
    max_iters_current = config.get("training", {}).get("max_iters", 50000)
    lr_current = config.get("training", {}).get("learning_rate", 1e-4)
    recommendations = analysis.get("recommendations", {})

    print(f"📊 Overall fidelity : {overall_fidelity:.3f}")
    print(f"📊 Current max_iters: {max_iters_current}")

    # --- Apply rules in priority order --------------------------------------
    # Snapshot max_iters and learning_rate BEFORE any mutations so all rules
    # compare against the original values, preventing compounding and ensuring
    # higher-priority rules take precedence.
    _apply_fidelity_rules(config, overall_fidelity, max_iters_current, changes)
    _apply_recommendation_rules(config, recommendations, max_iters_current, changes)
    _apply_trigger_rules(config, trigger, max_iters_current, lr_current, changes)
    if eval_report:
        _apply_eval_report_rules(config, eval_report, changes)

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

    dest = output_path or config_path

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
    args = parser.parse_args()

    apply_adaptation(
        analysis_path=args.analysis,
        trigger_path=args.trigger,
        config_path=args.config,
        eval_dir=args.eval_report,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
