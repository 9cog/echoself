#!/usr/bin/env python3
"""
Tests for cross-cycle improvement tracking in NanEchoAutomationIntegrator.

Validates that:
1. improvement_history is persisted to / loaded from disk across restarts.
2. get_cycle_improvement_summary() correctly identifies improving, stable, and
   declining trends.
3. save_analysis_results() appends to history and calls _save_improvement_history.
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Make NanEcho importable from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "NanEcho"))


def _make_integrator(history_file: str):
    """Create a NanEchoAutomationIntegrator pointing at a temp history file."""
    from NanEcho.automation_integration import NanEchoAutomationIntegrator
    return NanEchoAutomationIntegrator(history_file=history_file)


def _make_fake_results(overall_fidelity: float, training_mode: str = "ci") -> dict:
    """Return a minimal analysis-results dict with the given overall fidelity."""
    from datetime import datetime
    return {
        "timestamp": datetime.now().isoformat(),
        "model_path": "/fake/model.pt",
        "training_mode": training_mode,
        "overall_fidelity": overall_fidelity,
        "quality_gate_status": {
            "status": "passed" if overall_fidelity >= 0.70 else "failed",
            "overall_passed": overall_fidelity >= 0.70,
            "deployment_ready": overall_fidelity >= 0.85,
            "individual_gates": {
                "overall_fidelity": {
                    "passed": overall_fidelity >= 0.70,
                    "score": overall_fidelity,
                    "threshold": 0.70,
                }
            },
        },
        "performance_analysis": {"trend": "initial"},
        "recommendations": {"immediate": [], "next_training_cycle": [], "long_term": [], "hyperparameter_adjustments": []},
        "next_actions": {"continue_training": True, "deploy_model": False, "retrain_required": False,
                         "enable_relentless_mode": False, "schedule_next_cycle": True, "manual_review_needed": False},
        "automation_triggers": {"trigger_next_training": False, "training_delay_hours": 0,
                                "hyperparameter_adjustments": {}, "dataset_enhancements": [], "notification_required": False},
    }


# ─── Test: history persists across restarts ────────────────────────────────────

def test_history_persists_across_restarts():
    """Improvement history saved in one session must be visible in the next."""
    print("🧪 test_history_persists_across_restarts ...")
    tmp_dir = tempfile.mkdtemp()
    try:
        history_file = os.path.join(tmp_dir, "history.json")

        # First session – record one cycle
        integrator1 = _make_integrator(history_file)
        assert len(integrator1.improvement_history) == 0, "Should start empty"

        results1 = _make_fake_results(0.72)
        integrator1.save_analysis_results(results1, os.path.join(tmp_dir, "out1.json"))
        assert len(integrator1.improvement_history) == 1

        # Second session – history should be restored from disk
        integrator2 = _make_integrator(history_file)
        assert len(integrator2.improvement_history) == 1, \
            f"Expected 1 entry from disk, got {len(integrator2.improvement_history)}"
        assert integrator2.improvement_history[0]["fidelity_metrics"]["overall_fidelity"] == 0.72

        print("   ✅ History persisted and restored correctly")
    finally:
        shutil.rmtree(tmp_dir)


# ─── Test: improvement summary – no data ──────────────────────────────────────

def test_summary_no_data():
    """Summary with no history should return 'no_data' trend."""
    print("🧪 test_summary_no_data ...")
    tmp_dir = tempfile.mkdtemp()
    try:
        integrator = _make_integrator(os.path.join(tmp_dir, "history.json"))
        summary = integrator.get_cycle_improvement_summary()
        assert summary["total_cycles"] == 0
        assert summary["trend"] == "no_data"
        assert summary["delta"] == 0.0
        print("   ✅ No-data case handled correctly")
    finally:
        shutil.rmtree(tmp_dir)


# ─── Test: improvement summary – improving trend ──────────────────────────────

def test_summary_improving_trend():
    """Fidelity rising over cycles should produce 'improving' trend."""
    print("🧪 test_summary_improving_trend ...")
    tmp_dir = tempfile.mkdtemp()
    try:
        integrator = _make_integrator(os.path.join(tmp_dir, "history.json"))

        # Simulate 6 cycles with clearly rising fidelity
        for fidelity in [0.60, 0.62, 0.65, 0.72, 0.78, 0.83]:
            integrator.improvement_history.append({
                "timestamp": "2026-01-01T00:00:00",
                "fidelity_metrics": {"overall_fidelity": fidelity},
                "quality_status": "passed",
                "training_mode": "ci",
            })

        summary = integrator.get_cycle_improvement_summary()
        assert summary["total_cycles"] == 6
        assert summary["trend"] == "improving", f"Expected 'improving', got '{summary['trend']}'"
        assert summary["delta"] > 0.05, f"Expected delta > 0.05, got {summary['delta']}"
        print(f"   ✅ Improving trend detected: delta={summary['delta']:+.3f}")
    finally:
        shutil.rmtree(tmp_dir)


# ─── Test: improvement summary – declining trend ──────────────────────────────

def test_summary_declining_trend():
    """Fidelity dropping over cycles should produce 'declining' trend."""
    print("🧪 test_summary_declining_trend ...")
    tmp_dir = tempfile.mkdtemp()
    try:
        integrator = _make_integrator(os.path.join(tmp_dir, "history.json"))

        for fidelity in [0.85, 0.80, 0.75, 0.65, 0.60, 0.55]:
            integrator.improvement_history.append({
                "timestamp": "2026-01-01T00:00:00",
                "fidelity_metrics": {"overall_fidelity": fidelity},
                "quality_status": "failed",
                "training_mode": "ci",
            })

        summary = integrator.get_cycle_improvement_summary()
        assert summary["trend"] == "declining", f"Expected 'declining', got '{summary['trend']}'"
        assert summary["delta"] < -0.05, f"Expected delta < -0.05, got {summary['delta']}"
        print(f"   ✅ Declining trend detected: delta={summary['delta']:+.3f}")
    finally:
        shutil.rmtree(tmp_dir)


# ─── Test: improvement summary – stable trend ─────────────────────────────────

def test_summary_stable_trend():
    """Fidelity hovering within a narrow band should produce 'stable' trend."""
    print("🧪 test_summary_stable_trend ...")
    tmp_dir = tempfile.mkdtemp()
    try:
        integrator = _make_integrator(os.path.join(tmp_dir, "history.json"))

        for fidelity in [0.72, 0.71, 0.73, 0.72, 0.71, 0.73]:
            integrator.improvement_history.append({
                "timestamp": "2026-01-01T00:00:00",
                "fidelity_metrics": {"overall_fidelity": fidelity},
                "quality_status": "passed",
                "training_mode": "ci",
            })

        summary = integrator.get_cycle_improvement_summary()
        assert summary["trend"] == "stable", f"Expected 'stable', got '{summary['trend']}'"
        assert abs(summary["delta"]) <= 0.05, f"Expected |delta| ≤ 0.05, got {summary['delta']}"
        print(f"   ✅ Stable trend detected: delta={summary['delta']:+.3f}")
    finally:
        shutil.rmtree(tmp_dir)


# ─── Test: save_analysis_results appends to history ───────────────────────────

def test_save_appends_history():
    """Each call to save_analysis_results must append exactly one entry."""
    print("🧪 test_save_appends_history ...")
    tmp_dir = tempfile.mkdtemp()
    try:
        integrator = _make_integrator(os.path.join(tmp_dir, "history.json"))
        for i, fidelity in enumerate([0.68, 0.72, 0.76]):
            results = _make_fake_results(fidelity)
            integrator.save_analysis_results(results, os.path.join(tmp_dir, f"out{i}.json"))

        assert len(integrator.improvement_history) == 3, \
            f"Expected 3 entries, got {len(integrator.improvement_history)}"

        # Reload from disk and verify count
        integrator2 = _make_integrator(os.path.join(tmp_dir, "history.json"))
        assert len(integrator2.improvement_history) == 3, \
            f"Expected 3 entries on reload, got {len(integrator2.improvement_history)}"
        print("   ✅ save_analysis_results correctly appends and persists")
    finally:
        shutil.rmtree(tmp_dir)


# ─── Test: report includes cross-cycle section ────────────────────────────────

def test_report_includes_cross_cycle_section():
    """generate_automation_report must contain a Cross-Cycle Improvement section."""
    print("🧪 test_report_includes_cross_cycle_section ...")
    tmp_dir = tempfile.mkdtemp()
    try:
        integrator = _make_integrator(os.path.join(tmp_dir, "history.json"))
        results = _make_fake_results(0.75)
        integrator.save_analysis_results(results, os.path.join(tmp_dir, "out.json"))
        report = integrator.generate_automation_report(results)
        assert "Cross-Cycle Improvement Trend" in report, \
            "Report should contain Cross-Cycle Improvement Trend section"
        print("   ✅ Report contains Cross-Cycle Improvement Trend section")
    finally:
        shutil.rmtree(tmp_dir)


# ─── Runner ───────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_history_persists_across_restarts,
        test_summary_no_data,
        test_summary_improving_trend,
        test_summary_declining_trend,
        test_summary_stable_trend,
        test_save_appends_history,
        test_report_includes_cross_cycle_section,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("🎉 All cycle improvement tests passed!")


if __name__ == "__main__":
    main()
