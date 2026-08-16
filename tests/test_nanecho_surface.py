"""Smoke tests for the NanEcho surface adapter. No invented training metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echoself.data.nanecho.harmonic_resonance_esn import (
    HarmonicResonanceESN,
    OscillatorState,
    OscillatorStateError,
    persona_oscillators,
)
from echoself.data.nanecho.surface import (
    PERSONA_WEIGHTS,
    SurfaceError,
    compose_surface,
    find_local_checkpoint,
    fold_lineage,
    make_vorticog_agent,
    resolve_nanecho_surface,
    tiny_infer,
)


class TestNanechoSurface(unittest.TestCase):
    def test_resolver_selects_echoself_data_nanecho(self) -> None:
        ref = resolve_nanecho_surface(ROOT)
        self.assertEqual(ref.path, "echoself/data/nanecho")
        self.assertEqual(ref.kind, "failed")
        self.assertIsNotNone(ref.reason)
        self.assertNotIn("empty corpus", (ref.reason or "").lower())

    def test_missing_bins_are_prep_failure_not_empty_corpus(self) -> None:
        ref = resolve_nanecho_surface(ROOT)
        self.assertEqual(ref.kind, "failed")
        self.assertFalse((ROOT / "echoself/data/nanecho/train.bin").exists())
        self.assertFalse((ROOT / "data/nanecho").exists())
        self.assertFalse((ROOT / "NanEcho/data/nanecho").exists())

    def test_empty_oscillator_state_raises(self) -> None:
        with self.assertRaises(OscillatorStateError):
            OscillatorState(phases=(), amplitudes=())
        with self.assertRaises(OscillatorStateError):
            OscillatorState(phases=(0.1,), amplitudes=())
        with self.assertRaises(OscillatorStateError):
            persona_oscillators(())

    def test_harmonic_step_rotates_phase(self) -> None:
        state = OscillatorState(phases=(0.0, 1.0), amplitudes=(0.15, 0.10))
        node = HarmonicResonanceESN(state)
        after = node.step((0.2, 0.1))
        self.assertNotEqual(after.phases, state.phases)
        self.assertTrue(all(amp >= 0.0 for amp in after.amplitudes))
        self.assertTrue(abs(node.readout()) < 10.0)

    def test_vorticog_agent_requires_type(self) -> None:
        with self.assertRaises(SurfaceError):
            make_vorticog_agent("")
        with self.assertRaises(SurfaceError):
            make_vorticog_agent("persona")
        agent = make_vorticog_agent("persona", dimension="cognitive")
        self.assertEqual(agent.type, "persona")
        self.assertEqual(agent.dimension, "cognitive")

    def test_tiny_infer_unavailable_without_weights(self) -> None:
        self.assertIsNone(find_local_checkpoint(ROOT))
        result = tiny_infer("hello", ROOT)
        self.assertEqual(result.kind, "unavailable")
        self.assertIn("refusing model download", result.reason)

    def test_compose_runs_echogenesis_and_refuses_train(self) -> None:
        report = compose_surface(ROOT)
        self.assertEqual(report["surface"]["kind"], "failed")
        self.assertIsNone(report["corpus"].get("fallbackCorpus"))
        self.assertGreaterEqual(report["echogenesis"]["fragment_count"], 1)
        self.assertEqual(report["inference"]["kind"], "unavailable")
        self.assertEqual(report["training_command"], "restore")
        self.assertEqual(report["lineage"]["readiness"], "restore_required")
        self.assertFalse(report["lineage"]["latest_checkpoint_present"])
        types = {agent["type"] for agent in report["vorticog"]["agents"]}
        self.assertEqual(types, {"persona", "need", "dreamcog", "erebus"})
        self.assertEqual(len(PERSONA_WEIGHTS), 8)

    def test_lineage_cites_named_files_only(self) -> None:
        lineage = fold_lineage(ROOT)
        ids = {head["generation"] for head in lineage["heads"]}
        self.assertIn("504", ids)
        self.assertIn("695", ids)
        if not lineage["generation_827_on_disk"]:
            self.assertNotIn("827", ids)
        summary = ROOT / ".training-progress/nanecho-cached-ci/training_summary.json"
        self.assertTrue(summary.is_file())
        self.assertFalse(
            (ROOT / ".training-progress/checkpoints/latest_checkpoint.pt").is_file()
        )


if __name__ == "__main__":
    unittest.main()
