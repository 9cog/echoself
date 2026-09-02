"""Smoke tests for autognosis observe. No invented training metrics."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echoself.autognosis import AutognosisError, load_config, observe
from echoself.autognosis.observe import LAYERS


class TestAutognosis(unittest.TestCase):
    def setUp(self) -> None:
        self._prev_config = os.environ.pop("AUTOGNOSIS_CONFIG", None)
        self._prev_remember = os.environ.pop("AUTOGNOSIS_REMEMBER", None)

    def tearDown(self) -> None:
        if self._prev_config is None:
            os.environ.pop("AUTOGNOSIS_CONFIG", None)
        else:
            os.environ["AUTOGNOSIS_CONFIG"] = self._prev_config
        if self._prev_remember is None:
            os.environ.pop("AUTOGNOSIS_REMEMBER", None)
        else:
            os.environ["AUTOGNOSIS_REMEMBER"] = self._prev_remember

    def test_config_loads(self) -> None:
        config = load_config(ROOT)
        self.assertEqual(config["version"], 1)
        self.assertEqual(config["memory"]["backend"], "mech0")
        self.assertEqual(config["memory"]["type"], "autognosic")
        self.assertFalse(config["memory"]["cloud_mem0_required"])
        for layer in LAYERS:
            self.assertIn(layer, config["layers"])
            self.assertGreater(len(config["layers"][layer]), 0)

    def test_observe_reports_present_and_absent_without_invented_metrics(self) -> None:
        report = observe(ROOT)
        config = load_config(ROOT)
        expected_ids = {
            source["id"]
            for layer in LAYERS
            for source in config["layers"][layer]
        }
        observed_ids = {item["id"] for item in report["observations"]}
        self.assertEqual(observed_ids, expected_ids)

        invented = {"val_loss", "iteration", "quality_score", "tokens_processed"}
        for item in report["observations"]:
            self.assertIn(item["layer"], LAYERS)
            self.assertIsInstance(item["present"], bool)
            self.assertTrue(item["path"])
            self.assertFalse(invented.intersection(item))

        latest = next(
            item for item in report["observations"] if item["id"] == "latest_checkpoint"
        )
        self.assertFalse(latest["present"])
        self.assertEqual(latest["required_for"], "train")
        self.assertFalse(
            (ROOT / ".training-progress/checkpoints/latest_checkpoint.pt").is_file()
        )

        present = [item for item in report["observations"] if item["present"]]
        absent = [item for item in report["observations"] if not item["present"]]
        self.assertGreater(len(present), 0)
        self.assertGreater(len(absent), 0)

    def test_missing_checkpoint_is_not_train_ready(self) -> None:
        report = observe(ROOT)
        self.assertFalse(report["train_ready"])
        self.assertIsNone(report["local_checkpoint"])
        self.assertEqual(report["next_command"], "restore")
        from echoself.data.nanecho.surface import LINEAGE_KINDS, RESTORE_LINEAGE

        self.assertIn(report["lineage"]["kind"], LINEAGE_KINDS)
        self.assertIn(report["lineage"]["kind"], RESTORE_LINEAGE)
        self.assertNotEqual(report["next_command"], "train")

    def test_missing_config_raises(self) -> None:
        missing = ROOT / "does-not-exist-autognosis.json"
        os.environ["AUTOGNOSIS_CONFIG"] = str(missing)
        with self.assertRaises(AutognosisError):
            load_config(ROOT)

    def test_invalid_config_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "autognosis.json"
            bad.write_text(json.dumps({"version": 2, "layers": {}}), encoding="utf-8")
            os.environ["AUTOGNOSIS_CONFIG"] = str(bad)
            with self.assertRaises(AutognosisError):
                load_config(ROOT)

            bad.write_text(json.dumps({"version": 1, "layers": {"l0_observation": []}}), encoding="utf-8")
            with self.assertRaises(AutognosisError):
                load_config(ROOT)


if __name__ == "__main__":
    unittest.main()
