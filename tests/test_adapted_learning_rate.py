"""Cached training must consume an adapted learning rate when one was written."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapted_learning_rate import apply_adapted_learning_rate


def test_adapted_learning_rate_overrides_the_session_value(tmp_path):
    path = tmp_path / "adapted_config.json"
    path.write_text(
        json.dumps({"training": {"learning_rate": 0.00015}, "model": {"n_layer": 4}}),
        encoding="utf-8",
    )
    config = SimpleNamespace(learning_rate=0.0002)
    assert apply_adapted_learning_rate(config, path) is True
    assert config.learning_rate == 0.00015


def test_missing_adaptation_leaves_the_session_learning_rate(tmp_path):
    config = SimpleNamespace(learning_rate=0.0002)
    assert apply_adapted_learning_rate(config, tmp_path / "adapted_config.json") is False
    assert config.learning_rate == 0.0002


def test_delta_without_learning_rate_is_not_a_silent_change(tmp_path):
    path = tmp_path / "adapted_config.json"
    path.write_text(json.dumps({"echo_self": {"max_recursion_depth": 6}}), encoding="utf-8")
    config = SimpleNamespace(learning_rate=0.0002)
    assert apply_adapted_learning_rate(config, path) is False
    assert config.learning_rate == 0.0002
