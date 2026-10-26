"""Load the one adapted field cached training actually consumes.

``apply_adaptation.py`` writes ``training.learning_rate`` into a delta config.
``train_cached.py`` used to ignore that file. This reader applies only that
field, and only when it is present, so the 20% clamp in the adapter remains
the safety check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def adapted_learning_rate(path: Path) -> float | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    training = payload.get("training")
    if not isinstance(training, dict) or "learning_rate" not in training:
        return None
    value = float(training["learning_rate"])
    if value <= 0:
        raise ValueError(f"adapted learning_rate must be positive, got {value}")
    return value


def apply_adapted_learning_rate(config: Any, path: Path) -> bool:
    value = adapted_learning_rate(path)
    if value is None:
        return False
    config.learning_rate = value
    return True
