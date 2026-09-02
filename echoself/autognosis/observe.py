"""Observe autognosis.json sources. Missing files are observations, not invented facts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

LAYERS = ("l0_observation", "l1_pattern", "l2_meta")
CONFIG_NAME = "autognosis.json"


class AutognosisError(ValueError):
    """Invalid autognosis config or layer."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = root or repo_root()
    override = os.environ.get("AUTOGNOSIS_CONFIG")
    path = Path(override) if override else base / CONFIG_NAME
    if not path.is_file():
        raise AutognosisError(f"autognosis config missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise AutognosisError("autognosis.json version must be 1")
    layers = data.get("layers")
    if not isinstance(layers, dict) or any(name not in layers for name in LAYERS):
        raise AutognosisError(f"layers must include {', '.join(LAYERS)}")
    return data


def _present(base: Path, relative: str) -> bool:
    target = base / relative
    return target.is_file() or target.is_dir()


def observe(root: Path | None = None) -> dict[str, Any]:
    base = root or repo_root()
    config = load_config(base)
    observations: list[dict[str, Any]] = []
    for layer in LAYERS:
        for source in config["layers"][layer]:
            path = source["path"]
            observations.append(
                {
                    "layer": layer,
                    "id": source["id"],
                    "kind": source["kind"],
                    "path": path,
                    "present": _present(base, path),
                    "required_for": source.get("required_for"),
                }
            )

    from echoself.data.nanecho.surface import (
        RESTORE_LINEAGE,
        find_local_checkpoint,
        fold_lineage,
        resolve_nanecho_surface,
    )

    surface = resolve_nanecho_surface(base)
    lineage = fold_lineage(base)
    checkpoint = find_local_checkpoint(base)
    train_ready = any(
        item["id"] == "latest_checkpoint" and item["present"] for item in observations
    ) or checkpoint is not None

    verdict = lineage.get("kind", "uninitialized")
    next_command = (
        "restore" if verdict in RESTORE_LINEAGE or not train_ready else "respond"
    )

    band = tuple(config.get("esn", {}).get("spectral_radius_band", [0.85, 0.95]))
    return {
        "identity": config.get("identity", {}),
        "checkout": str(base),
        "memory": config.get("memory", {}),
        "esn": {"spectral_radius_band": list(band)},
        "observations": observations,
        "surface": {
            "kind": surface.kind,
            "path": surface.path,
            "reason": surface.reason,
        },
        "lineage": lineage,
        "local_checkpoint": None if checkpoint is None else checkpoint.as_posix(),
        "train_ready": bool(train_ready),
        "next_command": next_command,
    }


def remember(report: dict[str, Any], root: Path | None = None) -> int:
    from mech0.model import MemoryRecord
    from mech0.store import MemoryStore

    base = root or repo_root()
    store = MemoryStore(base / ".mech0" / "data")
    facts = (
        (
            "identity",
            f"{report['identity'].get('name')} / {report['identity'].get('profile')} "
            f"checkout {report['checkout']}",
        ),
        (
            "capability",
            f"NanEcho surface {report['surface'].get('kind')} at {report['surface'].get('path')}; "
            f"lineage {report['lineage'].get('kind')}; next_command {report['next_command']}",
        ),
        (
            "belief",
            "Local mech0 is the autognosic memory backend; cloud Mem0 is not required.",
        ),
        (
            "checkout",
            f"train_ready={report['train_ready']} local_checkpoint={report['local_checkpoint']}",
        ),
    )
    added = 0
    try:
        for about, content in facts:
            store.save(
                MemoryRecord.create(
                    memory_type="autognosic",
                    content=content,
                    source="autognosis.json",
                    payload={"about": about, "subject": "self"},
                    confidence=1.0,
                    pinned=True,
                    metadata={"source": "echoself.autognosis"},
                )
            )
            added += 1
    finally:
        store.close()
    return added


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    env_remember = os.environ.get("AUTOGNOSIS_REMEMBER", "0").strip().lower()
    write = "--remember" in args or env_remember in {"1", "true", "yes"}
    report = observe()
    print(json.dumps(report, indent=2))
    if write:
        count = remember(report)
        print(f"remembered {count} autognosic facts in mech0", file=sys.stderr)
    return 0
