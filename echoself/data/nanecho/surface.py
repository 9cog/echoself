"""NanEcho corpus/runtime surface: fail-closed path, compose, no empty corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .harmonic_resonance_esn import (
    HarmonicResonanceESN,
    OscillatorState,
    OscillatorStateError,
    persona_oscillators,
)

CANDIDATE_PATHS: tuple[str, ...] = (
    "echoself/data/nanecho",
    "NanEcho/data/nanecho",
    "data/nanecho",
)
PREPARED_EVIDENCE: tuple[str, ...] = ("train.bin", "train.txt", "metadata.json")
PERSONA_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("cognitive", 0.15),
    ("introspective", 0.15),
    ("adaptive", 0.15),
    ("recursive", 0.15),
    ("synergistic", 0.10),
    ("holographic", 0.10),
    ("neural-symbolic", 0.10),
    ("dynamic", 0.10),
)
VORTICOG_TYPES: tuple[str, ...] = ("persona", "need", "dreamcog", "erebus")
NEED_KINDS: tuple[str, ...] = (
    "energy",
    "social",
    "safety",
    "curiosity",
    "coherence",
)
PERSONA_DIMENSIONS: tuple[str, ...] = tuple(name for name, _ in PERSONA_WEIGHTS)
CHECKPOINT_CANDIDATES: tuple[str, ...] = (
    ".training-progress/checkpoints/latest_checkpoint.pt",
    "out-nanecho/best_model.pt",
    "out-nanecho/ckpt.pt",
)
SUMMARY_504 = Path(".training-progress/nanecho-cached-ci/training_summary.json")
SUMMARY_827 = Path(".training-progress/nanecho-cached-ci/training_summary (2).json")
BACKUP_695 = Path(".training-progress/checkpoints/backup_manifest.json")


class SurfaceError(ValueError):
    """Invalid NanEcho surface or Vorticog construction."""


@dataclass(frozen=True)
class SurfaceRef:
    kind: Literal["runtime_surface", "prepared", "failed", "missing"]
    path: str | None
    reason: str | None = None
    tried: tuple[str, ...] = ()


@dataclass(frozen=True)
class VorticogAgent:
    type: str
    dimension: str | None = None
    need: str | None = None

    def __post_init__(self) -> None:
        if not self.type:
            raise SurfaceError("Vorticog agent requires a type")
        if self.type not in VORTICOG_TYPES:
            raise SurfaceError(
                f"unknown Vorticog agent type {self.type!r}; allowed: {', '.join(VORTICOG_TYPES)}"
            )
        if self.type == "persona":
            if self.dimension not in PERSONA_DIMENSIONS:
                raise SurfaceError("persona agent requires a PersonaDimension")
        if self.type == "need":
            if self.need not in NEED_KINDS:
                raise SurfaceError("need agent requires a closed need kind")


@dataclass(frozen=True)
class TinyInferenceResult:
    kind: Literal["generated", "unavailable"]
    reason: str
    text: str | None = None
    backend: str = "nanecho_runtime"


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_nanecho_surface(root: Path | None = None) -> SurfaceRef:
    base = root or repo_root_from_here()
    tried: list[str] = []
    for relative in CANDIDATE_PATHS:
        tried.append(relative)
        candidate = base / relative
        if not candidate.is_dir():
            continue
        if any((candidate / name).is_file() for name in PREPARED_EVIDENCE):
            return SurfaceRef(kind="prepared", path=relative)
        return SurfaceRef(
            kind="failed",
            path=relative,
            reason=(
                f"{relative} exists as a runtime surface but has no prepared "
                "corpus (train.bin|train.txt|metadata.json)"
            ),
            tried=tuple(tried),
        )
    return SurfaceRef(kind="missing", path=None, tried=tuple(tried), reason="no candidate directory exists")


def find_local_checkpoint(root: Path | None = None) -> Path | None:
    base = root or repo_root_from_here()
    for relative in CHECKPOINT_CANDIDATES:
        path = base / relative
        if path.is_file():
            return path
    return None


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fold_lineage(root: Path | None = None) -> dict[str, Any]:
    """Report live lineage from named files. Do not invent metrics."""
    base = root or repo_root_from_here()
    summary_504 = _read_json(base / SUMMARY_504)
    summary_827 = _read_json(base / SUMMARY_827)
    backup_695 = _read_json(base / BACKUP_695)
    latest = (base / CHECKPOINT_CANDIDATES[0]).is_file()
    heads: list[dict[str, Any]] = []
    if summary_504 and isinstance(summary_504.get("best_checkpoint"), dict):
        best = summary_504["best_checkpoint"]
        heads.append(
            {
                "generation": "504",
                "evidence": str(SUMMARY_504).replace("\\", "/"),
                "id": best.get("id"),
                "iteration": best.get("iteration"),
                "val_loss": best.get("val_loss"),
            }
        )
    if summary_827 and isinstance(summary_827.get("best_checkpoint"), dict):
        best = summary_827["best_checkpoint"]
        heads.append(
            {
                "generation": "827",
                "evidence": str(SUMMARY_827).replace("\\", "/"),
                "id": best.get("id"),
                "iteration": best.get("iteration"),
                "val_loss": best.get("val_loss"),
            }
        )
    if backup_695:
        heads.append(
            {
                "generation": "695",
                "evidence": str(BACKUP_695).replace("\\", "/"),
                "id": None,
                "iteration": backup_695.get("iteration"),
                "val_loss": backup_695.get("val_loss"),
            }
        )
    if len(heads) >= 2:
        verdict = "divergent"
        readiness = "restore_required"
    elif heads and not latest:
        verdict = "divergent" if len(heads) > 1 else "metadata_only"
        readiness = "restore_required"
    elif not heads and not latest:
        verdict = "uninitialized"
        readiness = "restore_required"
    else:
        verdict = "coherent"
        readiness = "restore_required" if not latest else "restore_required"
    return {
        "kind": verdict,
        "readiness": readiness,
        "latest_checkpoint_present": latest,
        "generation_827_on_disk": summary_827 is not None,
        "heads": heads,
        "force_fresh_start": False,
        "command": "restore" if readiness == "restore_required" else "respond",
    }


def tiny_infer(prompt: str, root: Path | None = None) -> TinyInferenceResult:
    checkpoint = find_local_checkpoint(root)
    if checkpoint is None:
        return TinyInferenceResult(
            kind="unavailable",
            reason="no local NanEcho .pt checkpoint; refusing model download",
        )
    if not prompt.strip():
        return TinyInferenceResult(kind="unavailable", reason="empty prompt")
    return TinyInferenceResult(
        kind="unavailable",
        reason=(
            f"checkpoint path present at {checkpoint.as_posix()} but this slice "
            "does not load weights or start training"
        ),
        backend="nanecho_runtime",
    )


def make_vorticog_agent(
    type: str,
    *,
    dimension: str | None = None,
    need: str | None = None,
) -> VorticogAgent:
    return VorticogAgent(type=type, dimension=dimension, need=need)


def compose_surface(
    root: Path | None = None,
    *,
    salience_text: str = "Deep Tree Echo keeps eight persona dimensions in one registry.",
) -> dict[str, Any]:
    base = root or repo_root_from_here()
    surface = resolve_nanecho_surface(base)
    lineage = fold_lineage(base)
    if surface.kind == "prepared":
        corpus: dict[str, Any] = {"kind": "prepared", "source": surface.path}
    else:
        corpus = {
            "kind": "failed",
            "reason": surface.reason,
            "fallbackCorpus": None,
            "path": surface.path,
        }

    from echogenesis import initialize_generation

    genesis = initialize_generation()
    generation = genesis.evolve(
        [
            {
                "text": salience_text,
                "aspect": "cognitive",
                "salience": 0.7,
            }
        ],
        max_fragments=2,
    )

    oscillators = persona_oscillators([weight for _, weight in PERSONA_WEIGHTS])
    reservoir = HarmonicResonanceESN(oscillators)
    after = reservoir.step([weight for _, weight in PERSONA_WEIGHTS])
    inference = tiny_infer(salience_text, base)
    agents = (
        make_vorticog_agent("persona", dimension="cognitive"),
        make_vorticog_agent("need", need="coherence"),
        make_vorticog_agent("dreamcog"),
        make_vorticog_agent("erebus"),
    )
    memory_proposals = (
        {
            "type": "autognosic",
            "content": "Local mech0 is the memory backend; cloud Mem0 is not required.",
            "applied": False,
        },
        {
            "type": "semantic",
            "content": "echoself/data/nanecho is the runtime surface; prepared bins are absent.",
            "applied": False,
        },
    )
    return {
        "surface": {
            "kind": surface.kind,
            "path": surface.path,
            "reason": surface.reason,
            "tried": list(surface.tried),
        },
        "corpus": corpus,
        "lineage": lineage,
        "training_command": "restore",
        "echogenesis": {
            "fragment_count": len(generation.fragments),
            "example_count": len(generation.training_examples),
        },
        "harmonic": {
            "phases": list(after.phases),
            "amplitudes": list(after.amplitudes),
            "readout": reservoir.readout(),
        },
        "inference": {
            "kind": inference.kind,
            "reason": inference.reason,
            "backend": inference.backend,
        },
        "vorticog": {
            "agents": [
                {"type": agent.type, "dimension": agent.dimension, "need": agent.need}
                for agent in agents
            ],
            "memory_proposals": list(memory_proposals),
        },
    }


__all__ = [
    "CANDIDATE_PATHS",
    "HarmonicResonanceESN",
    "OscillatorState",
    "OscillatorStateError",
    "PERSONA_WEIGHTS",
    "SurfaceError",
    "SurfaceRef",
    "TinyInferenceResult",
    "VorticogAgent",
    "compose_surface",
    "find_local_checkpoint",
    "fold_lineage",
    "make_vorticog_agent",
    "persona_oscillators",
    "repo_root_from_here",
    "resolve_nanecho_surface",
    "tiny_infer",
]
