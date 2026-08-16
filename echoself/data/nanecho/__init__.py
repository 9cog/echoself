"""
NanEcho Training Data and Cache Module
"""

from .harmonic_resonance_esn import (
    HarmonicResonanceESN,
    OscillatorState,
    OscillatorStateError,
    persona_oscillators,
)
from .surface import (
    CANDIDATE_PATHS,
    SurfaceError,
    SurfaceRef,
    TinyInferenceResult,
    VorticogAgent,
    compose_surface,
    fold_lineage,
    make_vorticog_agent,
    resolve_nanecho_surface,
    tiny_infer,
)

__all__ = [
    "CANDIDATE_PATHS",
    "CacheConfig",
    "CheckpointMetadata",
    "HarmonicResonanceESN",
    "OscillatorState",
    "OscillatorStateError",
    "SurfaceError",
    "SurfaceRef",
    "TinyInferenceResult",
    "TrainingCache",
    "VorticogAgent",
    "compose_surface",
    "fold_lineage",
    "make_vorticog_agent",
    "persona_oscillators",
    "resolve_nanecho_surface",
    "tiny_infer",
]


def __getattr__(name: str):
    if name in {"TrainingCache", "CacheConfig", "CheckpointMetadata"}:
        from .training_cache import CacheConfig, CheckpointMetadata, TrainingCache

        exports = {
            "TrainingCache": TrainingCache,
            "CacheConfig": CacheConfig,
            "CheckpointMetadata": CheckpointMetadata,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
