"""Local ech0-mem0 / mech0 — self-hosted EchoSelf memory (no cloud Mem0)."""

from .client import Mech0Client
from .model import (
    MEMORY_TYPES,
    AutognosicSpec,
    DualWriteCommand,
    EpisodicSpec,
    MemoryRecord,
    MemoryType,
    MemoryTypeRegistry,
    ProceduralSpec,
    SemanticSpec,
)

__all__ = [
    "MEMORY_TYPES",
    "AutognosicSpec",
    "DualWriteCommand",
    "EpisodicSpec",
    "Mech0Client",
    "MemoryRecord",
    "MemoryType",
    "MemoryTypeRegistry",
    "ProceduralSpec",
    "SemanticSpec",
]
