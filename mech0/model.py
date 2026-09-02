"""Typed memory domain for local ech0-mem0 / mech0.

Memory type is a closed registry (semantic | episodic | procedural | autognosic).
Records are constructed only through the registry so these states cannot exist:

- a memory with no type
- the same fact stored as two types without DualWriteCommand
- autognosic entries that are not about the system/self
- procedural entries without a callable/procedure shape
- cloud-only as the sole backend (this module has no cloud config)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Mapping, Sequence

MemoryType = Literal["semantic", "episodic", "procedural", "autognosic"]
MEMORY_TYPES: tuple[MemoryType, ...] = (
    "semantic",
    "episodic",
    "procedural",
    "autognosic",
)

SelfAspect = Literal["identity", "capability", "checkout", "belief", "self_model"]
SELF_ASPECTS: frozenset[str] = frozenset(
    {"identity", "capability", "checkout", "belief", "self_model"}
)

_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?$"
)
_INSTRUMENT_RE = re.compile(r"^[A-Za-z_][\w./:-]*$")


class MemoryTypeError(ValueError):
    """Unknown or missing memory type."""


class DualWriteRequired(ValueError):
    """Same fact already lives under another type; use DualWriteCommand."""


class AutognosicSubjectError(ValueError):
    """Autognosic memories must be about the system/self."""


class ProceduralShapeError(ValueError):
    """Procedural memories must name a callable instrument."""


class MemoryValidationError(ValueError):
    """Payload failed type-specific validation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_memory_type(value: Any) -> MemoryType:
    if value is None or value == "":
        raise MemoryTypeError("Memory type is required (semantic|episodic|procedural|autognosic)")
    if not isinstance(value, str):
        raise MemoryTypeError(f"Memory type must be a string, got {type(value).__name__}")
    name = value.strip().lower()
    if name not in MEMORY_TYPES:
        raise MemoryTypeError(
            f"Unknown memory type {value!r}. Allowed: {', '.join(MEMORY_TYPES)}"
        )
    return name  # type: ignore[return-value]


@dataclass(frozen=True)
class SemanticSpec:
    kind: Literal["semantic"] = "semantic"
    concepts: tuple[str, ...] = ()
    weights: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "concepts": list(self.concepts),
            "weights": {k: v for k, v in self.weights},
        }


@dataclass(frozen=True)
class EpisodicSpec:
    kind: Literal["episodic"] = "episodic"
    occurred_at: str = ""
    event: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "occurred_at": self.occurred_at,
            "event": self.event,
        }


@dataclass(frozen=True)
class ProceduralSpec:
    kind: Literal["procedural"] = "procedural"
    instrument: str = ""
    signature: str | None = None
    steps: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "instrument": self.instrument,
            "signature": self.signature,
            "steps": list(self.steps),
        }


@dataclass(frozen=True)
class AutognosicSpec:
    kind: Literal["autognosic"] = "autognosic"
    about: SelfAspect = "self_model"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "about": self.about}


TypeSpec = SemanticSpec | EpisodicSpec | ProceduralSpec | AutognosicSpec


def _as_str_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        item = value.strip()
        return (item,) if item else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise MemoryValidationError(f"{field_name} must be a list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise MemoryValidationError(f"{field_name} entries must be non-empty strings")
        out.append(item.strip())
    return tuple(out)


def _as_weight_pairs(value: Any) -> tuple[tuple[str, float], ...]:
    if value is None or value == {}:
        return ()
    if not isinstance(value, Mapping):
        raise MemoryValidationError("semantic.weights must be an object of name -> number")
    pairs: list[tuple[str, float]] = []
    for key, raw in value.items():
        if not isinstance(key, str) or not key.strip():
            raise MemoryValidationError("weight keys must be non-empty strings")
        try:
            pairs.append((key.strip(), float(raw)))
        except (TypeError, ValueError) as exc:
            raise MemoryValidationError(f"weight {key!r} must be numeric") from exc
    return tuple(pairs)


def validate_semantic(payload: Mapping[str, Any]) -> SemanticSpec:
    return SemanticSpec(
        concepts=_as_str_tuple(payload.get("concepts"), "concepts"),
        weights=_as_weight_pairs(payload.get("weights")),
    )


def validate_episodic(payload: Mapping[str, Any]) -> EpisodicSpec:
    occurred_at = payload.get("occurred_at") or payload.get("occurredAt")
    event = payload.get("event")
    if not isinstance(occurred_at, str) or not occurred_at.strip():
        raise MemoryValidationError("episodic memories require occurred_at (ISO date/time)")
    stamp = occurred_at.strip()
    if not _ISO_RE.match(stamp):
        raise MemoryValidationError(f"episodic.occurred_at is not an ISO date/time: {stamp!r}")
    if not isinstance(event, str) or not event.strip():
        raise MemoryValidationError("episodic memories require an event name")
    return EpisodicSpec(occurred_at=stamp, event=event.strip())


def validate_procedural(payload: Mapping[str, Any]) -> ProceduralSpec:
    instrument = payload.get("instrument") or payload.get("callable")
    if not isinstance(instrument, str) or not instrument.strip():
        raise ProceduralShapeError(
            "procedural memories require instrument (callable name), e.g. checkpoint_guardian.restore"
        )
    name = instrument.strip()
    if not _INSTRUMENT_RE.match(name):
        raise ProceduralShapeError(f"procedural.instrument is not a callable identifier: {name!r}")
    signature = payload.get("signature")
    if signature is not None and (not isinstance(signature, str) or not signature.strip()):
        raise ProceduralShapeError("procedural.signature must be a non-empty string when set")
    steps = _as_str_tuple(payload.get("steps"), "steps")
    if signature is None and not steps:
        raise ProceduralShapeError(
            "procedural memories need a callable shape: signature and/or steps"
        )
    return ProceduralSpec(
        instrument=name,
        signature=signature.strip() if isinstance(signature, str) else None,
        steps=steps,
    )


def validate_autognosic(payload: Mapping[str, Any]) -> AutognosicSpec:
    about = payload.get("about")
    if not isinstance(about, str) or not about.strip():
        raise AutognosicSubjectError(
            "autognosic memories require about=identity|capability|checkout|belief|self_model"
        )
    aspect = about.strip().lower()
    if aspect not in SELF_ASPECTS:
        raise AutognosicSubjectError(
            f"autognosic.about {about!r} is not a self/system aspect. "
            f"Allowed: {', '.join(sorted(SELF_ASPECTS))}"
        )
    subject = payload.get("subject")
    if subject is not None:
        if not isinstance(subject, str) or subject.strip().lower() not in {
            "self",
            "system",
            "echoself",
            "deep tree echo",
            "nanecho",
            "mech0",
        }:
            raise AutognosicSubjectError(
                "autognosic subject must be the system/self (self|system|echoself|deep tree echo|nanecho|mech0)"
            )
    return AutognosicSpec(about=aspect)  # type: ignore[arg-type]


TypeValidator = Callable[[Mapping[str, Any]], TypeSpec]


class MemoryTypeRegistry:
    """Single collection of type handlers — not four parallel if/else piles."""

    def __init__(self) -> None:
        self._validators: dict[MemoryType, TypeValidator] = {
            "semantic": validate_semantic,
            "episodic": validate_episodic,
            "procedural": validate_procedural,
            "autognosic": validate_autognosic,
        }

    def types(self) -> tuple[MemoryType, ...]:
        return MEMORY_TYPES

    def validate(self, memory_type: MemoryType, payload: Mapping[str, Any] | None) -> TypeSpec:
        handler = self._validators.get(memory_type)
        if handler is None:
            raise MemoryTypeError(f"No handler registered for {memory_type!r}")
        return handler(payload or {})

    def spec_from_dict(self, data: Mapping[str, Any]) -> TypeSpec:
        kind = parse_memory_type(data.get("kind") or data.get("type"))
        return self.validate(kind, data)


REGISTRY = MemoryTypeRegistry()


@dataclass(frozen=True)
class DualWriteCommand:
    """Explicit permission to store one fact under exactly two types."""

    types: tuple[MemoryType, MemoryType]
    content: str
    source: str
    confidence: float = 0.8
    pinned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    specs: dict[MemoryType, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.types) != 2 or self.types[0] == self.types[1]:
            raise MemoryValidationError("dual-write requires exactly two distinct types")
        for item in self.types:
            parse_memory_type(item)
        if not self.content.strip():
            raise MemoryValidationError("dual-write content is required")


@dataclass
class MemoryRecord:
    id: str
    type: MemoryType
    content: str
    created_at: str
    confidence: float
    pinned: bool
    source: str
    metadata: dict[str, Any]
    spec: TypeSpec
    dual_write_id: str | None = None

    def __post_init__(self) -> None:
        if self.type != self.spec.kind:
            raise MemoryTypeError(
                f"Record type {self.type!r} does not match spec.kind {self.spec.kind!r}"
            )
        if not self.content.strip():
            raise MemoryValidationError("content is required")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise MemoryValidationError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "created_at": self.created_at,
            "confidence": self.confidence,
            "pinned": self.pinned,
            "source": self.source,
            "metadata": self.metadata,
            "spec": self.spec.to_dict(),
            "dual_write_id": self.dual_write_id,
        }

    @classmethod
    def create(
        cls,
        *,
        memory_type: Any,
        content: str,
        source: str,
        payload: Mapping[str, Any] | None = None,
        confidence: float = 0.8,
        pinned: bool = False,
        metadata: Mapping[str, Any] | None = None,
        record_id: str | None = None,
        created_at: str | None = None,
        dual_write_id: str | None = None,
        registry: MemoryTypeRegistry | None = None,
    ) -> MemoryRecord:
        kind = parse_memory_type(memory_type)
        spec = (registry or REGISTRY).validate(kind, payload)
        return cls(
            id=record_id or str(uuid.uuid4()),
            type=kind,
            content=content.strip(),
            created_at=created_at or utc_now(),
            confidence=float(confidence),
            pinned=bool(pinned),
            source=source.strip() or "unknown",
            metadata=dict(metadata or {}),
            spec=spec,
            dual_write_id=dual_write_id,
        )
