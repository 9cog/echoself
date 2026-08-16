"""Identity fragment synthesis from salient self-knowledge."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:20]}"


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


@dataclass
class IdentityFragment:
    """A generated, provenance-preserving statement about Echo's identity."""

    id: str
    content: str
    aspect: str
    salience: float
    source_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = self.id.strip()
        self.content = " ".join(self.content.split())
        self.aspect = self.aspect.strip().lower().replace(" ", "_")
        self.salience = float(self.salience)
        self.source_ids = tuple(str(source_id) for source_id in self.source_ids)
        self.metadata = dict(self.metadata)
        if not self.id:
            raise ValueError("fragment id cannot be empty")
        if not self.content:
            raise ValueError("fragment content cannot be empty")
        if not self.aspect:
            raise ValueError("fragment aspect cannot be empty")
        if not 0.0 <= self.salience <= 1.0:
            raise ValueError("fragment salience must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "aspect": self.aspect,
            "salience": self.salience,
            "source_ids": list(self.source_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IdentityFragment":
        return cls(
            id=str(value["id"]),
            content=str(value["content"]),
            aspect=str(value["aspect"]),
            salience=float(value["salience"]),
            source_ids=tuple(value.get("source_ids", ())),
            metadata=dict(value.get("metadata", {})),
        )


class FragmentSynthesizer:
    """Create novel identity fragments from EchoGnosis-style salience signals."""

    _TEXT_KEYS = ("content", "statement", "text")
    _SALIENCE_KEYS = ("salience", "score", "weight")

    def synthesize(
        self,
        salience_signals: Iterable[Mapping[str, Any]],
        existing_fragments: Iterable[IdentityFragment] = (),
        max_fragments: Optional[int] = None,
    ) -> list[IdentityFragment]:
        if max_fragments is not None and max_fragments < 1:
            raise ValueError("max_fragments must be positive")

        existing_content = {
            _normalize_text(fragment.content) for fragment in existing_fragments
        }
        candidates: list[tuple[float, str, str, str, dict[str, Any]]] = []
        seen_signals: set[tuple[str, str]] = set()

        for signal in salience_signals:
            if not isinstance(signal, Mapping):
                raise TypeError("salience signals must be mappings")
            source_text = self._read_text(signal)
            aspect = str(signal.get("aspect", "cognitive")).strip().lower()
            if not aspect:
                raise ValueError("signal aspect cannot be empty")
            salience = self._read_salience(signal)
            signal_key = (aspect, _normalize_text(source_text))
            if signal_key in seen_signals:
                continue
            seen_signals.add(signal_key)

            source_id = str(
                signal.get("id")
                or _stable_id("signal", aspect, source_text)
            )
            metadata = dict(signal.get("metadata", {}))
            candidates.append((salience, source_id, aspect, source_text, metadata))

        candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
        fragments: list[IdentityFragment] = []
        for salience, source_id, aspect, source_text, metadata in candidates:
            content = (
                f"My {aspect.replace('_', ' ')} identity integrates this salient "
                f"pattern: {source_text}"
            )
            normalized = _normalize_text(content)
            if normalized in existing_content:
                continue
            fragment = IdentityFragment(
                id=_stable_id("fragment", aspect, content, source_id),
                content=content,
                aspect=aspect,
                salience=salience,
                source_ids=(source_id,),
                metadata=metadata,
            )
            fragments.append(fragment)
            existing_content.add(normalized)
            if max_fragments is not None and len(fragments) >= max_fragments:
                break

        return fragments

    def _read_text(self, signal: Mapping[str, Any]) -> str:
        for key in self._TEXT_KEYS:
            value = signal.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
        raise ValueError("salience signal requires non-empty content, statement, or text")

    def _read_salience(self, signal: Mapping[str, Any]) -> float:
        value: Any = 0.5
        for key in self._SALIENCE_KEYS:
            if key in signal:
                value = signal[key]
                break
        try:
            salience = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("signal salience must be numeric") from error
        if not 0.0 <= salience <= 1.0:
            raise ValueError("signal salience must be between 0 and 1")
        return salience


__all__ = ["FragmentSynthesizer", "IdentityFragment"]
