"""Identity refinement through integration, elaboration, and correction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional

from .fragment_synthesizer import IdentityFragment, _stable_id


class RefinementType(str, Enum):
    INTEGRATION = "integration"
    ELABORATION = "elaboration"
    CORRECTION = "correction"


@dataclass
class Refinement:
    """A traceable transformation of one or more identity fragments."""

    id: str
    refinement_type: RefinementType
    content: str
    aspect: str
    source_ids: tuple[str, ...]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = self.id.strip()
        self.refinement_type = RefinementType(self.refinement_type)
        self.content = " ".join(self.content.split())
        self.aspect = self.aspect.strip().lower().replace(" ", "_")
        self.source_ids = tuple(str(source_id) for source_id in self.source_ids)
        self.confidence = float(self.confidence)
        self.metadata = dict(self.metadata)
        if not self.id or not self.content or not self.aspect:
            raise ValueError("refinement id, content, and aspect cannot be empty")
        if not self.source_ids:
            raise ValueError("refinement requires at least one source")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("refinement confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "refinement_type": self.refinement_type.value,
            "content": self.content,
            "aspect": self.aspect,
            "source_ids": list(self.source_ids),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


class RefinementEngine:
    """Orchestrate semantically justified identity refinements."""

    def __init__(self, integration_bias: float = 0.774):
        if not 0.0 <= integration_bias <= 1.0:
            raise ValueError("integration_bias must be between 0 and 1")
        self.integration_bias = float(integration_bias)

    def integrate(
        self, fragments: Iterable[IdentityFragment], aspect: Optional[str] = None
    ) -> Refinement:
        sources = list(fragments)
        if len(sources) < 2:
            raise ValueError("integration requires at least two fragments")
        source_ids = tuple(fragment.id for fragment in sources)
        target_aspect = aspect or self._shared_aspect(sources)
        joined = " ".join(fragment.content for fragment in sources)
        content = (
            f"My {target_aspect.replace('_', ' ')} identity coherently integrates "
            f"these patterns: {joined}"
        )
        confidence = sum(fragment.salience for fragment in sources) / len(sources)
        return self._create(
            RefinementType.INTEGRATION,
            content,
            target_aspect,
            source_ids,
            confidence,
        )

    def elaborate(
        self, fragment: IdentityFragment, evidence: str, confidence: Optional[float] = None
    ) -> Refinement:
        detail = " ".join(evidence.split())
        if not detail:
            raise ValueError("elaboration evidence cannot be empty")
        return self._create(
            RefinementType.ELABORATION,
            f"{fragment.content} This is further grounded by: {detail}",
            fragment.aspect,
            (fragment.id,),
            fragment.salience if confidence is None else confidence,
        )

    def correct(
        self,
        fragment: IdentityFragment,
        corrected_content: str,
        confidence: Optional[float] = None,
    ) -> Refinement:
        content = " ".join(corrected_content.split())
        if not content:
            raise ValueError("corrected content cannot be empty")
        return self._create(
            RefinementType.CORRECTION,
            content,
            fragment.aspect,
            (fragment.id,),
            fragment.salience if confidence is None else confidence,
            {"supersedes": fragment.id},
        )

    def orchestrate(
        self,
        fragments: Iterable[IdentityFragment],
        existing_fragments: Iterable[IdentityFragment] = (),
        evidence: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> list[Refinement]:
        generated = list(fragments)
        existing = list(existing_fragments)
        evidence = evidence or {}
        refinements: list[Refinement] = []

        for fragment in generated:
            item = evidence.get(fragment.id)
            if item:
                kind = RefinementType(item.get("type", RefinementType.ELABORATION))
                confidence = item.get("confidence")
                if kind is RefinementType.CORRECTION:
                    refinements.append(
                        self.correct(fragment, str(item.get("content", "")), confidence)
                    )
                elif kind is RefinementType.ELABORATION:
                    refinements.append(
                        self.elaborate(fragment, str(item.get("content", "")), confidence)
                    )

        by_aspect: dict[str, list[IdentityFragment]] = {}
        for fragment in [*existing, *generated]:
            by_aspect.setdefault(fragment.aspect, []).append(fragment)
        generated_ids = {fragment.id for fragment in generated}
        for aspect, related in by_aspect.items():
            if len(related) < 2 or not generated_ids.intersection(
                fragment.id for fragment in related
            ):
                continue
            refinements.append(self.integrate(related, aspect))

        unique: dict[str, Refinement] = {}
        for refinement in refinements:
            unique.setdefault(refinement.id, refinement)
        return list(unique.values())

    def select_refinement_type(self, key: str) -> RefinementType:
        """Return a reproducible scheduling suggestion using the integration bias."""
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        sample = int.from_bytes(digest[:8], "big") / (2**64 - 1)
        if sample < self.integration_bias:
            return RefinementType.INTEGRATION
        midpoint = self.integration_bias + (1.0 - self.integration_bias) / 2.0
        if sample < midpoint:
            return RefinementType.ELABORATION
        return RefinementType.CORRECTION

    def _create(
        self,
        refinement_type: RefinementType,
        content: str,
        aspect: str,
        source_ids: tuple[str, ...],
        confidence: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Refinement:
        return Refinement(
            id=_stable_id(
                "refinement",
                refinement_type.value,
                aspect,
                content,
                *source_ids,
            ),
            refinement_type=refinement_type,
            content=content,
            aspect=aspect,
            source_ids=source_ids,
            confidence=confidence,
            metadata=metadata or {},
        )

    @staticmethod
    def _shared_aspect(fragments: list[IdentityFragment]) -> str:
        aspects = {fragment.aspect for fragment in fragments}
        return aspects.pop() if len(aspects) == 1 else "synergistic"


RefinementOrchestrator = RefinementEngine

__all__ = [
    "Refinement",
    "RefinementEngine",
    "RefinementOrchestrator",
    "RefinementType",
]
