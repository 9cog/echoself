"""Generate corpus text consumable by NanEcho's existing preparation pipeline."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .fragment_synthesizer import IdentityFragment
from .refinement_engine import Refinement


@dataclass
class TrainingExample:
    source_id: str
    category: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_id = self.source_id.strip()
        self.category = self.category.strip().lower()
        self.text = " ".join(self.text.split())
        self.metadata = dict(self.metadata)
        if not self.source_id or not self.category or not self.text:
            raise ValueError("training source_id, category, and text cannot be empty")

    def render(self) -> str:
        return f"[ECHOGENESIS:{self.category.upper()}]\n{self.text}"


class TrainingDataGenerator:
    """Convert generated identity artifacts into a plain-text training corpus."""

    def generate(
        self,
        fragments: Iterable[IdentityFragment],
        refinements: Iterable[Refinement] = (),
    ) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        seen: set[tuple[str, str]] = set()

        for fragment in fragments:
            key = ("identity_fragment", fragment.id)
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                TrainingExample(
                    source_id=fragment.id,
                    category=f"identity_{fragment.aspect}",
                    text=fragment.content,
                    metadata={"salience": fragment.salience},
                )
            )

        for refinement in refinements:
            key = (refinement.refinement_type.value, refinement.id)
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                TrainingExample(
                    source_id=refinement.id,
                    category=f"refinement_{refinement.refinement_type.value}",
                    text=refinement.content,
                    metadata={"confidence": refinement.confidence},
                )
            )

        return examples

    def render_corpus(self, examples: Iterable[TrainingExample]) -> str:
        rendered = [example.render() for example in examples]
        return "\n\n".join(rendered) + ("\n" if rendered else "")

    def write_corpus(
        self,
        path: os.PathLike[str] | str,
        examples: Iterable[TrainingExample],
        append: bool = False,
    ) -> None:
        target = Path(path)
        rendered = [example.render() for example in examples]
        if append and target.exists():
            existing = target.read_text(encoding="utf-8")
            existing_blocks = set(existing.strip().split("\n\n"))
            rendered = [block for block in rendered if block not in existing_blocks]
            content = existing.rstrip()
            if rendered:
                content += ("\n\n" if content else "") + "\n\n".join(rendered)
            content += "\n" if content else ""
        else:
            content = "\n\n".join(rendered) + ("\n" if rendered else "")

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = handle.name
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)


__all__ = ["TrainingDataGenerator", "TrainingExample"]
