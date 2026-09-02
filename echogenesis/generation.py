"""Autopoietic orchestration for the EchoGenesis generation layer."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any, Iterable, Mapping, Optional

from .fragment_synthesizer import FragmentSynthesizer, IdentityFragment
from .pattern_propagator import PatternPropagator, empty_hypergraph
from .refinement_engine import Refinement, RefinementEngine
from .training_generator import TrainingDataGenerator, TrainingExample


@dataclass
class GenerationResult:
    fragments: list[IdentityFragment]
    refinements: list[Refinement]
    hypergraph: dict[str, Any]
    training_examples: list[TrainingExample]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragments": [fragment.to_dict() for fragment in self.fragments],
            "refinements": [refinement.to_dict() for refinement in self.refinements],
            "hypergraph": self.hypergraph,
            "training_examples": [
                {
                    "source_id": example.source_id,
                    "category": example.category,
                    "text": example.text,
                    "metadata": dict(example.metadata),
                }
                for example in self.training_examples
            ],
        }


class EchoGenesis:
    """Coordinate synthesis, refinement, propagation, and training output."""

    def __init__(self, integration_bias: float = 0.774):
        self.synthesizer = FragmentSynthesizer()
        self.refinement_engine = RefinementEngine(integration_bias)
        self.propagator = PatternPropagator()
        self.training_generator = TrainingDataGenerator()

    def evolve(
        self,
        salience_signals: Iterable[Mapping[str, Any]],
        existing_fragments: Iterable[IdentityFragment | Mapping[str, Any]] = (),
        evidence: Optional[Mapping[str, Mapping[str, Any]]] = None,
        hypergraph: Optional[Mapping[str, Any]] = None,
        hypergraph_path: Optional[PathLike[str] | str] = None,
        training_path: Optional[PathLike[str] | str] = None,
        append_training: bool = True,
        max_fragments: Optional[int] = None,
    ) -> GenerationResult:
        if hypergraph is not None and hypergraph_path is not None:
            raise ValueError("provide hypergraph or hypergraph_path, not both")

        existing = [
            fragment
            if isinstance(fragment, IdentityFragment)
            else IdentityFragment.from_dict(fragment)
            for fragment in existing_fragments
        ]
        fragments = self.synthesizer.synthesize(
            salience_signals, existing, max_fragments
        )
        refinements = self.refinement_engine.orchestrate(
            fragments, existing, evidence
        )

        if hypergraph_path is not None:
            graph = self.propagator.update_file(
                hypergraph_path, [*existing, *fragments], refinements
            )
        else:
            graph = self.propagator.propagate(
                hypergraph or empty_hypergraph(),
                [*existing, *fragments],
                refinements,
            )

        training_examples = self.training_generator.generate(fragments, refinements)
        if training_path is not None:
            self.training_generator.write_corpus(
                training_path, training_examples, append=append_training
            )

        return GenerationResult(
            fragments=fragments,
            refinements=refinements,
            hypergraph=graph,
            training_examples=training_examples,
        )


__all__ = ["EchoGenesis", "GenerationResult"]
