"""Rolling, observable persona drift monitoring for generated NanEcho text."""

from __future__ import annotations

from collections import deque
from statistics import mean
from typing import Deque, Dict


DIMENSION_TERMS = {
    "cognitive": ("reason", "analysis", "cognitive"),
    "introspective": ("introspect", "reflect", "self-examination"),
    "adaptive": ("adaptive", "attention", "threshold"),
    "recursive": ("recursive", "recursion", "multi-level"),
    "synergistic": ("synergy", "emergent", "interaction"),
    "holographic": ("holographic", "whole", "perspective"),
    "neural_symbolic": ("neural-symbolic", "symbolic", "hypergraph"),
    "dynamic": ("dynamic", "evolve", "continuous"),
}


def score_persona_text(text: str) -> Dict[str, float]:
    lowered = text.lower()
    return {
        dimension: sum(term in lowered for term in terms) / len(terms)
        for dimension, terms in DIMENSION_TERMS.items()
    }


class PersonaDriftMonitor:
    """Tracks transparent lexical persona coverage; it does not claim model quality."""

    def __init__(self, window_size: int = 100, drift_threshold: float = 0.20) -> None:
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self._history: Deque[Dict[str, float]] = deque(maxlen=window_size)

    def observe(self, text: str) -> Dict[str, float]:
        scores = score_persona_text(text)
        self._history.append(scores)
        return scores

    def snapshot(self) -> Dict[str, object]:
        averages = {
            dimension: mean(item[dimension] for item in self._history)
            if self._history
            else 0.0
            for dimension in DIMENSION_TERMS
        }
        underrepresented = [
            dimension
            for dimension, score in averages.items()
            if self._history and score < self.drift_threshold
        ]
        return {
            "sample_count": len(self._history),
            "window_size": self.window_size,
            "dimension_coverage": averages,
            "drift_threshold": self.drift_threshold,
            "underrepresented_dimensions": underrepresented,
            "method": "rolling lexical coverage of generated text",
        }
