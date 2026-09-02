"""Validated propagation of identity artifacts into a JSON hypergraph."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from .fragment_synthesizer import IdentityFragment, _stable_id
from .refinement_engine import Refinement


def empty_hypergraph() -> dict[str, Any]:
    return {"nodes": [], "hyperedges": [], "metadata": {"schema_version": 1}}


class PatternPropagator:
    """Expand identity topology without duplicating nodes or hyperedges."""

    def propagate(
        self,
        hypergraph: Mapping[str, Any],
        fragments: Iterable[IdentityFragment],
        refinements: Iterable[Refinement] = (),
    ) -> dict[str, Any]:
        graph = self._validated_copy(hypergraph)
        nodes = {node["id"]: node for node in graph["nodes"]}
        hyperedges = {edge["id"]: edge for edge in graph["hyperedges"]}

        for fragment in fragments:
            nodes.setdefault(
                fragment.id,
                {
                    "id": fragment.id,
                    "type": "identity_fragment",
                    "content": fragment.content,
                    "aspect": fragment.aspect,
                    "salience": fragment.salience,
                    "metadata": dict(fragment.metadata),
                },
            )

        pending_refinements = list(refinements)
        for refinement in pending_refinements:
            missing = [source_id for source_id in refinement.source_ids if source_id not in nodes]
            if missing:
                raise ValueError(
                    f"refinement {refinement.id} references missing nodes: {missing}"
                )
            nodes.setdefault(
                refinement.id,
                {
                    "id": refinement.id,
                    "type": "identity_refinement",
                    "content": refinement.content,
                    "aspect": refinement.aspect,
                    "confidence": refinement.confidence,
                    "metadata": dict(refinement.metadata),
                },
            )
            edge_id = _stable_id(
                "hyperedge",
                refinement.refinement_type.value,
                *refinement.source_ids,
                refinement.id,
            )
            hyperedges.setdefault(
                edge_id,
                {
                    "id": edge_id,
                    "type": "refinement",
                    "relation": refinement.refinement_type.value,
                    "sources": list(refinement.source_ids),
                    "target": refinement.id,
                },
            )

        graph["nodes"] = list(nodes.values())
        graph["hyperedges"] = list(hyperedges.values())
        return graph

    def update_file(
        self,
        path: os.PathLike[str] | str,
        fragments: Iterable[IdentityFragment],
        refinements: Iterable[Refinement] = (),
    ) -> dict[str, Any]:
        target = Path(path)
        graph = self.load(target) if target.exists() else empty_hypergraph()
        updated = self.propagate(graph, fragments, refinements)
        self.save(target, updated)
        return updated

    def load(self, path: os.PathLike[str] | str) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise ValueError("hypergraph root must be an object")
        return self._validated_copy(value)

    def save(self, path: os.PathLike[str] | str, hypergraph: Mapping[str, Any]) -> None:
        target = Path(path)
        graph = self._validated_copy(hypergraph)
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
                json.dump(graph, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _validated_copy(hypergraph: Mapping[str, Any]) -> dict[str, Any]:
        graph = deepcopy(dict(hypergraph))
        graph.setdefault("nodes", [])
        graph.setdefault("hyperedges", [])
        graph.setdefault("metadata", {"schema_version": 1})
        if not isinstance(graph["nodes"], list) or not isinstance(
            graph["hyperedges"], list
        ):
            raise ValueError("hypergraph nodes and hyperedges must be lists")
        for collection_name in ("nodes", "hyperedges"):
            seen: set[str] = set()
            for item in graph[collection_name]:
                if not isinstance(item, Mapping) or not str(item.get("id", "")).strip():
                    raise ValueError(f"every {collection_name} item requires an id")
                item_id = str(item["id"])
                if item_id in seen:
                    raise ValueError(f"duplicate {collection_name} id: {item_id}")
                seen.add(item_id)
        return graph


__all__ = ["PatternPropagator", "empty_hypergraph"]
