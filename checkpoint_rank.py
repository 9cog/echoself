"""Rank NanEcho checkpoints without the legacy quality score.

``quality_score`` on cached CI metadata mixes loss with ``tokens_processed``.
Promotion and cleanup must use validation loss. The legacy metric remains
available only as an audit ordering so a rescoring can say whether the
selected id moved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def _row(metadata: Mapping[str, Any], checkpoint_id: str) -> Mapping[str, Any]:
    row = metadata[checkpoint_id]
    if not isinstance(row, Mapping):
        raise TypeError(f"checkpoint {checkpoint_id} metadata must be a mapping")
    return row


def rank_checkpoint_ids(
    metadata: Mapping[str, Any],
    metric: str = "val_loss",
) -> List[str]:
    """Return ids best-first. ``quality_score`` is audit-only."""
    if metric == "quality_score":
        return sorted(
            metadata,
            key=lambda key: float(_row(metadata, key)["quality_score"]),
            reverse=True,
        )
    if metric == "val_loss":
        return sorted(
            metadata,
            key=lambda key: (
                float(_row(metadata, key)["val_loss"]),
                int(_row(metadata, key).get("iteration", 0)),
            ),
        )
    if metric == "iteration":
        return sorted(
            metadata,
            key=lambda key: int(_row(metadata, key).get("iteration", 0)),
            reverse=True,
        )
    raise ValueError(f"Unknown metric: {metric}")


def select_best_checkpoint_id(
    metadata: Mapping[str, Any],
    metric: str = "val_loss",
) -> Optional[str]:
    """Choose a checkpoint for resume or export. Never by ``quality_score``."""
    if metric == "quality_score":
        raise ValueError(
            "quality_score must not select a checkpoint; rank by val_loss"
        )
    if not metadata:
        return None
    return rank_checkpoint_ids(metadata, metric)[0]
