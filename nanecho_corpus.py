"""Source-group train/val split and the curated manifest the data audit reads.

Groups are assigned before token streams are concatenated. The manifest records
the groups that were actually written. It does not invent a reviewer: pass
``reviewed_by`` only when a person has reviewed the corpus.
"""

from __future__ import annotations

import hashlib
import json
import sys
from array import array
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

SCHEMA = "nanecho-curated-split-v1"


def assign_source_groups(names: Sequence[str]) -> Tuple[list[str], list[str]]:
    """Hold out one whole group for validation. Groups never appear in both splits."""
    ordered = sorted({name.strip() for name in names if name and name.strip()})
    if len(ordered) < 2:
        raise ValueError("source-group split needs at least two distinct groups")
    return ordered[:-1], [ordered[-1]]


def _write_tokens(path: Path, tokens: Sequence[int]) -> str:
    raw = array("H", (int(token) for token in tokens))
    if sys.byteorder != "little":
        raw.byteswap()
    with path.open("wb") as handle:
        raw.tofile(handle)
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_source_group_split(
    output_dir: Path,
    group_tokens: Mapping[str, Sequence[int]],
    *,
    tokenizer: Mapping[str, object],
    reviewed_by: str = "",
    source_revision: str = "",
    synthetic_samples: int = 0,
    extra_metadata: Optional[Mapping[str, object]] = None,
    min_split_tokens: int = 1,
) -> Tuple[int, int]:
    """Write train.bin, val.bin, metadata.json, and source_manifest.json.

    Returns (0, 0) when the groups cannot form two nonempty disjoint splits.
    """
    output_dir = Path(output_dir)
    usable = {
        name: list(tokens)
        for name, tokens in group_tokens.items()
        if tokens
    }
    try:
        train_groups, val_groups = assign_source_groups(list(usable))
    except ValueError:
        return 0, 0
    train_ids = [token for name in train_groups for token in usable[name]]
    val_ids = [token for name in val_groups for token in usable[name]]
    if len(train_ids) < min_split_tokens or len(val_ids) < min_split_tokens:
        return 0, 0
    if any(token < 0 or token > 0xFFFF for token in train_ids + val_ids):
        raise ValueError("token id does not fit in uint16")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_sha = _write_tokens(output_dir / "train.bin", train_ids)
    val_sha = _write_tokens(output_dir / "val.bin", val_ids)
    revision = source_revision.strip() or hashlib.sha256(
        json.dumps(
            {name: usable[name][:32] for name in sorted(usable)},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    token_meta = dict(tokenizer)
    manifest: Dict[str, object] = {
        "schema": SCHEMA,
        "reviewed_by": reviewed_by,
        "source_revision": revision,
        "synthetic_samples": int(synthetic_samples),
        "tokenizer": token_meta,
        "train": {
            "sha256": train_sha,
            "token_count": len(train_ids),
            "source_groups": train_groups,
        },
        "val": {
            "sha256": val_sha,
            "token_count": len(val_ids),
            "source_groups": val_groups,
        },
    }
    metadata: Dict[str, object] = {
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "vocab_size": token_meta.get("vocab_size"),
        "tokenizer": token_meta,
        "synthetic_samples": int(synthetic_samples),
        "split_strategy": "source-group-before-tokenization",
        "source_groups": {
            "train": train_groups,
            "val": val_groups,
        },
        "deterministic": True,
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
        # The audit compares these fields with the manifest. Keep them honest.
        metadata["train_tokens"] = len(train_ids)
        metadata["val_tokens"] = len(val_ids)
        metadata["tokenizer"] = token_meta
        metadata["synthetic_samples"] = int(synthetic_samples)
        metadata["vocab_size"] = token_meta.get("vocab_size")
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(train_ids), len(val_ids)
