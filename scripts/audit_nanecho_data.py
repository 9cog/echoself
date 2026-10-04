#!/usr/bin/env python3
"""Pre-training provenance gate; integrity success is NOT identity improvement.

The source-group manifest must be independently reviewed. A JSON claim of distinct
source groups is not proof of authorship or a substitute for an unseen test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from array import array
from pathlib import Path

SCHEMA = "nanecho-curated-split-v1"
WINDOW = 64
MAX_BYTES_PER_SPLIT = 64 * 1024 * 1024  # fail closed; use a streaming audit for larger corpora


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tokens(path: Path) -> array:
    import sys
    raw = array("H")
    with path.open("rb") as source:
        raw.fromfile(source, path.stat().st_size // 2)
    if sys.byteorder != "little":
        raw.byteswap()
    return raw


def _rolling_windows(tokens: array):
    """O(n) memory-compact rolling 64-token windows; hash collisions fail closed."""
    if len(tokens) < WINDOW:
        return
    mask = (1 << 64) - 1
    base = 65599
    power = pow(base, WINDOW - 1, 1 << 64)
    value = 0
    for token in tokens[:WINDOW]:
        value = (value * base + int(token) + 1) & mask
    yield value
    for i in range(WINDOW, len(tokens)):
        value = ((value - (int(tokens[i - WINDOW]) + 1) * power) * base + int(tokens[i]) + 1) & mask
        yield value


def audit(data_dir: Path, block_size: int = 1024, model_vocab: int = 50257) -> dict:
    errors: list[str] = []
    result: dict = {"schema": SCHEMA, "passed": False, "errors": errors}
    try:
        metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
        manifest = json.loads((data_dir / "source_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"metadata.json and independently reviewed source_manifest.json required: {type(exc).__name__}")
        return result
    if manifest.get("schema") != SCHEMA:
        errors.append("unrecognized source manifest schema")
    if not manifest.get("reviewed_by") or not manifest.get("source_revision"):
        errors.append("source manifest requires reviewed_by and source_revision")
    if metadata.get("created_by") == "workflow_fallback" or metadata.get("fallback_mode") or metadata.get("synthetic_samples", 0) != 0 or manifest.get("synthetic_samples", 0) != 0:
        errors.append("fallback/repeated/synthetic samples are not eligible for identity training")
    tok = metadata.get("tokenizer", {})
    if not isinstance(tok, dict) or not all(k in tok for k in ("name", "vocab_size", "eos_token_id")):
        errors.append("tokenizer provenance is missing")
    elif not isinstance(tok["vocab_size"], int) or tok["vocab_size"] > model_vocab or tok["vocab_size"] != metadata.get("vocab_size"):
        errors.append("tokenizer and model vocabulary mismatch")
    if manifest.get("tokenizer") != tok:
        errors.append("source manifest tokenizer disagrees with dataset metadata")
    if isinstance(tok, dict) and tok.get("name") == "dte_bpe":
        tokenizer_path = Path(__file__).resolve().parents[1] / "NanEcho/dte_tokenizer/tokenizer.json"
        try:
            tokenizer_sha = hashlib.sha256(json.dumps(
                json.loads(tokenizer_path.read_text(encoding="utf-8")), sort_keys=True,
            ).encode()).hexdigest()
        except (OSError, ValueError):
            tokenizer_sha = ""
        if (not tokenizer_sha or tok.get("tokenizer_sha256") != tokenizer_sha
                or metadata.get("tokenizer_semantic_sha256") != tokenizer_sha
                or manifest.get("tokenizer_semantic_sha256") != tokenizer_sha):
            errors.append("DTE 8192 tokenizer semantic digest mismatched or missing")
        sources = manifest.get("source_files_sha256")
        if not isinstance(sources, dict) or not sources:
            errors.append("DTE source-file content hashes missing")
        else:
            root = Path(__file__).resolve().parents[1]
            for name, expected_hash in sources.items():
                path = (root / name).resolve()
                if not path.is_relative_to(root) or not path.is_file() or _sha256(path) != expected_hash:
                    errors.append(f"DTE source-file hash mismatch: {name}")
            declared_groups = set()
            for record in (manifest.get("train"), manifest.get("val")):
                if isinstance(record, dict):
                    declared_groups.update(record.get("source_groups", []))
            if declared_groups != set(sources):
                errors.append("DTE source groups do not cover exactly the hashed source files")
    sequences: dict[str, array] = {}
    groups: dict[str, set[str]] = {}
    for split in ("train", "val"):
        record = manifest.get(split, {})
        if not isinstance(record, dict):
            errors.append(f"{split}: manifest record missing")
            continue
        file = data_dir / f"{split}.bin"
        if not file.is_file():
            errors.append(f"{split}: token file missing")
            continue
        size = file.stat().st_size
        if size % 2 or size > MAX_BYTES_PER_SPLIT or size <= 2 * block_size:
            errors.append(f"{split}: invalid size for uint16 token stream and block_size")
            continue
        digest = _sha256(file)
        result[f"{split}_sha256"] = digest
        if record.get("sha256") != digest or record.get("token_count") != size // 2 or metadata.get(f"{split}_tokens") != size // 2:
            errors.append(f"{split}: manifest/content token count or SHA-256 mismatch")
        source_groups = record.get("source_groups")
        if not isinstance(source_groups, list) or not source_groups or not all(isinstance(x, str) and x.strip() for x in source_groups):
            errors.append(f"{split}: explicit nonempty source_groups required")
        else:
            groups[split] = set(source_groups)
        sequence = _tokens(file)
        sequences[split] = sequence
        if isinstance(tok, dict) and isinstance(tok.get("vocab_size"), int) and sequence and max(sequence) >= tok["vocab_size"]:
            errors.append(f"{split}: token id exceeds declared vocabulary")
    if len(groups) == 2 and groups["train"] & groups["val"]:
        errors.append("train/val source groups overlap")
    if len(sequences) == 2:
        train, val = sequences["train"], sequences["val"]
        train_windows = set(_rolling_windows(train))
        val_total = max(0, len(val) - WINDOW + 1)
        fraction = sum(window in train_windows for window in _rolling_windows(val)) / max(1, val_total)
        result["overlapping_64_token_windows_fraction"] = fraction
        if fraction > 0.01:
            errors.append("train/val 64-token overlap exceeds 1% (contamination or repeated corpus)")
        result["train_tokens"] = len(train)
        result["val_tokens"] = len(val)
    result["passed"] = not errors
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--model-vocab", type=int, default=50257)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.block_size < 1 or args.model_vocab < 1:
        parser.error("block-size and model-vocab must be positive")
    report = audit(args.data_dir, args.block_size, args.model_vocab)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
