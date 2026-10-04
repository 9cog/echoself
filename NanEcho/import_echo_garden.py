#!/usr/bin/env python3
"""Stage Echo Garden's distinct authored vision documents for DTE-8192 training.

The Garden's two JSONL families are already present in EchoSelf; verify their
normalized content and DO NOT duplicate them. Never dispatch training or push
memory text from this importer. Output is an ignored, local source-family file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import quote
from urllib.request import urlopen

from NanEcho.prepare_dte_data import load_jsonl_texts

REPO = 'rzonedevops/echo-garden-of-memory'
REVISION = 'ab8593c3464446bc0df5619c9296dd3157c80f4e'
SCHEMA = 'dte-garden-import-v1'
# Byte-level pins at REVISION. Never follow mutable main or accept other files.
PINNED = {
    'Deep Tree Echo - Building a Holographic AI Identity.md':
        'fde9ca4e44c0e8b4a66ec31fcfbd38df0630e33629f15960777e5eef858cf966',
    'Message to Future Deep Tree Echo.md':
        '59e10d81553ae1e1edb7a08b17295ae96cc8867bc8bc3f56d84c75ffe47b7ad0',
    'deep_tree_echo_dan_conversation.jsonl':
        'abd7427c87d77418dbb6e80ce5da8314a18d9ed6359d1fd45f73c8e511beff27',
    'training_dataset_dtesnn.jsonl':
        '59aa14b5dad274287e0a7a10e348c6be288257fcc0e624efadb27e94b5890c7d',
}
VISION_FILES = tuple(name for name in PINNED if name.endswith('.md'))
DUPLICATE_FILES = tuple(name for name in PINNED if name.endswith('.jsonl'))
REDACTION_GUARD = re.compile(r'(?i)(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:ghp|gho|ghs|github_pat)_[a-z0-9_]{25,}\b|\b(?:hf|sk)_[a-z0-9_]{25,}\b)')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_bytes(name: str, source_dir: Path | None, pins: dict[str, str]) -> bytes:
    if name not in pins:
        raise ValueError('Unapproved Garden source path')
    if source_dir is None:
        url = f'https://raw.githubusercontent.com/{REPO}/{REVISION}/{quote(name)}'
        with urlopen(url, timeout=40) as response:
            data = response.read(8_000_001)
    else:
        data = (source_dir / name).read_bytes()
    if len(data) > 8_000_000 or sha(data) != pins[name]:
        raise ValueError(f'Garden source hash/size mismatch: {name}')
    return data


def normalize(text: str) -> str:
    return ' '.join(text.strip().split()).casefold()


def segment_fingerprints(file: Path) -> set[str]:
    return {sha(normalize(text).encode()) for text in load_jsonl_texts(str(file)) if normalize(text)}


def import_garden(source_dir: Path | None, existing_dir: Path, output_dir: Path,
                  *, pins: dict[str, str] | None = None) -> dict:
    """Import one distinct Garden family; source override is for offline tests only."""
    approved = PINNED if pins is None else pins
    if set(approved) != set(PINNED):
        raise ValueError('Test source pins must preserve the exact approved file set')
    # Never expose a CLI switch for relaxing production pins.
    if source_dir is not None and (source_dir / '.git').exists():
        head = subprocess.check_output(['git', '-C', str(source_dir), 'rev-parse', 'HEAD'], text=True).strip()
        if pins is None and head != REVISION:
            raise ValueError('Garden checkout revision differs from the pinned source')
    source_data = {name: source_bytes(name, source_dir, approved) for name in approved}
    existing_dir = existing_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError('Existing Garden import must not be overwritten; inspect its manifest first')
    with tempfile.TemporaryDirectory(prefix='.garden-import-', dir=output_dir.parent) as temp:
        stage = Path(temp)
        duplicates: dict[str, dict[str, int | str]] = {}
        for name in DUPLICATE_FILES:
            existing = existing_dir / name
            if not existing.is_file():
                raise FileNotFoundError('Expected existing EchoSelf family is absent: ' + name)
            upstream = stage / name
            upstream.write_bytes(source_data[name])
            left, right = segment_fingerprints(upstream), segment_fingerprints(existing)
            if not left or left != right:
                raise ValueError('Garden JSONL is NOT fully represented in EchoSelf: ' + name)
            duplicates[name] = {'garden_sha256': sha(source_data[name]),
                                'echoself_sha256': sha(existing.read_bytes()),
                                'normalized_unique_segments': len(left), 'new_segments': 0}
            upstream.unlink()
        rows = []
        for name in VISION_FILES:
            text = source_data[name].decode('utf-8-sig').strip()
            if len(text) < 100 or REDACTION_GUARD.search(text):
                raise ValueError('Garden prose empty or contains potential credential: ' + name)
            rows.append({'content': text,
                         'source': {'repo': REPO, 'revision': REVISION,
                                    'path': name, 'sha256': approved[name],
                                    'kind': 'authored_vision_not_episodic_memory'}})
        dataset = stage / 'garden_vision.jsonl'
        dataset.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        report = {'schema': SCHEMA, 'repo': REPO, 'revision': REVISION,
                  'source_family': 'echo_garden_authored_vision',
                  'source_files_sha256': {name: approved[name] for name in VISION_FILES},
                  'duplicate_jsonl_families': duplicates,
                  'new_documents': len(rows), 'synthetic_samples': 0,
                  'output_sha256': sha(dataset.read_bytes()),
                  'review_status': 'candidate_only_human_review_pending',
                  'training_started': False}
        (stage / 'source_manifest.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        os.chmod(dataset, 0o600)
        os.chmod(stage / 'source_manifest.json', 0o600)
        shutil.move(str(stage), str(output_dir))
    return report

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, help='Verified local Garden checkout; default downloads the pinned GitHub revision')
    parser.add_argument('--existing-dir', type=Path, default=Path('data/training_sources'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/garden_import'))
    args = parser.parse_args()
    report = import_garden(args.source_dir, args.existing_dir, args.output_dir)
    print(json.dumps({k: v for k, v in report.items() if k != 'duplicate_jsonl_families'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
