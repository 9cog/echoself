#!/usr/bin/env python3
"""Compile DTE 8192 training/validation from disjoint source files.

A split passing technical integrity checks is still only an experiment candidate:
source permissions, semantic leakage and an independent test need separate review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from array import array
from pathlib import Path
import sys

from tokenizers import Tokenizer

from NanEcho.prepare_dte_data import load_jsonl_texts

SCHEMA = "nanecho-curated-split-v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compile_split(paths: list[Path], tokenizer: Tokenizer, target: Path) -> tuple[int, int]:
    tokens = array('H')
    seen: set[str] = set()
    count = 0
    with target.open('wb') as output:
        for path in paths:
            # Each file is an indivisible source group, even if its JSONL rows
            # contain multiple turns of the same conversation.
            texts = load_jsonl_texts(str(path))
            if not texts:
                raise ValueError(f'No parsable source records: {path}')
            for raw in texts:
                text = ' '.join(raw.strip().split())
                if not text:
                    continue
                normalized = hashlib.sha256(text.casefold().encode()).hexdigest()
                if normalized in seen:
                    continue
                seen.add(normalized)
                ids = tokenizer.encode(f'<|startoftext|> {text} <|endoftext|>', add_special_tokens=False).ids
                if not ids or max(ids) >= 8192:
                    raise ValueError(f'Tokenizer produced an out-of-range ID for {path}')
                tokens.extend(ids)
                count += 1
            if len(tokens) > 32_000_000:
                raise ValueError('Split exceeds the current bounded audit size; stream the audit first')
        if len(tokens) < 1025:
            raise ValueError(f'Split too short for a 1024-token block: {target}')
        if sys.byteorder != 'little':
            tokens.byteswap()
        tokens.tofile(output)
    return len(tokens), count


def compile_dataset(train_files: list[Path], val_files: list[Path], tokenizer_file: Path, output: Path, reviewed_by: str) -> dict:
    train_files = [p.resolve(strict=True) for p in train_files]
    val_files = [p.resolve(strict=True) for p in val_files]
    root = Path.cwd().resolve()
    names = {p: str(p.relative_to(root)) for p in set(train_files + val_files)}
    if not train_files or not val_files or set(train_files) & set(val_files):
        raise ValueError('Both splits need disjoint nonempty sets of source files')
    if not reviewed_by.strip():
        raise ValueError('Document the source-group reviewer or automated audit owner')
    tok_json = json.loads(tokenizer_file.read_text(encoding='utf-8'))
    tokenizer_semantic_sha = hashlib.sha256(json.dumps(tok_json, sort_keys=True).encode()).hexdigest()
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    if tokenizer.get_vocab_size() != 8192:
        raise ValueError('Expected exact DTE 8192-token vocabulary')
    for label, expected in (('<|startoftext|>', 2), ('<|endoftext|>', 1), ('<|pad|>', 0)):
        if tokenizer.token_to_id(label) != expected:
            raise ValueError(f'Unexpected DTE {label} token ID')
    output.mkdir(parents=True, exist_ok=True)
    lengths = {}
    records = {}
    for name, files in (('train', train_files), ('val', val_files)):
        lengths[name], records[name] = compile_split(files, tokenizer, output / f'{name}.bin')
    tok_spec = {'name': 'dte_bpe', 'vocab_size': 8192, 'eos_token': '<|endoftext|>',
                'eos_token_id': 1, 'tokenizer_sha256': tokenizer_semantic_sha}
    sources = {names[p]: digest(p) for p in sorted(set(train_files + val_files))}
    source_revision = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    metadata = {
        'tokenizer_type': 'dte_bpe', 'tokenizer': tok_spec, 'tokenizer_semantic_sha256': tokenizer_semantic_sha,
        'vocab_size': 8192, 'block_size': 1024, 'dtype': 'uint16',
        'train_tokens': lengths['train'], 'val_tokens': lengths['val'],
        'train_documents': records['train'], 'val_documents': records['val'],
        'source_revision': source_revision, 'synthetic_samples': 0,
    }
    manifest = {
        'schema': SCHEMA, 'reviewed_by': reviewed_by, 'source_revision': source_revision,
        'source_files_sha256': sources, 'tokenizer': tok_spec,
        'tokenizer_semantic_sha256': tokenizer_semantic_sha, 'synthetic_samples': 0,
    }
    for name, files in (('train', train_files), ('val', val_files)):
        manifest[name] = {
            'sha256': digest(output / f'{name}.bin'), 'token_count': lengths[name],
            'source_groups': [names[p] for p in sorted(files)],
        }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'train_tokens': lengths['train'], 'val_tokens': lengths['val'],
            'train_documents': records['train'], 'val_documents': records['val'],
            'tokenizer_semantic_sha256': tokenizer_semantic_sha, 'source_revision': source_revision}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-files', nargs='+', type=Path, required=True)
    parser.add_argument('--val-files', nargs='+', type=Path, required=True)
    parser.add_argument('--tokenizer-file', type=Path, default=Path('NanEcho/dte_tokenizer/tokenizer.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/nanecho_dte_curated'))
    parser.add_argument('--reviewed-by', required=True)
    args = parser.parse_args()
    result = compile_dataset(args.train_files, args.val_files, args.tokenizer_file, args.output_dir, args.reviewed_by)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
