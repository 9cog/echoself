#!/usr/bin/env python3
"""Private Hugging Face backup of native DTE 8192 NanEcho candidate checkpoints.

No GPT-2 remapping, public release, or identity-promotion decision occurs here.
Only checkpoints with the same full model/tokenizer/dataset lineage can resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

MODEL = {'vocab_size': 8192, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'block_size': 1024}
PATH = 'candidate/native_checkpoint.pt'
MANIFEST_PATH = 'candidate/checkpoint_manifest.json'
METADATA_PATH = 'candidate/cache_metadata.json'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate(checkpoint: dict, source_manifest: dict, data_dir: Path) -> dict:
    """Check architecture, tokenizer semantic hash, source and token-byte lineage."""
    if not isinstance(checkpoint, dict):
        raise ValueError('Native checkpoint must be a dictionary')
    state = checkpoint.get('model_state_dict')
    config = checkpoint.get('model_config') or checkpoint.get('config')
    if not isinstance(state, dict) or not state or not isinstance(config, dict):
        raise ValueError('Missing native NanEcho model_state_dict or model_config')
    for key, expected in MODEL.items():
        if config.get(key) != expected:
            raise ValueError(f'Incompatible native DTE architecture: {key}')
    tok = checkpoint.get('tokenizer')
    wanted = source_manifest.get('tokenizer')
    if not isinstance(tok, dict) or not isinstance(wanted, dict) or tok != wanted or tok.get('name') != 'dte_bpe':
        raise ValueError('Checkpoint DTE tokenizer and source manifest disagree')
    data = checkpoint.get('data_config')
    if not isinstance(data, dict):
        raise ValueError('Checkpoint has no content-addressed dataset lineage')
    for name in ('train.bin', 'val.bin', 'metadata.json', 'source_manifest.json'):
        local = data_dir / name
        if not local.is_file() or data.get(name + '_sha256') != sha256(local):
            raise ValueError(f'DTE checkpoint dataset hash mismatch: {name}')
    if (data['train.bin_sha256'] != source_manifest.get('train', {}).get('sha256')
            or data['val.bin_sha256'] != source_manifest.get('val', {}).get('sha256')):
        raise ValueError('Checkpoint tokens differ from source manifest')
    embeddings = state.get('token_embedding.weight')
    if embeddings is None or tuple(embeddings.shape) != (8192, 256):
        raise ValueError('Native DTE token embedding must be exactly [8192,256]')
    if not isinstance(checkpoint.get('optimizer_state_dict'), dict):
        raise ValueError('Native candidate requires optimizer state for exact continuation')
    checkpoint_id = checkpoint.get('checkpoint_id')
    if not isinstance(checkpoint_id, str) or not checkpoint_id.startswith('ckpt_'):
        raise ValueError('Native candidate lacks its cache checkpoint ID')
    return {'format': 'nanecho-dte-native-candidate-v1', 'checkpoint_id': checkpoint_id,
            'iteration': int(checkpoint.get('iteration', 0)),
            'source_revision': source_manifest.get('source_revision'),
            'tokenizer_sha256': tok['tokenizer_sha256']}


def load_native(path: Path, *, trusted_local: bool = False) -> dict:
    import torch
    # Never execute untrusted Hub pickle. Local training output may contain
    # NumPy scalars; sanitize it before uploading in a weights-only container.
    return torch.load(path, map_location='cpu', weights_only=not trusted_local)


def _safe_value(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor) or value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k) if not isinstance(k, (int, str)) else k: _safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    raise ValueError(f'Cannot safely serialize checkpoint field of type {type(value).__name__}')


def select(cache_dir: Path, source_manifest: dict, data_dir: Path) -> tuple[Path, dict, dict]:
    metadata_file = cache_dir / 'metadata.json'
    entries = json.loads(metadata_file.read_text())
    candidates = []
    for checkpoint_id, meta in entries.items():
        path = cache_dir / 'checkpoints' / (checkpoint_id + '.pt')
        if not path.is_file():
            continue
        if any(meta.get('model_config', {}).get(k) != v for k, v in MODEL.items()):
            continue
        try:
            data = load_native(path, trusted_local=True)
            evidence = validate(data, source_manifest, data_dir)
        except (ValueError, RuntimeError, KeyError):
            continue
        candidates.append((float(meta['val_loss']), path, meta, evidence))
    if not candidates:
        raise ValueError('No matching native DTE 8192 checkpoint with optimizer state found')
    _, path, meta, evidence = min(candidates, key=lambda candidate: candidate[0])
    return path, meta, evidence


def pull(repo_id: str, cache_dir: Path, manifest: dict, data_dir: Path, token: str, allow_missing: bool) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    api = HfApi(token=token)
    try:
        info = api.model_info(repo_id)
    except RepositoryNotFoundError:
        if allow_missing:
            return {'status': 'no_prior_candidate', 'repo_id': repo_id}
        raise
    if not info.private:
        raise ValueError('Refusing to resume from public/unreviewed Hugging Face model')
    try:
        files = [Path(hf_hub_download(repo_id, item, token=token, revision=info.sha))
                 for item in (MANIFEST_PATH, METADATA_PATH, PATH)]
    except EntryNotFoundError:
        if allow_missing:
            return {'status': 'no_prior_candidate', 'repo_id': repo_id}
        raise
    remote = json.loads(files[0].read_text())
    meta = json.loads(files[1].read_text())
    if remote.get('sha256') != sha256(files[2]):
        raise ValueError('Hub native checkpoint content hash mismatch')
    evidence = validate(load_native(files[2]), manifest, data_dir)
    checkpoint_id = evidence['checkpoint_id']
    if meta.get('checkpoint_id') != checkpoint_id or meta.get('data_config') != load_native(files[2])['data_config']:
        raise ValueError('Hub cache metadata and checkpoint lineage disagree')
    (cache_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    dest = cache_dir / 'checkpoints' / (checkpoint_id + '.pt')
    shutil.copy2(files[2], dest)
    metadata_file = cache_dir / 'metadata.json'
    old = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
    old[checkpoint_id] = meta
    metadata_file.write_text(json.dumps(old, indent=2) + '\n')
    return {'status': 'resumed_native_candidate', 'repo_id': repo_id, 'hub_sha': info.sha,
            'sha256': remote['sha256'], **evidence}


def push(repo_id: str, cache_dir: Path, manifest: dict, data_dir: Path, token: str, run_id: str) -> dict:
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    import torch
    path, meta, evidence = select(cache_dir, manifest, data_dir)
    api = HfApi(token=token)
    try:
        info = api.model_info(repo_id)
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type='model', private=True, exist_ok=False)
        info = api.model_info(repo_id)
    if not info.private:
        raise ValueError('Refusing to publish unreviewed candidate to public model repository')
    try:
        previous_file = hf_hub_download(repo_id, MANIFEST_PATH, token=token, revision=info.sha)
    except EntryNotFoundError:
        previous_file = None
    if previous_file:
        previous = json.loads(Path(previous_file).read_text())
        if (previous.get('source_revision') != evidence['source_revision']
                or previous.get('tokenizer_sha256') != evidence['tokenizer_sha256']):
            raise ValueError('Existing Hub candidate belongs to a different corpus/tokenizer lineage')
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / 'native_checkpoint.pt'
        # Local trusted training output may contain NumPy scalar metrics.
        torch.save(_safe_value(load_native(path, trusted_local=True)), candidate)
        # Ensure the exact Hub-bound object can be safely loaded before upload.
        validate(load_native(candidate), manifest, data_dir)
        record = {'status': 'candidate_not_promoted', 'sha256': sha256(candidate),
                  'source_repo': '9cog/echoself', 'github_run_id': run_id, **evidence}
        record_file = Path(directory) / 'checkpoint_manifest.json'
        record_file.write_text(json.dumps(record, indent=2) + '\n')
        meta_file = Path(directory) / 'cache_metadata.json'
        meta_file.write_text(json.dumps(_safe_value(meta), indent=2) + '\n')
        result = api.create_commit(
            repo_id=repo_id, repo_type='model',
            operations=[CommitOperationAdd(path_in_repo=PATH, path_or_fileobj=str(candidate)),
                        CommitOperationAdd(path_in_repo=MANIFEST_PATH, path_or_fileobj=str(record_file)),
                        CommitOperationAdd(path_in_repo=METADATA_PATH, path_or_fileobj=str(meta_file))],
            commit_message=f'Backup unpromoted DTE 8192 candidate, GitHub run {run_id}',
        )
    return {'status': 'private_native_candidate_backed_up', 'repo_id': repo_id,
            'commit_url': result.commit_url, **evidence}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('pull', 'push', 'verify'))
    parser.add_argument('--repo-id', default='drzo/echoself-dte')
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--allow-missing', action='store_true', help='Allow first run only if candidate repo/file is absent')
    parser.add_argument('--run-id', default='local')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.source_manifest.read_text())
    if manifest.get('schema') != 'nanecho-curated-split-v1':
        raise ValueError('Unexpected DTE source-group manifest schema')
    token = os.getenv('HF_TOKEN')
    if args.operation != 'verify' and not token:
        raise ValueError('HF_TOKEN required for private checkpoint exchange')
    if args.operation == 'pull':
        result = pull(args.repo_id, args.cache_dir, manifest, args.data_dir, token, args.allow_missing)
    else:
        result = (push(args.repo_id, args.cache_dir, manifest, args.data_dir, token, args.run_id)
                  if args.operation == 'push' else
                  {'status': 'native_candidate_verified', **select(args.cache_dir, manifest, args.data_dir)[2]})
    rendered = json.dumps(result, indent=2) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    print(rendered, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
