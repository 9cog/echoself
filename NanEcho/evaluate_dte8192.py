#!/usr/bin/env python3
"""Compare native DTE candidate checkpoints on identical fixed held-out windows.

This reports language-model held-out loss only. It does not measure identity
quality, source rights, privacy leakage or warrant model promotion.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from NanEcho.hf_checkpoint_bridge import MODEL, load_native, sha256
from nanecho_model import NanEchoConfig, NanEchoModel


def fixed_windows(length: int, block_size: int = 1024, count: int = 8) -> list[int]:
    maximum = length - block_size - 1
    if maximum < 0:
        raise ValueError('Held-out validation stream is too short')
    if count < 2 or maximum < count - 1:
        raise ValueError('Held-out sample count does not fit validation stream')
    return [(i * maximum) // (count - 1) for i in range(count)]


def score(path: Path, validation: np.memmap, positions: list[int]) -> dict:
    checkpoint = load_native(path, trusted_local=True)
    config = checkpoint.get('model_config') or checkpoint.get('config')
    for name, expected in MODEL.items():
        if config.get(name) != expected:
            raise ValueError('Cannot compare different model architectures')
    args = {key: val for key, val in config.items() if key in NanEchoConfig.__dataclass_fields__}
    args['dropout'] = 0.0
    model = NanEchoModel(NanEchoConfig(**args))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.current_iteration = int(checkpoint.get('current_iteration', checkpoint['iteration']))
    model.eval()
    losses = []
    with torch.no_grad():
        for start in positions:
            x = torch.from_numpy(validation[start:start + 1024].astype(np.int64)).unsqueeze(0)
            y = torch.from_numpy(validation[start + 1:start + 1025].astype(np.int64)).unsqueeze(0)
            value = float(model(x, labels=y)['loss'].item())
            if not math.isfinite(value):
                raise ValueError('Non-finite held-out loss')
            losses.append(value)
    return {'checkpoint_id': checkpoint.get('checkpoint_id'), 'iteration': checkpoint['iteration'],
            'mean_nll': sum(losses) / len(losses), 'per_window_nll': losses}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.cache_dir / 'metadata.json').read_text())
    entries = [(m['iteration'], args.cache_dir / 'checkpoints' / f'{key}.pt')
               for key, m in metadata.items() if (args.cache_dir / 'checkpoints' / f'{key}.pt').is_file()]
    if not entries or max(it for it, _ in entries) < 1:
        raise ValueError('No trained checkpoints available for held-out comparison')
    validation_file = args.data_dir / 'val.bin'
    validation = np.memmap(validation_file, mode='r', dtype=np.uint16)
    positions = fixed_windows(len(validation))
    chosen = [min(entries, key=lambda x: x[0]), max(entries, key=lambda x: x[0])]
    if chosen[0][0] == chosen[1][0]:
        chosen = [chosen[1]]
    scored = [score(path, validation, positions) for _, path in chosen]
    report = {'status': 'candidate_not_promoted', 'validation_sha256': sha256(validation_file),
              'fixed_window_starts': positions, 'window_tokens': 1024,
              'checkpoints': scored, 'identity_improvement_proven': False}
    if len(scored) == 2:
        report['nll_delta_last_minus_first'] = scored[1]['mean_nll'] - scored[0]['mean_nll']
        report['lower_fixed_nll'] = report['nll_delta_last_minus_first'] < 0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'checkpoints'}))
    print('checkpoints', [{'iteration': c['iteration'], 'mean_nll': c['mean_nll']} for c in scored])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
