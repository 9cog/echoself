"""Fail-closed tests for the native DTE inference and causal-LM objective."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from NanEcho.evaluate_dte8192 import fixed_windows, score
from scripts.evaluate_dte_candidate import (aggregate_diagnostics, canonical_hash, diagnostics, load_probes,
                                             private_write, teacher_forced_nll)
from train_nanecho import DataLoader, TrainingConfig


class IdentityCandidateTests(unittest.TestCase):
    def test_next_token_loader_labels_are_unshifted(self):
        cfg = TrainingConfig(data_dir='ignored', block_size=4, batch_size=2, device='cpu')
        loader = DataLoader(cfg)
        loader.train_data = np.arange(20, dtype=np.uint16)
        x, y = loader.get_batch('train')
        self.assertTrue(torch.equal(x, y))
        self.assertEqual(tuple(x.shape), (2, 4))

    def test_fixed_windows_and_scorer_use_same_x_for_labels(self):
        class SpyModel:
            def __init__(self, config):
                self.config = config
                self.observed = []
            def load_state_dict(self, *_args, **_kwargs):
                pass
            def eval(self):
                pass
            def __call__(self, x, labels):
                self.observed.append((x.clone(), labels.clone()))
                return {'loss': torch.tensor(2.0)}
        created = []
        def model_factory(config):
            m = SpyModel(config)
            created.append(m)
            return m
        from NanEcho.hf_checkpoint_bridge import MODEL
        ckpt = {'model_config': dict(MODEL), 'model_state_dict': {}, 'iteration': 24,
                'checkpoint_id': 'ckpt_test'}
        val = np.arange(3000, dtype=np.uint16)
        with patch('NanEcho.evaluate_dte8192.load_native', return_value=ckpt), patch(
                'NanEcho.evaluate_dte8192.NanEchoModel', side_effect=model_factory):
            result = score(Path('unused.pt'), val, [0, 100])
        self.assertEqual(result['per_window_nll'], [2.0, 2.0])
        self.assertEqual(len(created[0].observed), 2)
        for x, y in created[0].observed:
            self.assertTrue(torch.equal(x, y))
            self.assertEqual(x.numel(), 1024)
        self.assertEqual(len(fixed_windows(25066)), 8)

    def test_conditional_nll_scores_only_reference_tokens(self):
        class Tokenizer:
            def encode(self, value):
                return {'A': [1, 2], 'B': [3, 4], 'AB': [1, 2, 3, 4]}[value]
        class Model:
            def __call__(self, x):
                # Uniform logits across 8 tokens: each predicted position has log(8).
                return {'logits': torch.zeros(1, x.size(1), 8)}
        rt = SimpleNamespace(encode=Tokenizer().encode, config=SimpleNamespace(block_size=4),
                             device='cpu', model=Model())
        nll, count = teacher_forced_nll(rt, 'A', 'B', stride=2)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(nll, np.log(8), places=5)
        nll2, count2 = teacher_forced_nll(rt, 'AB', stride=2)
        self.assertEqual(count2, 3)
        self.assertAlmostEqual(nll2, np.log(8), places=5)

    def test_probe_validation_and_output_privacy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probes.jsonl'
            path.write_text(json.dumps({'id':'a','family':'test','prompt':'A','reference':'B'})+'\n')
            probes = load_probes(path)
            self.assertEqual(probes[0]['reference'], 'B')
            private = Path(directory) / 'result.json'
            private_write(private, {'prompt_hash': canonical_hash(probes)})
            self.assertEqual(private.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                private_write(private, {})
            path.write_text(json.dumps(probes[0])+'\n'+json.dumps(probes[0])+'\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_probes(path)
            self.assertTrue(diagnostics('')['empty'])

    def test_identical_zero_word_outputs_flag_collapse(self):
        samples=[{'generation_sha256':'same','diagnostics': {'words':0}},
                 {'generation_sha256':'same','diagnostics': {'words':0}}]
        summary=aggregate_diagnostics(samples)
        self.assertTrue(summary['collapse_warning'])
        self.assertEqual(summary['zero_word_generations'],2)
        self.assertIsNone(summary['human_identity_verdict'])


if __name__ == '__main__':
    unittest.main()
