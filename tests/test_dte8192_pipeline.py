"""Offline DTE 8192 candidate-provenance and Hugging Face bridge tests."""
from __future__ import annotations

from array import array
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from NanEcho.hf_checkpoint_bridge import MODEL, _safe_value, pull, validate
from NanEcho.prepare_dte_grouped import compile_dataset, digest
from NanEcho.spec import TokenizerSpec, tokenizer_from_spec
from scripts.audit_nanecho_data import audit

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / 'NanEcho/dte_tokenizer/tokenizer.json'


class Shape:
    shape = (8192, 256)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.semantic = hashlib.sha256(json.dumps(json.loads(TOKENIZER.read_text()), sort_keys=True).encode()).hexdigest()
        cls.spec = TokenizerSpec('dte_bpe', 8192, '<|endoftext|>', 1, cls.semantic)

    def test_original_commit_tokenizer_semantic_digest(self):
        self.assertEqual(self.semantic, '5ca83b21d48d0f9d43a1b983a586a8c37f6779b4630a1714e530f79df6bba325')
        adapter = tokenizer_from_spec(self.spec)
        sample = 'Deep Tree Echo candidate verification.'
        self.assertEqual(adapter.decode(adapter.encode(sample)), sample)
        self.assertEqual(adapter.provenance()['tokenizer_sha256'], self.semantic)

    def test_same_size_wrong_tokenizer_refused(self):
        wrong = TokenizerSpec('dte_bpe', 8192, '<|endoftext|>', 1, '0' * 64)
        with self.assertRaisesRegex(ValueError, 'digest'):
            tokenizer_from_spec(wrong)
        with self.assertRaisesRegex(ValueError, 'digest'):
            tokenizer_from_spec(TokenizerSpec('dte_bpe', 8192, '<|endoftext|>', 1))

    def test_unknown_tokenizer_never_maps_to_gpt2(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported tokenizer'):
            tokenizer_from_spec(TokenizerSpec('unapproved', 8192, '<|endoftext|>', 1))

    def test_hub_payload_strips_numpy_scalars_in_nested_metrics(self):
        fake_torch = types.SimpleNamespace(Tensor=type('FakeTensor', (), {}))
        with patch.dict(sys.modules, {'torch': fake_torch}):
            cleaned = _safe_value({'metrics': {'validation': np.float64(6.2),
                                               'tokens': np.int64(10)}})
        self.assertIs(type(cleaned['metrics']['validation']), float)
        self.assertIs(type(cleaned['metrics']['tokens']), int)

    def test_private_untrained_candidate_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest_file = Path(folder) / 'hub_manifest.json'
            manifest_file.write_text(json.dumps({'iteration': 0, 'status': 'candidate_not_promoted'}))
            info = types.SimpleNamespace(private=True, sha='test-revision')
            with patch('huggingface_hub.HfApi') as api, patch(
                    'huggingface_hub.hf_hub_download', return_value=str(manifest_file)) as download:
                api.return_value.model_info.return_value = info
                result = pull('drzo/echoself-dte', Path(folder) / 'cache', {},
                              Path(folder), 'fake', allow_missing=True)
            self.assertEqual(result['status'], 'untrained_baseline_ignored')
            self.assertEqual(download.call_count, 1)

    def test_compiled_grouped_split_and_bridge_checks(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            train = root / 'train.jsonl'
            val = root / 'val.jsonl'
            train.write_text(json.dumps({'messages': [{'role': 'user', 'content': 'Polytope neuron circuit.' * 900}]}) + '\n')
            val.write_text(json.dumps({'messages': [{'role': 'assistant', 'content': 'Separate identity lineage proof.' * 900}]}) + '\n')
            out = root / 'compiled'
            with patch('pathlib.Path.cwd', return_value=ROOT):
                result = compile_dataset([train], [val], TOKENIZER, out, 'automated-test-human-review-pending')
            self.assertGreater(result['train_tokens'], 1024)
            manifest = json.loads((out / 'source_manifest.json').read_text())
            self.assertEqual(len(set(manifest['train']['source_groups']) & set(manifest['val']['source_groups'])), 0)
            self.assertEqual(manifest['tokenizer']['tokenizer_sha256'], self.semantic)
            report = audit(out, 1024, 8192)
            self.assertTrue(report['passed'], report['errors'])
            config = dict(MODEL)
            checkpoint = {
                'model_state_dict': {'token_embedding.weight': Shape()},
                'model_config': config, 'tokenizer': manifest['tokenizer'],
                'optimizer_state_dict': {}, 'checkpoint_id': 'ckpt_test', 'iteration': 1,
                'data_config': {name + '_sha256': digest(out / name)
                                for name in ('train.bin', 'val.bin', 'metadata.json', 'source_manifest.json')},
            }
            self.assertEqual(validate(checkpoint, manifest, out)['iteration'], 1)
            untrained = dict(checkpoint, iteration=0)
            with self.assertRaisesRegex(ValueError, 'iteration-zero baseline'):
                validate(untrained, manifest, out)
            tampered = dict(checkpoint, tokenizer=dict(checkpoint['tokenizer'], tokenizer_sha256='f' * 64))
            with self.assertRaisesRegex(ValueError, 'tokenizer'):
                validate(tampered, manifest, out)
            tampered = dict(checkpoint, model_config=dict(config, vocab_size=50257))
            with self.assertRaisesRegex(ValueError, 'architecture'):
                validate(tampered, manifest, out)
            (out / 'train.bin').write_bytes((out / 'train.bin').read_bytes() + b'\x00\x00')
            with self.assertRaisesRegex(ValueError, 'dataset hash'):
                validate(checkpoint, manifest, out)

    def test_identical_file_cannot_be_train_and_val(self):
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            compile_dataset([TOKENIZER], [TOKENIZER], TOKENIZER, ROOT / 'unused-dte-test', 'test')


if __name__ == '__main__':
    unittest.main()
