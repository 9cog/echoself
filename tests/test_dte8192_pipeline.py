"""Offline DTE 8192 candidate-provenance and Hugging Face bridge tests."""
from __future__ import annotations

from array import array
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from NanEcho.hf_checkpoint_bridge import MODEL, validate
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

    def test_jsonl_loader_stays_importable_without_numpy(self):
        """The provenance guard installs only tokenizers; numpy must stay optional."""
        tree = ast.parse((ROOT / 'NanEcho/prepare_dte_data.py').read_text(encoding='utf-8'))
        top_level = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.extend(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.append(node.module.split('.')[0])
        self.assertNotIn('numpy', top_level)
        blocked = subprocess.run(
            [sys.executable, '-c', (
                'import sys\n'
                'class BlockNumpy:\n'
                '    def find_spec(self, fullname, path=None, target=None):\n'
                "        if fullname == 'numpy' or fullname.startswith('numpy.'):\n"
                '            raise ModuleNotFoundError(fullname)\n'
                '        return None\n'
                'sys.meta_path.insert(0, BlockNumpy())\n'
                'from NanEcho.prepare_dte_grouped import compile_dataset, digest\n'
                'from NanEcho.prepare_dte_data import load_jsonl_texts\n'
                'print("imports_ok_without_numpy")\n'
            )],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(blocked.returncode, 0, blocked.stderr)
        self.assertIn('imports_ok_without_numpy', blocked.stdout)


if __name__ == '__main__':
    unittest.main()
