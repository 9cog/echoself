"""Offline regression tests for the training provenance gate."""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_nanecho_data import audit, SCHEMA


class DataAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.tokenizer = {"name": "gpt2", "vocab_size": 50257, "eos_token": "<|endoftext|>", "eos_token_id": 50256}
        self.train = list(range(101, 5101))
        self.val = list(range(6101, 8201))
        self._write()

    def _write(self, *, metadata_changes=None, manifest_changes=None):
        meta = {"train_tokens": len(self.train), "val_tokens": len(self.val), "vocab_size": 50257, "tokenizer": self.tokenizer, "synthetic_samples": 0}
        manifest = {"schema": SCHEMA, "reviewed_by": "corpus-steward", "source_revision": "immutable-source-digest", "tokenizer": self.tokenizer, "synthetic_samples": 0}
        for split in ("train", "val"):
            tokens = getattr(self, split)
            raw = struct.pack("<" + "H" * len(tokens), *tokens)
            (self.path / f"{split}.bin").write_bytes(raw)
            manifest[split] = {"sha256": hashlib.sha256(raw).hexdigest(), "token_count": len(tokens), "source_groups": [f"source-{split}-1"]}
        meta.update(metadata_changes or {})
        manifest.update(manifest_changes or {})
        (self.path / "metadata.json").write_text(json.dumps(meta))
        (self.path / "source_manifest.json").write_text(json.dumps(manifest))

    def test_disjoint_split_passes(self):
        self.assertTrue(audit(self.path)["passed"])

    def test_repeated_run_like_split_fails_even_if_source_claims_disjoint(self):
        pattern = list(range(100, 200))
        self.train = pattern * 50
        self.val = pattern * 21
        self._write()
        report = audit(self.path)
        self.assertFalse(report["passed"])
        self.assertEqual(report["overlapping_64_token_windows_fraction"], 1.0)

    def test_same_source_group_fails(self):
        self._write(manifest_changes={"val": {"sha256": hashlib.sha256((self.path / "val.bin").read_bytes()).hexdigest(), "token_count": len(self.val), "source_groups": ["source-train-1"]}})
        self.assertIn("source groups overlap", " ".join(audit(self.path)["errors"]))

    def test_synthetic_fallback_fails(self):
        self._write(metadata_changes={"created_by": "workflow_fallback"})
        self.assertIn("synthetic", " ".join(audit(self.path)["errors"]))

    def test_missing_manifest_fails(self):
        (self.path / "source_manifest.json").unlink()
        self.assertFalse(audit(self.path)["passed"])

    def test_modified_file_fails_digest(self):
        self.val[-1] = 9000
        (self.path / "val.bin").write_bytes(struct.pack("<" + "H" * len(self.val), *self.val))
        self.assertIn("SHA-256 mismatch", " ".join(audit(self.path)["errors"]))

    def test_undeclared_tokenizer_fails(self):
        self._write(metadata_changes={"tokenizer": {"name": "char", "vocab_size": 256, "eos_token_id": 0}})
        self.assertFalse(audit(self.path)["passed"])


if __name__ == "__main__":
    unittest.main()
