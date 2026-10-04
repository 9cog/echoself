"""Garden-of-Memory ingestion tests; no network, user conversations or model training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from NanEcho.import_echo_garden import DUPLICATE_FILES, PINNED, VISION_FILES, import_garden


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class GardenImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.garden = self.root / 'garden'
        self.existing = self.root / 'existing'
        self.garden.mkdir()
        self.existing.mkdir()
        self.out = self.root / 'staged'
        self.pins = {}
        for idx, name in enumerate(VISION_FILES):
            data = (f'Authored vision {idx}: truth, curiosity and memory. ' * 50).encode()
            (self.garden / name).write_bytes(data)
            self.pins[name] = sha(data)
        for name in DUPLICATE_FILES:
            garden_bytes = (json.dumps({'messages': [{'content': 'Original authored context.' * 25}]}) + '\n').encode()
            existing_bytes = (json.dumps({'messages': [{'content': ' Original  authored context. ' * 25}]}) + '\n').encode()
            # These parse to distinct strings; instead vary only JSON serialization.
            existing_bytes = (json.dumps({'messages': [{'content': 'Original authored context.' * 25}]}, indent=1) + '\n').encode()
            (self.garden / name).write_bytes(garden_bytes)
            (self.existing / name).write_bytes(existing_bytes)
            self.pins[name] = sha(garden_bytes)

    def test_distinct_source_family_staged_and_private(self):
        report = import_garden(self.garden, self.existing, self.out, pins=self.pins)
        self.assertEqual(report['new_documents'], 2)
        self.assertFalse(report['training_started'])
        self.assertTrue(all(item['new_segments'] == 0 for item in report['duplicate_jsonl_families'].values()))
        self.assertEqual(sorted(self.out.iterdir()), sorted([self.out / 'garden_vision.jsonl', self.out / 'source_manifest.json']))
        rows = [json.loads(line) for line in (self.out / 'garden_vision.jsonl').read_text().splitlines()]
        self.assertEqual({row['source']['path'] for row in rows}, set(VISION_FILES))
        self.assertTrue(all(row['source']['kind'] == 'authored_vision_not_episodic_memory' for row in rows))
        self.assertEqual(report['output_sha256'], sha((self.out / 'garden_vision.jsonl').read_bytes()))
        self.assertEqual((self.out / 'garden_vision.jsonl').stat().st_mode & 0o777, 0o600)

    def test_changed_source_hash_rejected_without_output(self):
        (self.garden / VISION_FILES[0]).write_text('edited after source pin')
        with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
            import_garden(self.garden, self.existing, self.out, pins=self.pins)
        self.assertFalse(self.out.exists())

    def test_novel_garden_conversation_rejected_without_output(self):
        (self.existing / DUPLICATE_FILES[0]).write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'NOT fully represented'):
            import_garden(self.garden, self.existing, self.out, pins=self.pins)
        self.assertFalse(self.out.exists())

    def test_missing_old_family_rejected(self):
        (self.existing / DUPLICATE_FILES[0]).unlink()
        with self.assertRaises(FileNotFoundError):
            import_garden(self.garden, self.existing, self.out, pins=self.pins)
        self.assertFalse(self.out.exists())

    def test_idempotence_requires_manifest_review(self):
        import_garden(self.garden, self.existing, self.out, pins=self.pins)
        with self.assertRaises(FileExistsError):
            import_garden(self.garden, self.existing, self.out, pins=self.pins)

    def test_refuse_new_unreviewed_path(self):
        pins = dict(self.pins, **{'private_memory_dump.jsonl': '0' * 64})
        with self.assertRaisesRegex(ValueError, 'exact approved file set'):
            import_garden(self.garden, self.existing, self.out, pins=pins)

    def test_secret_like_source_rejected(self):
        content = ('Safe. ' * 100 + 'ghp_' + 'A' * 36).encode()
        (self.garden / VISION_FILES[0]).write_bytes(content)
        self.pins[VISION_FILES[0]] = sha(content)
        with self.assertRaisesRegex(ValueError, 'potential credential'):
            import_garden(self.garden, self.existing, self.out, pins=self.pins)
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
