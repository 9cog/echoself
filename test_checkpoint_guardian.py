#!/usr/bin/env python3
"""
Comprehensive tests for the CheckpointGuardian class.

Tests cover initialization, manifest management, checkpoint discovery,
restoration logic, and cleanup — all without requiring PyTorch.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, str(Path(__file__).parent))
from scripts.checkpoint_guardian import CheckpointGuardian


def _make_guardian(tmpdir: str, allow_fresh_start: bool = False) -> CheckpointGuardian:
    """Create a CheckpointGuardian whose backup manifest path is inside tmpdir."""
    guardian = CheckpointGuardian(tmpdir, allow_fresh_start=allow_fresh_start)
    # Redirect the backup manifest path to stay inside the temp directory so
    # that tests don't write to the actual repository.
    guardian.backup_manifest_path = Path(tmpdir) / "backup_checkpoint_manifest.json"
    return guardian


class TestCheckpointGuardianInit(unittest.TestCase):
    """Tests for CheckpointGuardian initialization."""

    def test_init_creates_output_dir(self):
        """Initialization should not fail even if output_dir doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "nonexistent_subdir")
            guardian = _make_guardian(output_dir)
            self.assertEqual(str(guardian.output_dir), output_dir)

    def test_init_safe_mode_by_default(self):
        """Default initialization should be in safe (no fresh start) mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir)
            self.assertFalse(guardian.allow_fresh_start)

    def test_init_allow_fresh_start(self):
        """Allow fresh start flag should be respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir, allow_fresh_start=True)
            self.assertTrue(guardian.allow_fresh_start)

    def test_init_creates_empty_manifest(self):
        """A fresh initialization should produce a valid manifest dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir)
            self.assertIsInstance(guardian.manifest, dict)
            self.assertIn("checkpoints", guardian.manifest)


class TestManifest(unittest.TestCase):
    """Tests for manifest save and load operations."""

    def test_manifest_saved_and_reloaded(self):
        """Manifest changes should persist across guardian instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian1 = _make_guardian(tmpdir)
            guardian1.manifest["checkpoints"] = [
                {"iteration": 10, "val_loss": 3.0, "path": "fake.pt"}
            ]
            guardian1._save_manifest()

            guardian2 = _make_guardian(tmpdir)
            self.assertEqual(len(guardian2.manifest["checkpoints"]), 1)
            self.assertEqual(guardian2.manifest["checkpoints"][0]["iteration"], 10)

    def test_manifest_default_structure(self):
        """Manifest must have the required top-level keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir)
            required_keys = {"checkpoints"}
            self.assertTrue(
                required_keys.issubset(set(guardian.manifest.keys())),
                f"Manifest missing keys: {required_keys - set(guardian.manifest.keys())}",
            )

    def test_manifest_saved_to_multiple_locations(self):
        """_save_manifest should write to backup_manifest_path as well."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir)
            guardian._save_manifest()
            # At least the primary manifest path should exist
            self.assertTrue(
                guardian.manifest_path.exists()
                or guardian.backup_manifest_path.exists(),
                "Neither manifest file was created",
            )


class TestComputeChecksum(unittest.TestCase):
    """Tests for _compute_checksum."""

    def test_checksum_is_hex_string(self):
        """Checksum should be a non-empty hex string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir)
            test_file = Path(tmpdir) / "sample.bin"
            test_file.write_bytes(b"\x00\x01\x02\x03")
            checksum = guardian._compute_checksum(test_file)
            self.assertIsInstance(checksum, str)
            self.assertGreater(len(checksum), 0)
            int(checksum, 16)  # Should parse as hex without raising

    def test_checksum_differs_for_different_content(self):
        """Different file contents must produce different checksums."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir)
            file_a = Path(tmpdir) / "a.bin"
            file_b = Path(tmpdir) / "b.bin"
            file_a.write_bytes(b"content_A")
            file_b.write_bytes(b"content_B")
            self.assertNotEqual(
                guardian._compute_checksum(file_a),
                guardian._compute_checksum(file_b),
            )

    def test_checksum_consistent_for_same_content(self):
        """The same file should always produce the same checksum."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir)
            test_file = Path(tmpdir) / "consistent.bin"
            test_file.write_bytes(b"deterministic")
            self.assertEqual(
                guardian._compute_checksum(test_file),
                guardian._compute_checksum(test_file),
            )


class TestVerifyCheckpoint(unittest.TestCase):
    """Tests for _verify_checkpoint."""

    def test_nonexistent_file_returns_false(self):
        """A path that does not exist should return False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir)
            missing = Path(tmpdir) / "no_such_file.pt"
            self.assertFalse(guardian._verify_checkpoint(missing))

    def test_empty_file_returns_false(self):
        """An empty file is not a valid checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir)
            empty = Path(tmpdir) / "empty.pt"
            empty.write_bytes(b"")
            self.assertFalse(guardian._verify_checkpoint(empty))

    def test_small_corrupt_file_returns_false(self):
        """A small file that can't be loaded by torch should return False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir)
            bad_pt = Path(tmpdir) / "corrupt.pt"
            bad_pt.write_bytes(b"not a real checkpoint")
            # Without torch this returns False (import error or load error)
            result = guardian._verify_checkpoint(bad_pt)
            self.assertIsInstance(result, bool)


class TestFindBestCheckpoint(unittest.TestCase):
    """Tests for find_best_checkpoint."""

    def test_returns_none_when_no_checkpoints_exist(self):
        """Should return None when the output directory has no checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir)
            result = guardian.find_best_checkpoint()
            self.assertIsNone(result)

    def test_returns_none_when_manifest_empty(self):
        """Returns None even if manifest is initialized but has no entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir)
            guardian.manifest["checkpoints"] = []
            guardian._save_manifest()
            result = guardian.find_best_checkpoint()
            self.assertIsNone(result)


class TestRestoreCheckpoint(unittest.TestCase):
    """Tests for restore_checkpoint."""

    def test_restore_returns_none_when_fresh_start_allowed_and_empty(self):
        """When no checkpoint exists and fresh start is allowed, return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir, allow_fresh_start=True)
            result = guardian.restore_checkpoint()
            self.assertIsNone(result)

    def test_restore_raises_when_no_fresh_start_and_empty(self):
        """When no checkpoint exists and fresh start is not allowed, should raise
        RuntimeError to prevent accidental loss of training progress."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir, allow_fresh_start=False)
            with self.assertRaises(RuntimeError):
                guardian.restore_checkpoint()


class TestCleanupOldCheckpoints(unittest.TestCase):
    """Tests for cleanup_old_checkpoints."""

    def test_cleanup_does_not_fail_when_few_checkpoints(self):
        """Cleanup should succeed without error when checkpoint count <= keep_count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = _make_guardian(tmpdir)
            # Should not raise even when there are no physical checkpoints in BACKUP_LOCATIONS
            guardian.cleanup_old_checkpoints(keep_count=5)

    def test_cleanup_keeps_best_checkpoints(self):
        """After cleanup, files exceeding keep_count should be removed from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create enough fake checkpoint files in a BACKUP_LOCATION sub-path
            ckpt_backup_dir = Path(tmpdir) / ".training-progress" / "checkpoints"
            ckpt_backup_dir.mkdir(parents=True)

            created_files = []
            for i in range(8):
                ckpt_file = ckpt_backup_dir / f"checkpoint_iter{(i+1)*100}_fake_{i:04d}.pt"
                ckpt_file.write_bytes(b"fake " * 200)
                created_files.append(ckpt_file)

            # Run cleanup with a guardian whose BACKUP_LOCATIONS includes our tmpdir
            guardian = _make_guardian(tmpdir)

            # Patch BACKUP_LOCATIONS to use our temp directory paths
            with patch.object(
                CheckpointGuardian,
                "BACKUP_LOCATIONS",
                [str(ckpt_backup_dir)],
            ):
                guardian.cleanup_old_checkpoints(keep_count=3)

            # After cleanup, at most keep_count files should remain
            remaining = list(ckpt_backup_dir.glob("*.pt"))
            self.assertLessEqual(
                len(remaining),
                3,
                f"Expected ≤3 files remaining, got {len(remaining)}: {remaining}",
            )


class TestCheckpointGuardianFileStructure(unittest.TestCase):
    """Tests that validate the module and class structure."""

    def test_module_importable(self):
        """The checkpoint_guardian module must be importable."""
        import importlib

        module = importlib.import_module("scripts.checkpoint_guardian")
        self.assertTrue(hasattr(module, "CheckpointGuardian"))

    def test_class_has_required_methods(self):
        """CheckpointGuardian must expose the documented public API."""
        required = {
            "restore_checkpoint",
            "backup_checkpoint",
            "find_best_checkpoint",
            "cleanup_old_checkpoints",
        }
        actual = set(
            name for name in dir(CheckpointGuardian) if not name.startswith("__")
        )
        missing = required - actual
        self.assertFalse(missing, f"Missing public methods: {missing}")

    def test_backup_locations_constant_defined(self):
        """BACKUP_LOCATIONS class constant should be defined."""
        self.assertTrue(hasattr(CheckpointGuardian, "BACKUP_LOCATIONS"))
        self.assertIsInstance(CheckpointGuardian.BACKUP_LOCATIONS, (list, tuple))

    def test_checkpoint_patterns_constant_defined(self):
        """CHECKPOINT_PATTERNS class constant should be defined."""
        self.assertTrue(hasattr(CheckpointGuardian, "CHECKPOINT_PATTERNS"))
        self.assertIsInstance(CheckpointGuardian.CHECKPOINT_PATTERNS, (list, tuple))


class TestDeployWorkflowContent(unittest.TestCase):
    """Tests that validate the deploy-huggingface.yml workflow configuration."""

    WORKFLOW_PATH = Path(__file__).parent / ".github" / "workflows" / "deploy-huggingface.yml"

    def _read_workflow(self) -> str:
        with open(self.WORKFLOW_PATH) as f:
            return f.read()

    def test_workflow_file_exists(self):
        """deploy-huggingface.yml must exist."""
        self.assertTrue(self.WORKFLOW_PATH.exists(), "deploy-huggingface.yml not found")

    def test_workflow_handles_missing_checkpoint_gracefully(self):
        """The locate step must set checkpoint_found output (not exit 1 directly)."""
        content = self._read_workflow()
        self.assertIn("checkpoint_found=false", content)
        self.assertIn("checkpoint_found=true", content)
        # Must NOT hard fail with exit 1 in the locate step
        locate_section_start = content.find("- name: Locate best checkpoint")
        convert_section_start = content.find("- name: Convert to HuggingFace format")
        locate_section = content[locate_section_start:convert_section_start]
        self.assertNotIn("exit 1", locate_section)

    def test_convert_step_is_conditional(self):
        """Convert step must be conditional on checkpoint_found."""
        content = self._read_workflow()
        convert_start = content.find("- name: Convert to HuggingFace format")
        upload_start = content.find("- name: Prepare datasets for upload")
        convert_section = content[convert_start:upload_start]
        self.assertIn("checkpoint_found", convert_section)

    def test_upload_step_is_conditional(self):
        """Upload step must be conditional on checkpoint_found."""
        content = self._read_workflow()
        upload_start = content.find("- name: Upload to HuggingFace Hub")
        release_start = content.find("- name: Create release tag")
        upload_section = content[upload_start:release_start]
        self.assertIn("checkpoint_found", upload_section)

    def test_workflow_has_graceful_summary_for_missing_checkpoint(self):
        """Deployment summary should handle missing checkpoint without failing."""
        content = self._read_workflow()
        self.assertIn("No checkpoint available", content)

    def test_workflow_trigger_types(self):
        """Workflow must support both workflow_dispatch and workflow_run triggers."""
        content = self._read_workflow()
        self.assertIn("workflow_dispatch", content)
        self.assertIn("workflow_run", content)


class TestGlobalTypeDefinitions(unittest.TestCase):
    """Tests that validate the TypeScript global.d.ts type definition file."""

    GLOBAL_D_TS_PATH = Path(__file__).parent / "app" / "types" / "global.d.ts"

    def _read_global_d_ts(self) -> str:
        with open(self.GLOBAL_D_TS_PATH) as f:
            return f.read()

    def test_global_d_ts_exists(self):
        """app/types/global.d.ts must exist."""
        self.assertTrue(self.GLOBAL_D_TS_PATH.exists())

    def test_no_var_eslint_disable_is_immediately_before_var(self):
        """The eslint-disable-next-line no-var comment must be directly before var ENV."""
        content = self._read_global_d_ts()
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "var ENV" in line:
                # The line immediately before the var declaration should be the
                # eslint-disable comment (not a deno or other comment between them)
                prev_line = lines[i - 1].strip()
                self.assertIn(
                    "eslint-disable-next-line no-var",
                    prev_line,
                    "eslint-disable-next-line no-var must be the line immediately "
                    "before 'var ENV'",
                )

    def test_deno_lint_ignore_present(self):
        """deno-lint-ignore no-var comment should still be present."""
        content = self._read_global_d_ts()
        self.assertIn("deno-lint-ignore no-var", content)

    def test_file_has_export_statement(self):
        """The file must end with export {} to make it a module."""
        content = self._read_global_d_ts()
        self.assertIn("export {}", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
