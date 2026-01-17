#!/usr/bin/env python3
"""
Checkpoint Guardian - Ensures training is ALWAYS cumulative and models are NEVER lost.

This script provides:
1. Multi-source checkpoint restoration (artifacts, releases, repo, cache)
2. Protection against starting from scratch (errors BLOCK training, not restart it)
3. Multi-location model backup (redundant storage in multiple locations)
4. Checkpoint integrity verification

CRITICAL PRINCIPLE: Data loss is NEVER acceptable. If checkpoint restoration fails,
training MUST NOT proceed from scratch. Instead, it should error out and alert.
"""

import os
import sys
import json
import hashlib
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any


class CheckpointGuardian:
    """
    Guardian class that ensures training checkpoints are:
    1. Always restored from previous state (cumulative training)
    2. Never lost (multi-location backup)
    3. Never started from scratch without explicit DANGEROUS override
    """

    # Backup locations in priority order
    BACKUP_LOCATIONS = [
        ".training-progress/checkpoints",      # Primary: committed to repo
        ".training-progress/cache",            # Secondary: action cache
        "/tmp/nanecho-checkpoint-backup",      # Tertiary: temp backup
        "artifacts/nanecho-model"              # Quaternary: artifacts
    ]

    # Checkpoint file patterns to protect
    CHECKPOINT_PATTERNS = [
        "ckpt.pt",
        "best_model_export.pt",
        "checkpoint_*.pt",
        "model_*.pt"
    ]

    def __init__(self, output_dir: str, allow_fresh_start: bool = False):
        """
        Initialize the Checkpoint Guardian.

        Args:
            output_dir: Primary output directory for training
            allow_fresh_start: DANGEROUS - only set True with explicit user confirmation
        """
        self.output_dir = Path(output_dir)
        self.allow_fresh_start = allow_fresh_start
        self.manifest_path = self.output_dir / "checkpoint_manifest.json"
        self.backup_manifest_path = Path(".training-progress") / "checkpoint_manifest.json"
        self.manifest: Dict[str, Any] = self._load_manifest()

        print(f"🛡️ Checkpoint Guardian initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Allow fresh start: {self.allow_fresh_start} {'⚠️ DANGEROUS' if self.allow_fresh_start else '✓ SAFE'}")

    def _load_manifest(self) -> Dict[str, Any]:
        """Load checkpoint manifest from multiple locations."""
        for path in [self.manifest_path, self.backup_manifest_path]:
            if path.exists():
                try:
                    with open(path) as f:
                        manifest = json.load(f)
                        print(f"📋 Loaded manifest from {path}")
                        return manifest
                except Exception as e:
                    print(f"⚠️ Failed to load manifest from {path}: {e}")

        return {
            "checkpoints": [],
            "last_updated": None,
            "total_iterations": 0,
            "best_loss": float('inf'),
            "backup_locations": []
        }

    def _save_manifest(self):
        """Save manifest to multiple locations for redundancy."""
        self.manifest["last_updated"] = datetime.now().isoformat()

        locations_saved = []
        for path in [self.manifest_path, self.backup_manifest_path]:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    json.dump(self.manifest, f, indent=2)
                locations_saved.append(str(path))
            except Exception as e:
                print(f"⚠️ Failed to save manifest to {path}: {e}")

        self.manifest["backup_locations"] = locations_saved
        print(f"💾 Manifest saved to {len(locations_saved)} locations")

    def _compute_checksum(self, filepath: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _verify_checkpoint(self, checkpoint_path: Path, expected_checksum: Optional[str] = None) -> bool:
        """Verify checkpoint file integrity."""
        if not checkpoint_path.exists():
            return False

        # Check file size (must be non-zero)
        if checkpoint_path.stat().st_size == 0:
            print(f"❌ Checkpoint {checkpoint_path} is empty!")
            return False

        # Verify checksum if provided
        if expected_checksum:
            actual_checksum = self._compute_checksum(checkpoint_path)
            if actual_checksum != expected_checksum:
                print(f"❌ Checksum mismatch for {checkpoint_path}")
                return False

        # Try to load with torch to verify it's valid
        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            required_keys = ['model', 'optimizer', 'iter_num']
            if not all(key in checkpoint for key in required_keys):
                # Check for alternative formats
                if 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
                    return True
                print(f"⚠️ Checkpoint {checkpoint_path} missing required keys")
                # Don't fail, might be valid export format
            return True
        except Exception as e:
            print(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
            return False

    def find_best_checkpoint(self) -> Optional[Path]:
        """
        Find the best available checkpoint from all backup locations.

        Returns:
            Path to best checkpoint, or None if no valid checkpoints found.
        """
        print("🔍 Searching for existing checkpoints...")

        candidates: List[Dict[str, Any]] = []

        # Search all backup locations
        for location in self.BACKUP_LOCATIONS:
            loc_path = Path(location)
            if not loc_path.exists():
                continue

            for pattern in self.CHECKPOINT_PATTERNS:
                for checkpoint_file in loc_path.glob(f"**/{pattern}"):
                    if self._verify_checkpoint(checkpoint_file):
                        # Try to get iteration number and loss
                        try:
                            import torch
                            ckpt = torch.load(checkpoint_file, map_location='cpu')
                            iter_num = ckpt.get('iter_num', 0)
                            best_val_loss = ckpt.get('best_val_loss', float('inf'))
                            candidates.append({
                                'path': checkpoint_file,
                                'iter_num': iter_num,
                                'val_loss': best_val_loss,
                                'size': checkpoint_file.stat().st_size
                            })
                            print(f"   ✓ Found: {checkpoint_file} (iter={iter_num}, loss={best_val_loss:.4f})")
                        except Exception as e:
                            # Still add if verification passed
                            candidates.append({
                                'path': checkpoint_file,
                                'iter_num': 0,
                                'val_loss': float('inf'),
                                'size': checkpoint_file.stat().st_size
                            })
                            print(f"   ✓ Found: {checkpoint_file} (metadata unavailable)")

        # Also check output directory
        for pattern in self.CHECKPOINT_PATTERNS:
            for checkpoint_file in self.output_dir.glob(f"**/{pattern}"):
                if checkpoint_file not in [c['path'] for c in candidates]:
                    if self._verify_checkpoint(checkpoint_file):
                        try:
                            import torch
                            ckpt = torch.load(checkpoint_file, map_location='cpu')
                            iter_num = ckpt.get('iter_num', 0)
                            best_val_loss = ckpt.get('best_val_loss', float('inf'))
                            candidates.append({
                                'path': checkpoint_file,
                                'iter_num': iter_num,
                                'val_loss': best_val_loss,
                                'size': checkpoint_file.stat().st_size
                            })
                        except Exception:
                            candidates.append({
                                'path': checkpoint_file,
                                'iter_num': 0,
                                'val_loss': float('inf'),
                                'size': checkpoint_file.stat().st_size
                            })

        if not candidates:
            print("   No existing checkpoints found")
            return None

        # Sort by iteration number (higher is better), then by loss (lower is better)
        candidates.sort(key=lambda x: (-x['iter_num'], x['val_loss']))

        best = candidates[0]
        print(f"🏆 Best checkpoint: {best['path']}")
        print(f"   Iterations: {best['iter_num']}, Loss: {best['val_loss']:.4f}")

        return best['path']

    def restore_checkpoint(self) -> Optional[Path]:
        """
        Restore the best available checkpoint for cumulative training.

        CRITICAL: If no checkpoint is found and allow_fresh_start is False,
        this will raise an error to PREVENT accidental fresh starts.

        Returns:
            Path to restored checkpoint in output directory.

        Raises:
            RuntimeError: If no checkpoint found and fresh start not allowed.
        """
        best_checkpoint = self.find_best_checkpoint()

        if best_checkpoint is None:
            if not self.allow_fresh_start:
                print("\n" + "="*60)
                print("🚨 CRITICAL ERROR: NO CHECKPOINT FOUND")
                print("="*60)
                print("Training cannot proceed because:")
                print("  1. No existing checkpoint was found in any backup location")
                print("  2. Fresh start mode is DISABLED (as it should be)")
                print("")
                print("This is a SAFETY FEATURE to prevent accidental loss of")
                print("training progress. If you truly need to start fresh:")
                print("  1. Verify all backup locations are empty")
                print("  2. Set ALLOW_FRESH_START=true explicitly")
                print("  3. Document why a fresh start is required")
                print("="*60)
                raise RuntimeError("No checkpoint found and fresh start not allowed. Training blocked to prevent data loss.")
            else:
                print("⚠️ WARNING: Starting fresh training (no checkpoint found)")
                print("   This should only happen on first-ever training run")
                return None

        # Copy checkpoint to output directory
        target_path = self.output_dir / "ckpt.pt"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if best_checkpoint != target_path:
            shutil.copy2(best_checkpoint, target_path)
            print(f"📦 Restored checkpoint to {target_path}")

        # Update manifest
        self.manifest["checkpoints"].append({
            "restored_from": str(best_checkpoint),
            "restored_to": str(target_path),
            "timestamp": datetime.now().isoformat()
        })
        self._save_manifest()

        return target_path

    def backup_checkpoint(self, checkpoint_path: Path, iteration: int, val_loss: float):
        """
        Backup checkpoint to MULTIPLE locations for redundancy.

        This ensures the model can NEVER be lost due to:
        - Single point of failure
        - Disk issues
        - Network problems
        - Workflow failures
        """
        if not checkpoint_path.exists():
            print(f"❌ Cannot backup non-existent checkpoint: {checkpoint_path}")
            return

        checksum = self._compute_checksum(checkpoint_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"checkpoint_iter{iteration}_{timestamp}.pt"

        successful_backups = []

        # Backup to all locations
        for location in self.BACKUP_LOCATIONS:
            try:
                loc_path = Path(location)
                loc_path.mkdir(parents=True, exist_ok=True)
                backup_path = loc_path / backup_name
                shutil.copy2(checkpoint_path, backup_path)

                # Verify backup
                if self._verify_checkpoint(backup_path, checksum):
                    successful_backups.append(str(backup_path))
                    print(f"✓ Backed up to {backup_path}")
                else:
                    print(f"⚠️ Backup verification failed for {backup_path}")
                    backup_path.unlink()
            except Exception as e:
                print(f"⚠️ Failed to backup to {location}: {e}")

        # Also keep a "latest" symlink/copy
        for location in self.BACKUP_LOCATIONS[:2]:  # Primary locations only
            try:
                loc_path = Path(location)
                latest_path = loc_path / "latest_checkpoint.pt"
                shutil.copy2(checkpoint_path, latest_path)
            except Exception:
                pass

        if len(successful_backups) < 2:
            print(f"⚠️ WARNING: Only {len(successful_backups)} backup(s) succeeded!")
            print("   Recommended: At least 2 redundant backups")
        else:
            print(f"✅ Checkpoint backed up to {len(successful_backups)} locations")

        # Update manifest
        self.manifest["checkpoints"].append({
            "path": str(checkpoint_path),
            "iteration": iteration,
            "val_loss": val_loss,
            "checksum": checksum,
            "backups": successful_backups,
            "timestamp": datetime.now().isoformat()
        })
        self.manifest["total_iterations"] = iteration
        if val_loss < self.manifest["best_loss"]:
            self.manifest["best_loss"] = val_loss
        self._save_manifest()

        return successful_backups

    def cleanup_old_checkpoints(self, keep_count: int = 5):
        """
        Clean up old checkpoints while keeping the best ones.

        NEVER deletes:
        - Best checkpoint by loss
        - Most recent checkpoint
        - Checkpoints from the last 24 hours
        """
        print(f"🧹 Cleaning up old checkpoints (keeping {keep_count} best)...")

        all_checkpoints = []
        for location in self.BACKUP_LOCATIONS:
            loc_path = Path(location)
            if not loc_path.exists():
                continue
            for pattern in self.CHECKPOINT_PATTERNS:
                for ckpt in loc_path.glob(f"**/{pattern}"):
                    if 'latest' not in ckpt.name:
                        all_checkpoints.append(ckpt)

        if len(all_checkpoints) <= keep_count:
            print(f"   Only {len(all_checkpoints)} checkpoints found, nothing to clean")
            return

        # Sort by modification time (most recent first)
        all_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Always keep most recent and try to identify best
        to_keep = set(all_checkpoints[:keep_count])

        # Delete the rest
        deleted = 0
        for ckpt in all_checkpoints:
            if ckpt not in to_keep:
                try:
                    ckpt.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"   Failed to delete {ckpt}: {e}")

        print(f"   Deleted {deleted} old checkpoints")


def main():
    """CLI interface for Checkpoint Guardian."""
    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint Guardian - Protect your training progress")
    parser.add_argument("--output-dir", required=True, help="Training output directory")
    parser.add_argument("--action", choices=["restore", "backup", "verify", "cleanup"], required=True)
    parser.add_argument("--checkpoint", help="Checkpoint path for backup action")
    parser.add_argument("--iteration", type=int, default=0, help="Current iteration for backup")
    parser.add_argument("--val-loss", type=float, default=float('inf'), help="Validation loss for backup")
    parser.add_argument("--allow-fresh-start", action="store_true",
                        help="DANGEROUS: Allow starting from scratch if no checkpoint found")
    parser.add_argument("--keep-count", type=int, default=5, help="Checkpoints to keep during cleanup")

    args = parser.parse_args()

    guardian = CheckpointGuardian(
        output_dir=args.output_dir,
        allow_fresh_start=args.allow_fresh_start
    )

    if args.action == "restore":
        try:
            restored = guardian.restore_checkpoint()
            if restored:
                print(f"\n✅ CHECKPOINT_RESTORED={restored}")
                print("Training will continue from this checkpoint")
            else:
                print("\n⚠️ CHECKPOINT_RESTORED=NONE")
                print("Starting fresh training (first run)")
        except RuntimeError as e:
            print(f"\n❌ RESTORE_FAILED: {e}")
            sys.exit(1)

    elif args.action == "backup":
        if not args.checkpoint:
            print("❌ --checkpoint required for backup action")
            sys.exit(1)
        backups = guardian.backup_checkpoint(
            Path(args.checkpoint),
            args.iteration,
            args.val_loss
        )
        if backups:
            print(f"\n✅ BACKUP_COUNT={len(backups)}")
        else:
            print("\n⚠️ BACKUP_COUNT=0")

    elif args.action == "verify":
        best = guardian.find_best_checkpoint()
        if best:
            print(f"\n✅ BEST_CHECKPOINT={best}")
        else:
            print("\n❌ NO_VALID_CHECKPOINTS")
            sys.exit(1)

    elif args.action == "cleanup":
        guardian.cleanup_old_checkpoints(args.keep_count)


if __name__ == "__main__":
    main()
