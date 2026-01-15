#!/usr/bin/env python3
"""
Training Cache Manager for EchoSelf Model Training

This module implements a sophisticated build artifact caching system that enables
iterative improvement across training sessions instead of starting from scratch.

Features:
- Smart checkpoint management with metadata tracking
- Training state continuity (optimizer, scheduler, model state)
- Data pipeline caching and versioning
- Quality-based checkpoint selection
- Storage efficiency with cleanup policies
- Workflow integration for CI/CD environments
"""

import os
import json
import time
import shutil
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import torch
import numpy as np


@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint."""
    checkpoint_id: str
    created_at: str
    iteration: int
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    data_config: Dict[str, Any]
    metrics: Dict[str, float]
    tags: List[str]
    parent_checkpoint: Optional[str] = None
    notes: str = ""
    file_size_mb: float = 0.0
    quality_score: float = 0.0


@dataclass
class CacheConfig:
    """Configuration for the training cache."""
    cache_dir: str = "cache/training"
    max_checkpoints: int = 10
    max_cache_size_gb: float = 50.0
    min_improvement_threshold: float = 0.01
    checkpoint_interval: int = 1000
    quality_weight_loss: float = 0.7
    quality_weight_metrics: float = 0.3
    auto_cleanup: bool = True
    compress_checkpoints: bool = True
    data_cache_enabled: bool = True
    data_cache_ttl_hours: int = 168  # 1 week


class TrainingCache:
    """
    Advanced training cache manager for iterative model improvement.
    
    This class manages checkpoint storage, versioning, quality assessment,
    and intelligent resumption to enable continuous model improvement
    across training sessions.
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.checkpoints_dir = self.cache_dir / "checkpoints"
        self.data_cache_dir = self.cache_dir / "data"
        self.metadata_file = self.cache_dir / "metadata.json"
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.data_cache_dir.mkdir(exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
        
        print(f"🗂️  Training cache initialized at {self.cache_dir}")
        print(f"   Current checkpoints: {len(self.metadata)}")
        
        # Auto cleanup if enabled
        if config.auto_cleanup:
            self._cleanup_cache()
    
    def _load_metadata(self) -> Dict[str, CheckpointMetadata]:
        """Load checkpoint metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                return {
                    k: CheckpointMetadata(**v) for k, v in data.items()
                }
            except Exception as e:
                print(f"⚠️  Failed to load metadata: {e}")
                # Backup corrupted metadata
                backup_file = self.metadata_file.with_suffix('.backup')
                if self.metadata_file.exists():
                    shutil.copy2(self.metadata_file, backup_file)
        
        return {}
    
    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        try:
            data = {k: asdict(v) for k, v in self.metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save metadata: {e}")
    
    def _generate_checkpoint_id(
        self, 
        iteration: int, 
        model_config: Dict[str, Any],
        data_hash: str
    ) -> str:
        """Generate a unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(
            json.dumps(model_config, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"ckpt_{timestamp}_{iteration}_{config_hash}_{data_hash[:8]}"
    
    def _calculate_quality_score(self, metadata: CheckpointMetadata) -> float:
        """Calculate quality score for a checkpoint."""
        # Lower loss is better (inverse relationship)
        loss_score = 1.0 / (1.0 + metadata.val_loss)
        
        # Metrics score (higher is better)
        metrics_score = 0.0
        if metadata.metrics:
            # Consider specific metrics like accuracy, perplexity, etc.
            metrics_score = sum(metadata.metrics.values()) / len(metadata.metrics)
        
        # Weighted combination
        quality_score = (
            self.config.quality_weight_loss * loss_score +
            self.config.quality_weight_metrics * metrics_score
        )
        
        return quality_score
    
    def _get_data_hash(self, data_config: Dict[str, Any]) -> str:
        """Generate hash for data configuration."""
        # Create a deterministic hash of data configuration
        config_str = json.dumps(data_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        iteration: int,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        data_config: Dict[str, Any],
        metrics: Dict[str, float] = None,
        tags: List[str] = None,
        notes: str = "",
        force_save: bool = False
    ) -> str:
        """Save a training checkpoint with full state."""
        
        # Generate checkpoint ID
        data_hash = self._get_data_hash(data_config)
        checkpoint_id = self._generate_checkpoint_id(iteration, model_config, data_hash)
        
        # Check if we should save based on improvement
        if not force_save and self.metadata:
            best_loss = min(m.val_loss for m in self.metadata.values())
            improvement = (best_loss - val_loss) / best_loss
            if improvement < self.config.min_improvement_threshold:
                print(f"⏭️  Skipping save - improvement {improvement:.3%} below threshold")
                return checkpoint_id
        
        # Create checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'iteration': iteration,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'model_config': model_config,
            'training_config': training_config,
            'data_config': data_config,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
            'checkpoint_id': checkpoint_id,
            # Save connection ratio for backward compatibility and debugging
            'connection_ratio': getattr(model, 'connection_ratio', 0.0),
            'current_iteration': getattr(model, 'current_iteration', iteration)
        }
        
        # Save checkpoint file
        checkpoint_path = self.checkpoints_dir / f"{checkpoint_id}.pt"
        try:
            torch.save(checkpoint_data, checkpoint_path)
            file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            
            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                created_at=datetime.now().isoformat(),
                iteration=iteration,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=learning_rate,
                model_config=model_config,
                training_config=training_config,
                data_config=data_config,
                metrics=metrics or {},
                tags=tags or [],
                notes=notes,
                file_size_mb=file_size_mb
            )
            
            # Calculate quality score
            metadata.quality_score = self._calculate_quality_score(metadata)
            
            # Store metadata
            self.metadata[checkpoint_id] = metadata
            self._save_metadata()
            
            print(f"💾 Saved checkpoint {checkpoint_id}")
            print(f"   File size: {file_size_mb:.1f} MB")
            print(f"   Quality score: {metadata.quality_score:.3f}")
            print(f"   Val loss: {val_loss:.4f}")
            
            # Cleanup if needed
            if self.config.auto_cleanup:
                self._cleanup_cache()
            
            return checkpoint_id
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            raise
    
    def load_checkpoint(
        self, 
        checkpoint_id: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu"
    ) -> Tuple[Dict[str, Any], CheckpointMetadata]:
        """Load a checkpoint and optionally restore model/optimizer state."""
        
        # Find best checkpoint if none specified
        if checkpoint_id is None:
            checkpoint_id = self.get_best_checkpoint()
            if checkpoint_id is None:
                raise ValueError("No checkpoints available")
        
        if checkpoint_id not in self.metadata:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        # Load checkpoint file
        checkpoint_path = self.checkpoints_dir / f"{checkpoint_id}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
            metadata = self.metadata[checkpoint_id]
            
            # Restore states if objects provided
            if model is not None:
                model.load_state_dict(checkpoint_data['model_state_dict'])
                print(f"✅ Restored model state from {checkpoint_id}")
            
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                print(f"✅ Restored optimizer state from {checkpoint_id}")
            
            if scheduler is not None and checkpoint_data['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                print(f"✅ Restored scheduler state from {checkpoint_id}")
            
            print(f"📂 Loaded checkpoint {checkpoint_id}")
            print(f"   Iteration: {metadata.iteration}")
            print(f"   Val loss: {metadata.val_loss:.4f}")
            print(f"   Quality score: {metadata.quality_score:.3f}")
            
            return checkpoint_data, metadata
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint {checkpoint_id}: {e}")
            raise
    
    def get_best_checkpoint(
        self, 
        metric: str = "quality_score",
        tags: List[str] = None
    ) -> Optional[str]:
        """Get the best checkpoint based on specified metric."""
        if not self.metadata:
            return None
        
        # Filter by tags if specified
        candidates = self.metadata
        if tags:
            candidates = {
                k: v for k, v in candidates.items()
                if any(tag in v.tags for tag in tags)
            }
        
        if not candidates:
            return None
        
        # Find best based on metric
        if metric == "quality_score":
            best_id = max(candidates.keys(), key=lambda k: candidates[k].quality_score)
        elif metric == "val_loss":
            best_id = min(candidates.keys(), key=lambda k: candidates[k].val_loss)
        elif metric == "iteration":
            best_id = max(candidates.keys(), key=lambda k: candidates[k].iteration)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_id
    
    def get_compatible_checkpoints(
        self,
        model_config: Dict[str, Any],
        data_config: Dict[str, Any]
    ) -> List[str]:
        """Get checkpoints compatible with current configuration."""
        compatible = []
        current_data_hash = self._get_data_hash(data_config)
        
        for checkpoint_id, metadata in self.metadata.items():
            # Check model compatibility (architecture must match)
            if self._is_model_compatible(metadata.model_config, model_config):
                # Check data compatibility
                checkpoint_data_hash = self._get_data_hash(metadata.data_config)
                if checkpoint_data_hash == current_data_hash:
                    compatible.append(checkpoint_id)
        
        # Sort by quality score
        compatible.sort(
            key=lambda k: self.metadata[k].quality_score,
            reverse=True
        )
        
        return compatible
    
    def _is_model_compatible(
        self,
        saved_config: Dict[str, Any],
        current_config: Dict[str, Any]
    ) -> bool:
        """Check if saved model is compatible with current configuration."""
        # Core architecture parameters that must match
        critical_params = ['n_layer', 'n_head', 'n_embd', 'vocab_size', 'block_size']
        
        for param in critical_params:
            if saved_config.get(param) != current_config.get(param):
                return False
        
        return True
    
    def cache_data(
        self,
        data_key: str,
        data: Any,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Cache preprocessed data with metadata."""
        if not self.config.data_cache_enabled:
            return data_key
        
        cache_path = self.data_cache_dir / f"{data_key}.pkl"
        metadata_path = self.data_cache_dir / f"{data_key}_meta.json"
        
        try:
            # Save data
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Save metadata
            cache_metadata = {
                'created_at': datetime.now().isoformat(),
                'data_key': data_key,
                'file_size_mb': cache_path.stat().st_size / (1024 * 1024),
                'metadata': metadata or {}
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(cache_metadata, f, indent=2)
            
            print(f"💾 Cached data: {data_key} ({cache_metadata['file_size_mb']:.1f} MB)")
            return data_key
            
        except Exception as e:
            print(f"❌ Failed to cache data {data_key}: {e}")
            # Clean up partial files
            for path in [cache_path, metadata_path]:
                if path.exists():
                    path.unlink()
            raise
    
    def load_cached_data(self, data_key: str) -> Optional[Any]:
        """Load cached data if available and valid."""
        if not self.config.data_cache_enabled:
            return None
        
        cache_path = self.data_cache_dir / f"{data_key}.pkl"
        metadata_path = self.data_cache_dir / f"{data_key}_meta.json"
        
        if not cache_path.exists() or not metadata_path.exists():
            return None
        
        try:
            # Check if cache is still valid
            with open(metadata_path, 'r') as f:
                cache_metadata = json.load(f)
            
            created_at = datetime.fromisoformat(cache_metadata['created_at'])
            ttl = timedelta(hours=self.config.data_cache_ttl_hours)
            
            if datetime.now() - created_at > ttl:
                print(f"⏰ Data cache {data_key} expired, removing...")
                cache_path.unlink()
                metadata_path.unlink()
                return None
            
            # Load data
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"📂 Loaded cached data: {data_key} ({cache_metadata['file_size_mb']:.1f} MB)")
            return data
            
        except Exception as e:
            print(f"⚠️  Failed to load cached data {data_key}: {e}")
            # Clean up corrupted cache
            for path in [cache_path, metadata_path]:
                if path.exists():
                    path.unlink()
            return None
    
    def _cleanup_cache(self):
        """Clean up cache based on configured policies."""
        if not self.metadata:
            return
        
        print("🧹 Running cache cleanup...")
        
        # Sort by quality score (keep best ones)
        sorted_checkpoints = sorted(
            self.metadata.items(),
            key=lambda x: x[1].quality_score,
            reverse=True
        )
        
        # Calculate total size
        total_size_gb = sum(m.file_size_mb for _, m in sorted_checkpoints) / 1024
        
        # Remove excess checkpoints
        removed_count = 0
        for i, (checkpoint_id, metadata) in enumerate(sorted_checkpoints):
            should_remove = False
            
            # Check count limit
            if i >= self.config.max_checkpoints:
                should_remove = True
                
            # Check size limit
            if total_size_gb > self.config.max_cache_size_gb:
                should_remove = True
                total_size_gb -= metadata.file_size_mb / 1024
            
            if should_remove:
                self._remove_checkpoint(checkpoint_id)
                removed_count += 1
        
        if removed_count > 0:
            print(f"🗑️  Removed {removed_count} checkpoints during cleanup")
            self._save_metadata()
    
    def _remove_checkpoint(self, checkpoint_id: str):
        """Remove a checkpoint and its files."""
        try:
            # Remove checkpoint file
            checkpoint_path = self.checkpoints_dir / f"{checkpoint_id}.pt"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # Remove from metadata
            if checkpoint_id in self.metadata:
                del self.metadata[checkpoint_id]
            
            print(f"🗑️  Removed checkpoint {checkpoint_id}")
            
        except Exception as e:
            print(f"⚠️  Failed to remove checkpoint {checkpoint_id}: {e}")
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints sorted by quality."""
        checkpoints = list(self.metadata.values())
        checkpoints.sort(key=lambda x: x.quality_score, reverse=True)
        return checkpoints
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.metadata:
            return {
                'total_checkpoints': 0,
                'total_size_mb': 0,
                'total_size_gb': 0,
                'best_quality_score': 0,
                'best_val_loss': 0
            }
        
        total_size_mb = sum(m.file_size_mb for m in self.metadata.values())
        best_checkpoint = max(self.metadata.values(), key=lambda x: x.quality_score)
        
        return {
            'total_checkpoints': len(self.metadata),
            'total_size_mb': total_size_mb,
            'total_size_gb': total_size_mb / 1024,
            'best_quality_score': best_checkpoint.quality_score,
            'best_val_loss': best_checkpoint.val_loss,
            'cache_dir': str(self.cache_dir)
        }
    
    def export_best_checkpoint(
        self,
        export_path: str,
        include_optimizer: bool = False
    ) -> str:
        """Export the best checkpoint for deployment."""
        best_id = self.get_best_checkpoint()
        if not best_id:
            raise ValueError("No checkpoints available")
        
        checkpoint_data, metadata = self.load_checkpoint(best_id)
        
        # Create export data
        export_data = {
            'model_state_dict': checkpoint_data['model_state_dict'],
            'model_config': checkpoint_data['model_config'],
            'training_config': checkpoint_data['training_config'],
            'metadata': asdict(metadata),
            'exported_at': datetime.now().isoformat()
        }
        
        if include_optimizer:
            export_data['optimizer_state_dict'] = checkpoint_data['optimizer_state_dict']
            export_data['scheduler_state_dict'] = checkpoint_data['scheduler_state_dict']
        
        # Save export
        torch.save(export_data, export_path)
        
        print(f"📦 Exported best checkpoint to {export_path}")
        print(f"   Checkpoint ID: {best_id}")
        print(f"   Quality score: {metadata.quality_score:.3f}")
        
        return best_id


def create_training_cache(cache_dir: str = "cache/training") -> TrainingCache:
    """Create a training cache with default configuration."""
    config = CacheConfig(cache_dir=cache_dir)
    return TrainingCache(config)


if __name__ == "__main__":
    # Demo usage
    cache = create_training_cache()
    stats = cache.get_cache_stats()
    
    print(f"\n📊 Cache Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    checkpoints = cache.list_checkpoints()
    if checkpoints:
        print(f"\n📋 Available Checkpoints:")
        for i, ckpt in enumerate(checkpoints[:5]):  # Show top 5
            print(f"   {i+1}. {ckpt.checkpoint_id}")
            print(f"      Quality: {ckpt.quality_score:.3f} | Loss: {ckpt.val_loss:.4f}")
            print(f"      Iteration: {ckpt.iteration} | Size: {ckpt.file_size_mb:.1f} MB")