#!/usr/bin/env python3
"""
Enhanced Training Script with Build Artifact Caching

This script integrates the TrainingCache system to enable iterative improvement
across training sessions. Instead of starting from scratch, it intelligently
resumes from the best available checkpoint and continues building upon previous
training progress.
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import numpy as np

# Import training cache system
from training_cache import TrainingCache, CacheConfig, CheckpointMetadata

# Import existing training components
from nanecho_model import NanEchoModel, NanEchoConfig
from train_nanecho import (
    TrainingConfig, NanEchoTrainer, DataLoader, 
    EchoSelfLearningPhase, Introspection
)


class CachedNanEchoTrainer(NanEchoTrainer):
    """
    Enhanced NanEcho trainer with build artifact caching capabilities.
    
    This trainer automatically:
    1. Checks for compatible checkpoints to resume from
    2. Saves high-quality checkpoints during training
    3. Manages storage and cleanup of training artifacts
    4. Enables true iterative improvement across sessions
    """
    
    def __init__(
        self, 
        config: TrainingConfig,
        cache_config: Optional[CacheConfig] = None,
        force_fresh_start: bool = False
    ):
        # Initialize base trainer
        super().__init__(config)
        
        # Initialize training cache
        cache_config = cache_config or CacheConfig(
            cache_dir=os.path.join(config.out_dir, "cache"),
            max_checkpoints=10,
            max_cache_size_gb=20.0,
            min_improvement_threshold=0.005,  # 0.5% improvement required
            checkpoint_interval=config.checkpoint_interval
        )
        
        self.cache = TrainingCache(cache_config)
        self.force_fresh_start = force_fresh_start
        
        # Resume from checkpoint if available
        self.resumed_from_checkpoint = False
        self.starting_iteration = 0
        self.starting_epoch = 0
        
        if not force_fresh_start:
            self._attempt_resume_from_cache()
    
    def _attempt_resume_from_cache(self) -> bool:
        """Attempt to resume from the best compatible checkpoint."""
        print("\n🔍 Checking for compatible checkpoints to resume from...")
        
        # Create configuration fingerprints
        model_config = {
            'n_layer': self.config.n_layer,
            'n_head': self.config.n_head,
            'n_embd': self.config.n_embd,
            'vocab_size': self.config.vocab_size,
            'block_size': self.config.block_size,
            'dropout': self.config.dropout,
            'bias': self.config.bias
        }
        
        data_config = {
            'data_dir': self.config.data_dir,
            'batch_size': self.config.batch_size,
            'block_size': self.config.block_size
        }
        
        # Find compatible checkpoints
        compatible_checkpoints = self.cache.get_compatible_checkpoints(
            model_config, data_config
        )
        
        if not compatible_checkpoints:
            print("📝 No compatible checkpoints found - starting fresh training")
            return False
        
        # Load the best compatible checkpoint
        best_checkpoint_id = compatible_checkpoints[0]
        try:
            print(f"🔄 Attempting to resume from checkpoint: {best_checkpoint_id}")
            
            checkpoint_data, metadata = self.cache.load_checkpoint(
                best_checkpoint_id,
                model=self.model,
                optimizer=self.optimizer,
                device=self.config.device
            )
            
            # Update training state
            self.starting_iteration = metadata.iteration
            self.starting_epoch = metadata.epoch
            self.best_loss = metadata.val_loss
            self.resumed_from_checkpoint = True
            
            # Restore model iteration for progressive features
            self.model.current_iteration = checkpoint_data.get('current_iteration', metadata.iteration)
            
            # Restore connection ratio if applicable - this should now be automatic via state_dict
            if hasattr(self.model, 'connection_ratio'):
                # Connection ratio should be restored from the model state dict,
                # but we can double-check and restore from saved value if needed
                saved_ratio = checkpoint_data.get('connection_ratio', None)
                if saved_ratio is not None and abs(self.model.connection_ratio - saved_ratio) > 0.01:
                    print(f"⚠️  Connection ratio mismatch, adjusting: {self.model.connection_ratio:.3f} -> {saved_ratio:.3f}")
                    self.model.connection_ratio = saved_ratio
            
            print(f"✅ Successfully resumed from checkpoint!")
            print(f"   Starting iteration: {self.starting_iteration:,}")
            print(f"   Starting epoch: {self.starting_epoch}")
            print(f"   Previous best loss: {self.best_loss:.4f}")
            print(f"   Connection ratio: {getattr(self.model, 'connection_ratio', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to resume from checkpoint {best_checkpoint_id}: {e}")
            print("📝 Falling back to fresh training")
            return False
    
    def _create_checkpoint_tags(self, iteration: int, metrics: Dict[str, float]) -> list:
        """Create tags for checkpoint categorization."""
        tags = []
        
        # Phase-based tags
        phase_name, _ = self.phase_manager.get_current_phase(iteration)
        tags.append(f"phase_{phase_name}")
        
        # Quality-based tags
        if 'val_loss' in metrics:
            if metrics['val_loss'] < 2.0:
                tags.append("high_quality")
            elif metrics['val_loss'] < 4.0:
                tags.append("medium_quality")
            else:
                tags.append("low_quality")
        
        # Milestone tags
        if iteration % 10000 == 0:
            tags.append("milestone")
        
        # Model-specific tags
        tags.append("nanecho")
        if self.config.enable_curriculum_learning:
            tags.append("curriculum")
        if self.config.enable_introspection:
            tags.append("introspection")
        
        return tags
    
    def save_training_checkpoint(
        self,
        iteration: int,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, float] = None,
        force_save: bool = False
    ) -> Optional[str]:
        """Save a training checkpoint using the cache system."""
        
        # Prepare configurations
        model_config = {
            'n_layer': self.config.n_layer,
            'n_head': self.config.n_head,
            'n_embd': self.config.n_embd,
            'vocab_size': self.config.vocab_size,
            'block_size': self.config.block_size,
            'dropout': self.config.dropout,
            'bias': self.config.bias,
            'initial_connections': self.config.initial_connections,
            'connection_growth_rate': self.config.connection_growth_rate,
            'max_connections': self.config.max_connections
        }
        
        training_config = {
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'max_iters': self.config.max_iters,
            'warmup_iters': self.config.warmup_iters,
            'lr_decay_iters': self.config.lr_decay_iters,
            'min_lr': self.config.min_lr,
            'weight_decay': self.config.weight_decay,
            'grad_clip': self.config.grad_clip,
            'enable_curriculum_learning': self.config.enable_curriculum_learning,
            'enable_introspection': self.config.enable_introspection
        }
        
        data_config = {
            'data_dir': self.config.data_dir,
            'batch_size': self.config.batch_size,
            'block_size': self.config.block_size
        }
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Create enhanced metrics
        enhanced_metrics = metrics or {}
        enhanced_metrics.update({
            'connection_ratio': getattr(self.model, 'connection_ratio', 0.0),
            'tokens_processed': iteration * self.config.batch_size * self.config.block_size,
            'training_speed_iters_per_sec': getattr(self, '_recent_speed', 0.0)
        })
        
        # Generate tags
        tags = self._create_checkpoint_tags(iteration, enhanced_metrics)
        
        # Create notes
        notes = f"Training checkpoint at iteration {iteration}"
        if self.resumed_from_checkpoint:
            notes += f" (resumed from iteration {self.starting_iteration})"
        
        phase_name, phase_config = self.phase_manager.get_current_phase(iteration)
        notes += f" | Phase: {phase_name}"
        
        try:
            checkpoint_id = self.cache.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=None,  # Add scheduler if implemented
                iteration=iteration,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                model_config=model_config,
                training_config=training_config,
                data_config=data_config,
                metrics=enhanced_metrics,
                tags=tags,
                notes=notes,
                force_save=force_save
            )
            
            return checkpoint_id
            
        except Exception as e:
            print(f"⚠️  Failed to save checkpoint: {e}")
            return None
    
    def train(self):
        """Enhanced training loop with caching integration."""
        print(f"""
╔══════════════════════════════════════════════════════════╗
║         NanEcho Model Training with Caching              ║
╚══════════════════════════════════════════════════════════╝

🚀 Training Configuration:
   • Model parameters: {sum(p.numel() for p in self.model.parameters()):,}
   • Initial connections: {self.config.initial_connections:.1%}
   • Max iterations: {self.config.max_iters:,}
   • Batch size: {self.config.batch_size}
   • Learning rate: {self.config.learning_rate}
   • Device: {self.config.device}
   • Cache enabled: ✅
   
📊 Cache Status:
""")
        
        # Display cache statistics
        cache_stats = self.cache.get_cache_stats()
        for key, value in cache_stats.items():
            if isinstance(value, float):
                print(f"   • {key}: {value:.2f}")
            else:
                print(f"   • {key}: {value}")
        
        if self.resumed_from_checkpoint:
            print(f"\n🔄 Resuming from checkpoint:")
            print(f"   • Starting iteration: {self.starting_iteration:,}")
            print(f"   • Previous best loss: {self.best_loss:.4f}")
        
        print()
        
        self.model.train()
        running_loss = 0.0
        
        # Adjust starting points if resumed
        actual_max_iters = self.config.max_iters
        start_iteration = self.starting_iteration
        
        # Progress tracking variables
        start_time = time.time()
        last_progress_percent = -1
        progress_interval = max(1, (actual_max_iters - start_iteration) // 100)
        recent_losses = []
        last_checkpoint_time = start_time
        
        print(f"{'='*80}")
        print(f"📊 Progress updates every 1% ({progress_interval:,} iterations)")
        print(f"{'='*80}\n")
        
        for iteration in range(start_iteration, actual_max_iters):
            self.iteration = iteration
            self.model.current_iteration = iteration
            
            # Calculate progress percentage
            progress = iteration - start_iteration
            total_progress = actual_max_iters - start_iteration
            current_progress_percent = (progress * 100) // total_progress if total_progress > 0 else 0
            
            # Get current learning phase
            phase_name, phase_config = self.phase_manager.get_current_phase(iteration)
            
            # Update learning rate
            lr = self.get_lr(iteration)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Grow connections periodically - use absolute iteration count, not relative
            if iteration > 0 and iteration % self.config.connection_growth_interval == 0:
                old_ratio = self.model.connection_ratio
                self.model.grow_connections()
                new_ratio = self.model.connection_ratio
                
                if new_ratio > old_ratio:  # Only log if connections actually grew
                    print(f"\n🌱 Iteration {iteration}: Growing connections {old_ratio:.1%} → {new_ratio:.1%}")
                    print(f"   Learning phase: {phase_name} - {phase_config['description']}")
                    
                    # Force save checkpoint after connection growth
                    if iteration % self.config.eval_interval == 0:
                        eval_metrics = self.evaluate()
                        self.save_training_checkpoint(
                            iteration, 0, running_loss, eval_metrics['val_loss'],
                            eval_metrics, force_save=True
                        )
            
            # Training step
            self.optimizer.zero_grad(set_to_none=True)
            
            # Accumulate gradients
            for micro_step in range(self.config.gradient_accumulation_steps):
                x, y = self.data_loader.get_batch('train')
                
                with self.ctx:
                    outputs = self.model(x, labels=y)
                    loss = outputs['loss'] / self.config.gradient_accumulation_steps
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                running_loss += loss.item()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Track recent losses for smoothing
            if len(recent_losses) > 100:
                recent_losses.pop(0)
            recent_losses.append(running_loss / self.config.gradient_accumulation_steps)
            
            # Calculate speed for metrics
            current_time = time.time()
            if iteration > start_iteration:
                self._recent_speed = (iteration - start_iteration) / (current_time - start_time)
            
            # Verbose progress logging every 1%
            if current_progress_percent > last_progress_percent and iteration > start_iteration:
                elapsed_time = current_time - start_time
                iterations_remaining = actual_max_iters - iteration
                
                # Calculate ETA
                if progress > 0:
                    time_per_iter = elapsed_time / progress
                    eta_seconds = iterations_remaining * time_per_iter
                    eta_hours = int(eta_seconds // 3600)
                    eta_minutes = int((eta_seconds % 3600) // 60)
                    eta_seconds = int(eta_seconds % 60)
                    eta_str = f"{eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}"
                else:
                    eta_str = "--:--:--"
                
                # Calculate smoothed loss
                smoothed_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                
                # Get memory usage
                if self.config.device == 'cuda' and torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    memory_str = f"{gpu_memory_allocated:.2f}GB / {gpu_memory_total:.2f}GB"
                else:
                    memory_str = "CPU"
                
                # Cache statistics
                cache_stats = self.cache.get_cache_stats()
                
                print(f"\n{'='*80}")
                print(f"🔄 TRAINING PROGRESS: {current_progress_percent}% ({iteration:,}/{actual_max_iters:,} iterations)")
                if self.resumed_from_checkpoint:
                    print(f"📈 Session Progress: {progress:,} iterations since resume")
                print(f"{'='*80}")
                print(f"📊 Metrics:")
                print(f"   • Loss (smoothed): {smoothed_loss:.6f}")
                print(f"   • Learning Rate: {lr:.2e}")
                print(f"   • Connection Ratio: {self.model.connection_ratio:.1%}")
                print(f"   • Phase: {phase_name} - {phase_config['description']}")
                print(f"⏱️  Time:")
                print(f"   • Elapsed: {elapsed_time/3600:.2f} hours ({elapsed_time/60:.1f} min)")
                print(f"   • ETA: {eta_str}")
                print(f"   • Speed: {getattr(self, '_recent_speed', 0):.2f} iter/s")
                print(f"💾 Memory & Cache:")
                print(f"   • GPU: {memory_str}")
                print(f"   • Cached checkpoints: {cache_stats['total_checkpoints']}")
                print(f"   • Cache size: {cache_stats['total_size_gb']:.1f} GB")
                if cache_stats['total_checkpoints'] > 0:
                    print(f"   • Best cached loss: {cache_stats['best_val_loss']:.4f}")
                print(f"🧠 Model State:")
                print(f"   • Active params: ~{int(self.model.connection_ratio * sum(p.numel() for p in self.model.parameters())):,}")
                print(f"   • Total params: {sum(p.numel() for p in self.model.parameters()):,}")
                print(f"{'='*80}\n")
                
                last_progress_percent = current_progress_percent
            
            # Regular logging
            if iteration % self.config.log_interval == 0:
                avg_loss = running_loss / self.config.log_interval
                print(f"Iter {iteration:5d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Phase: {phase_name}")
                
                if hasattr(self, 'writer'):
                    self.writer.add_scalar('train/loss', avg_loss, iteration)
                    self.writer.add_scalar('train/lr', lr, iteration)
                    self.writer.add_scalar('train/connection_ratio', self.model.connection_ratio, iteration)
                
                running_loss = 0.0
            
            # Evaluation and checkpointing
            if iteration % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                print(f"Iter {iteration:5d} | Val Loss: {eval_metrics['val_loss']:.4f}")
                
                # Save checkpoint using cache system
                checkpoint_id = self.save_training_checkpoint(
                    iteration, 0, avg_loss if 'avg_loss' in locals() else 0.0,
                    eval_metrics['val_loss'], eval_metrics
                )
                
                if checkpoint_id:
                    last_checkpoint_time = current_time
                
                if hasattr(self, 'writer'):
                    for key, value in eval_metrics.items():
                        self.writer.add_scalar(f'eval/{key}', value, iteration)
            
            # Introspection
            if self.config.enable_introspection and iteration % self.config.introspection_interval == 0:
                report = self.introspection.generate_report(iteration)
                print(report)
                
                # Log introspection metrics
                metrics = self.introspection.evaluate_echo_self_quality(iteration)
                if hasattr(self, 'writer'):
                    for key, value in metrics.items():
                        self.writer.add_scalar(f'introspection/{key}', value, iteration)
        
        # Final save and export
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"{'='*80}")
        
        final_metrics = self.evaluate()
        
        # Save final checkpoint
        final_checkpoint_id = self.save_training_checkpoint(
            actual_max_iters, 0, 0.0, final_metrics['val_loss'],
            final_metrics, force_save=True
        )
        
        # Export best model for deployment
        export_path = os.path.join(self.config.out_dir, 'best_model_export.pt')
        best_checkpoint_id = self.cache.export_best_checkpoint(export_path)
        
        # Save introspection history
        introspection_path = os.path.join(self.config.out_dir, 'introspection_history.json')
        with open(introspection_path, 'w') as f:
            json.dump(self.introspection.metrics_history, f, indent=2)
        
        # Final statistics
        cache_stats = self.cache.get_cache_stats()
        session_iterations = actual_max_iters - start_iteration
        
        print(f"""
╔══════════════════════════════════════════════════════════╗
║            Training Summary                               ║
╚══════════════════════════════════════════════════════════╝

✅ Training completed successfully!
   • Final validation loss: {final_metrics['val_loss']:.4f}
   • Best cached loss: {cache_stats.get('best_val_loss', 'N/A')}
   • Final connection ratio: {self.model.connection_ratio:.1%}
   • Session iterations: {session_iterations:,}
   • Total training time: {total_time/3600:.2f} hours
   • Average speed: {session_iterations/total_time:.2f} iter/s
   
📦 Artifacts:
   • Best model exported: {export_path}
   • Cached checkpoints: {cache_stats['total_checkpoints']}
   • Cache size: {cache_stats['total_size_gb']:.1f} GB
   • Final checkpoint: {final_checkpoint_id or 'N/A'}
   • Best checkpoint: {best_checkpoint_id}
   
📁 Outputs saved to: {self.config.out_dir}
""")
        
        if hasattr(self, 'writer'):
            self.writer.close()


def main():
    """Main entry point for cached training."""
    parser = argparse.ArgumentParser(description='Train NanEcho model with caching')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/nanecho', help='Data directory')
    parser.add_argument('--out_dir', type=str, default='out-nanecho-cached', help='Output directory')
    parser.add_argument('--cache_dir', type=str, help='Cache directory (default: out_dir/cache)')
    parser.add_argument('--max_iters', type=int, default=10000, help='Maximum iterations')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--force_fresh_start', action='store_true', help='Ignore cached checkpoints')
    parser.add_argument('--max_checkpoints', type=int, default=10, help='Maximum cached checkpoints')
    parser.add_argument('--max_cache_size_gb', type=float, default=20.0, help='Maximum cache size in GB')
    
    args = parser.parse_args()
    
    # Create training configuration
    config = TrainingConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        n_layer=6,  # Smaller for demo
        n_head=6,
        n_embd=384,
        eval_interval=250,
        checkpoint_interval=500
    )
    
    # Create cache configuration
    cache_dir = args.cache_dir or os.path.join(args.out_dir, "cache")
    cache_config = CacheConfig(
        cache_dir=cache_dir,
        max_checkpoints=args.max_checkpoints,
        max_cache_size_gb=args.max_cache_size_gb,
        min_improvement_threshold=0.005
    )
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Create cached trainer and start training
    trainer = CachedNanEchoTrainer(
        config, 
        cache_config=cache_config,
        force_fresh_start=args.force_fresh_start
    )
    trainer.train()


if __name__ == "__main__":
    main()