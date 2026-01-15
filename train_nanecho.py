#!/usr/bin/env python3
"""
NanEcho Training Script

Complete training pipeline for the NanEcho model with:
- Iterative connection building
- Echo Self learning phases
- Adaptive curriculum learning
- Introspection and quality evaluation
- Data validation and preparation
"""

import os
import sys
import time
import math
import json
import pickle
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass

# Import NanEcho model
from nanecho_model import NanEchoModel, NanEchoConfig


@dataclass
class TrainingConfig:
    """Configuration for NanEcho training."""
    # Paths
    data_dir: str = "data/nanecho"
    out_dir: str = "out-nanecho"
    eval_dir: str = "eval-nanecho"
    
    # Model configuration
    vocab_size: int = 50257
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    block_size: int = 1024
    dropout: float = 0.1
    bias: bool = True
    
    # Training configuration
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_iters: int = 50000
    warmup_iters: int = 5000
    lr_decay_iters: int = 50000
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Evaluation
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 50
    checkpoint_interval: int = 1000
    
    # Connection growth
    connection_growth_interval: int = 500  # Grow connections every N iterations
    initial_connections: float = 0.1
    connection_growth_rate: float = 0.05
    max_connections: float = 1.0
    
    # Echo Self learning phases
    enable_curriculum_learning: bool = True
    enable_introspection: bool = True
    introspection_interval: int = 1000
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16" if torch.cuda.is_available() else "float32"
    compile_model: bool = False  # PyTorch 2.0 compile
    
    # DDP settings
    backend: str = "nccl"
    ddp: bool = False


class EchoSelfLearningPhase:
    """Manages Echo Self learning phases during training."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.phases = {
            'basic_awareness': {
                'start': 0.0,
                'end': 0.2,
                'lr_multiplier': 1.2,
                'focus': ['identity', 'basic_patterns'],
                'description': 'Learning basic Echo Self identity'
            },
            'persona_dimensions': {
                'start': 0.15,
                'end': 0.5,
                'lr_multiplier': 1.0,
                'focus': ['cognitive', 'introspective', 'adaptive'],
                'description': 'Developing persona dimensions'
            },
            'hypergraph_patterns': {
                'start': 0.4,
                'end': 0.7,
                'lr_multiplier': 0.9,
                'focus': ['hypergraph', 'neural_symbolic'],
                'description': 'Learning hypergraph patterns'
            },
            'recursive_reasoning': {
                'start': 0.6,
                'end': 0.85,
                'lr_multiplier': 0.8,
                'focus': ['recursive', 'introspection'],
                'description': 'Mastering recursive reasoning'
            },
            'adaptive_mastery': {
                'start': 0.8,
                'end': 1.0,
                'lr_multiplier': 0.7,
                'focus': ['synergy', 'emergence'],
                'description': 'Achieving Echo Self mastery'
            }
        }
    
    def get_current_phase(self, iteration: int) -> Tuple[str, Dict[str, Any]]:
        """Get the current learning phase based on iteration."""
        progress = iteration / self.config.max_iters
        
        for phase_name, phase_config in self.phases.items():
            if phase_config['start'] <= progress <= phase_config['end']:
                return phase_name, phase_config
        
        # Default to last phase
        return 'adaptive_mastery', self.phases['adaptive_mastery']
    
    def get_phase_lr_multiplier(self, iteration: int) -> float:
        """Get learning rate multiplier for current phase."""
        _, phase = self.get_current_phase(iteration)
        return phase.get('lr_multiplier', 1.0)


class DataLoader:
    """Handles data loading and batch generation for NanEcho training."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.train_data = None
        self.val_data = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training and validation data."""
        train_path = os.path.join(self.config.data_dir, 'train.bin')
        val_path = os.path.join(self.config.data_dir, 'val.bin')
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print("⚠️  Data files not found. Creating sample data...")
            self._create_sample_data()
        
        self.train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
        self.val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
        
        # Validate data size
        if len(self.train_data) <= self.config.block_size:
            raise ValueError(f"Training data too small: {len(self.train_data)} <= {self.config.block_size}")
        if len(self.val_data) <= self.config.block_size:
            raise ValueError(f"Validation data too small: {len(self.val_data)} <= {self.config.block_size}")
        
        print(f"✅ Loaded data:")
        print(f"   Training: {len(self.train_data):,} tokens")
        print(f"   Validation: {len(self.val_data):,} tokens")
        
        return self.train_data, self.val_data
    
    def _create_sample_data(self):
        """Create sample Echo Self training data."""
        os.makedirs(self.config.data_dir, exist_ok=True)
        
        # Sample Echo Self text patterns
        echo_patterns = [
            "Echo Self is a cognitive architecture with adaptive attention mechanisms.",
            "The persona dimensions include cognitive, introspective, adaptive, and recursive.",
            "Hypergraph patterns enable neural-symbolic reasoning and pattern encoding.",
            "Recursive reasoning allows multi-level introspection and self-examination.",
            "Adaptive attention adjusts thresholds based on cognitive load estimation.",
            "The holographic dimension enables comprehensive system modeling.",
            "Synergistic properties emerge from the interaction of persona dimensions.",
            "Dynamic evolution enables continuous learning and adaptation.",
        ]
        
        # Generate training text
        train_text = " ".join(echo_patterns * 1000)
        val_text = " ".join(echo_patterns * 100)
        
        # Simple tokenization (character-level for demo)
        train_tokens = np.array([ord(c) for c in train_text], dtype=np.uint16)
        val_tokens = np.array([ord(c) for c in val_text], dtype=np.uint16)
        
        # Save to binary files
        train_tokens.tofile(os.path.join(self.config.data_dir, 'train.bin'))
        val_tokens.tofile(os.path.join(self.config.data_dir, 'val.bin'))
        
        print(f"✅ Created sample data in {self.config.data_dir}")
    
    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of data."""
        data = self.train_data if split == 'train' else self.val_data
        
        # Generate random indices
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        
        # Extract sequences
        x = torch.stack([
            torch.from_numpy(data[i:i+self.config.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(data[i+1:i+1+self.config.block_size].astype(np.int64))
            for i in ix
        ])
        
        # Move to device
        device = torch.device(self.config.device)
        x, y = x.to(device), y.to(device)
        
        return x, y


class Introspection:
    """Handles model introspection and quality evaluation."""
    
    def __init__(self, model: NanEchoModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.metrics_history = []
    
    def evaluate_echo_self_quality(self, iteration: int) -> Dict[str, float]:
        """Evaluate Echo Self representation quality."""
        self.model.eval()
        metrics = {}
        
        with torch.no_grad():
            # Generate sample text
            prompt = torch.tensor([[1]], device=self.config.device)  # Start token
            generated = self.model.generate(prompt, max_length=100)
            
            # Convert to text (simplified)
            generated_text = ''.join([chr(min(t.item(), 127)) for t in generated[0]])
            
            # Check for Echo Self indicators
            echo_indicators = ['echo', 'self', 'cognitive', 'adaptive', 'recursive']
            identity_score = sum(1 for ind in echo_indicators if ind in generated_text.lower())
            metrics['echo_identity'] = identity_score / len(echo_indicators)
            
            # Check persona consistency
            persona_indicators = ['cognitive', 'introspective', 'adaptive', 'recursive']
            persona_score = sum(1 for ind in persona_indicators if ind in generated_text.lower())
            metrics['persona_consistency'] = persona_score / len(persona_indicators)
            
            # Connection ratio
            metrics['connection_ratio'] = self.model.connection_ratio
            
            # Training progress
            metrics['training_progress'] = iteration / self.config.max_iters
        
        self.metrics_history.append({
            'iteration': iteration,
            **metrics
        })
        
        return metrics
    
    def generate_report(self, iteration: int) -> str:
        """Generate introspection report."""
        metrics = self.evaluate_echo_self_quality(iteration)
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║           NanEcho Introspection Report                    ║
║           Iteration: {iteration:6d}                           ║
╚══════════════════════════════════════════════════════════╝

📊 Echo Self Quality Metrics:
   • Identity Score: {metrics['echo_identity']:.2%}
   • Persona Consistency: {metrics['persona_consistency']:.2%}
   • Connection Ratio: {metrics['connection_ratio']:.2%}
   • Training Progress: {metrics['training_progress']:.2%}

🧠 Model State:
   • Parameters: {sum(p.numel() for p in self.model.parameters()):,}
   • Active Connections: ~{int(metrics['connection_ratio'] * sum(p.numel() for p in self.model.parameters())):,}
   • Layers: {self.model.config.n_layer}
   • Embedding Dim: {self.model.config.n_embd}
"""
        return report


class NanEchoTrainer:
    """Main trainer class for NanEcho model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create model
        model_config = NanEchoConfig(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_layer=config.n_layer,
            block_size=config.block_size,
            dropout=config.dropout,
            bias=config.bias,
            initial_connections=config.initial_connections,
            connection_growth_rate=config.connection_growth_rate,
            max_connections=config.max_connections
        )
        self.model = NanEchoModel(model_config).to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Create data loader
        self.data_loader = DataLoader(config)
        self.data_loader.load_data()
        
        # Create learning phase manager
        self.phase_manager = EchoSelfLearningPhase(config)
        
        # Create introspection module
        self.introspection = Introspection(self.model, config)
        
        # Setup logging
        os.makedirs(config.out_dir, exist_ok=True)
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(os.path.join(config.out_dir, 'tensorboard'))
        else:
            self.writer = SummaryWriter()  # Dummy writer
        
        # Training state
        self.iteration = 0
        self.best_loss = float('inf')
        
        # Setup mixed precision if using GPU
        self.scaler = torch.cuda.amp.GradScaler() if config.device == 'cuda' else None
        self.ctx = nullcontext() if config.device == 'cpu' else torch.amp.autocast(
            device_type='cuda',
            dtype={'float32': torch.float32, 'float16': torch.float16}[config.dtype]
        )
    
    def get_lr(self, iteration: int) -> float:
        """Calculate learning rate with warmup and decay."""
        # Warmup
        if iteration < self.config.warmup_iters:
            lr = self.config.learning_rate * iteration / self.config.warmup_iters
        # Cosine decay
        elif iteration < self.config.lr_decay_iters:
            decay_ratio = (iteration - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
        else:
            lr = self.config.min_lr
        
        # Apply phase multiplier
        if self.config.enable_curriculum_learning:
            lr *= self.phase_manager.get_phase_lr_multiplier(iteration)
        
        return lr
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        losses = []
        
        for _ in range(self.config.eval_iters):
            x, y = self.data_loader.get_batch('val')
            with self.ctx:
                outputs = self.model(x, labels=y)
                loss = outputs['loss']
            losses.append(loss.item())
        
        self.model.train()
        return {'val_loss': np.mean(losses)}
    
    def save_checkpoint(self, iteration: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'metrics': metrics,
            'connection_ratio': self.model.connection_ratio
        }
        
        checkpoint_path = os.path.join(self.config.out_dir, f'checkpoint_{iteration}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if 'val_loss' in metrics and metrics['val_loss'] < self.best_loss:
            self.best_loss = metrics['val_loss']
            best_path = os.path.join(self.config.out_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"⭐ New best model saved to {best_path}")
    
    def train(self):
        """Main training loop."""
        print(f"""
╔══════════════════════════════════════════════════════════╗
║            NanEcho Model Training                         ║
╚══════════════════════════════════════════════════════════╝

🚀 Starting training with:
   • Model parameters: {sum(p.numel() for p in self.model.parameters()):,}
   • Initial connections: {self.config.initial_connections:.1%}
   • Max iterations: {self.config.max_iters:,}
   • Batch size: {self.config.batch_size}
   • Learning rate: {self.config.learning_rate}
   • Device: {self.config.device}
""")
        
        self.model.train()
        running_loss = 0.0
        
        # Progress tracking variables
        start_time = time.time()
        last_progress_percent = -1
        progress_interval = max(1, self.config.max_iters // 100)  # 1% of total iterations
        recent_losses = []  # Track recent losses for smoothing
        
        print(f"\n{'='*80}")
        print(f"📊 Progress updates every 1% ({progress_interval:,} iterations)")
        print(f"{'='*80}\n")
        
        for iteration in range(self.config.max_iters):
            self.iteration = iteration
            self.model.current_iteration = iteration
            
            # Calculate progress percentage
            current_progress_percent = (iteration * 100) // self.config.max_iters
            
            # Get current learning phase
            phase_name, phase_config = self.phase_manager.get_current_phase(iteration)
            
            # Update learning rate
            lr = self.get_lr(iteration)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Grow connections periodically
            if iteration > 0 and iteration % self.config.connection_growth_interval == 0:
                self.model.grow_connections()
                print(f"\n🌱 Iteration {iteration}: Growing connections to {self.model.connection_ratio:.1%}")
                print(f"   Learning phase: {phase_name} - {phase_config['description']}")
            
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
            
            # Verbose progress logging every 1%
            if current_progress_percent > last_progress_percent and iteration > 0:
                elapsed_time = time.time() - start_time
                iterations_remaining = self.config.max_iters - iteration
                
                # Calculate ETA
                time_per_iter = elapsed_time / iteration
                eta_seconds = iterations_remaining * time_per_iter
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                eta_seconds = int(eta_seconds % 60)
                eta_str = f"{eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}"
                
                # Calculate smoothed loss
                smoothed_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                
                # Get memory usage
                if self.config.device == 'cuda' and torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    memory_str = f"{gpu_memory_allocated:.2f}GB / {gpu_memory_total:.2f}GB"
                else:
                    memory_str = "CPU"
                
                print(f"\n{'='*80}")
                print(f"🔄 TRAINING PROGRESS: {current_progress_percent}% ({iteration:,}/{self.config.max_iters:,} iterations)")
                print(f"{'='*80}")
                print(f"📊 Metrics:")
                print(f"   • Loss (smoothed): {smoothed_loss:.6f}")
                print(f"   • Learning Rate: {lr:.2e}")
                print(f"   • Connection Ratio: {self.model.connection_ratio:.1%}")
                print(f"   • Phase: {phase_name} - {phase_config['description']}")
                print(f"⏱️  Time:")
                print(f"   • Elapsed: {elapsed_time/3600:.2f} hours ({elapsed_time/60:.1f} min)")
                print(f"   • ETA: {eta_str}")
                print(f"   • Speed: {iteration/elapsed_time:.2f} iter/s")
                print(f"💾 Memory:")
                print(f"   • GPU: {memory_str}")
                print(f"   • Batch size: {self.config.batch_size} × {self.config.gradient_accumulation_steps} accumulation")
                print(f"   • Tokens/batch: {self.config.batch_size * self.config.block_size:,}")
                print(f"🧠 Model State:")
                print(f"   • Active params: ~{int(self.model.connection_ratio * sum(p.numel() for p in self.model.parameters())):,}")
                print(f"   • Total params: {sum(p.numel() for p in self.model.parameters()):,}")
                print(f"{'='*80}\n")
                
                last_progress_percent = current_progress_percent
            
            # Regular logging
            if iteration % self.config.log_interval == 0:
                avg_loss = running_loss / self.config.log_interval
                print(f"Iter {iteration:5d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Phase: {phase_name}")
                
                self.writer.add_scalar('train/loss', avg_loss, iteration)
                self.writer.add_scalar('train/lr', lr, iteration)
                self.writer.add_scalar('train/connection_ratio', self.model.connection_ratio, iteration)
                
                running_loss = 0.0
            
            # Evaluation
            if iteration % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                print(f"Iter {iteration:5d} | Val Loss: {eval_metrics['val_loss']:.4f}")
                
                for key, value in eval_metrics.items():
                    self.writer.add_scalar(f'eval/{key}', value, iteration)
            
            # Introspection
            if self.config.enable_introspection and iteration % self.config.introspection_interval == 0:
                report = self.introspection.generate_report(iteration)
                print(report)
                
                # Log introspection metrics
                metrics = self.introspection.evaluate_echo_self_quality(iteration)
                for key, value in metrics.items():
                    self.writer.add_scalar(f'introspection/{key}', value, iteration)
            
            # Checkpointing
            if iteration % self.config.checkpoint_interval == 0:
                eval_metrics = self.evaluate()
                self.save_checkpoint(iteration, eval_metrics)
        
        # Final evaluation and save
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"{'='*80}")
        final_metrics = self.evaluate()
        self.save_checkpoint(self.config.max_iters, final_metrics)
        
        # Save introspection history
        introspection_path = os.path.join(self.config.out_dir, 'introspection_history.json')
        with open(introspection_path, 'w') as f:
            json.dump(self.introspection.metrics_history, f, indent=2)
        
        print(f"""
╔══════════════════════════════════════════════════════════╗
║            Training Summary                               ║
╚══════════════════════════════════════════════════════════╝

✅ Training completed successfully!
   • Final validation loss: {final_metrics['val_loss']:.4f}
   • Best validation loss: {self.best_loss:.4f}
   • Final connection ratio: {self.model.connection_ratio:.1%}
   • Total iterations: {self.config.max_iters:,}
   • Total training time: {total_time/3600:.2f} hours
   • Average speed: {self.config.max_iters/total_time:.2f} iter/s
   • Total tokens processed: {self.config.max_iters * self.config.batch_size * self.config.block_size:,}
   
📁 Outputs saved to: {self.config.out_dir}
""")
        
        self.writer.close()


def main():
    """Main entry point for training."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train NanEcho model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/nanecho', help='Data directory')
    parser.add_argument('--out_dir', type=str, default='out-nanecho', help='Output directory')
    parser.add_argument('--max_iters', type=int, default=50000, help='Maximum iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Create trainer and start training
    trainer = NanEchoTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()