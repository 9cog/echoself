"""
Trainer for Deep Tree Echo models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import os
from pathlib import Path
from tqdm import tqdm
import time

from .optimizer import create_optimizer, create_scheduler
from .metrics import MetricsTracker


class EchoTrainer:
    """
    Trainer class for Deep Tree Echo LLM.
    Handles the complete training loop with checkpointing and evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Any,
        train_dataset: Any,
        val_dataset: Any
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Extract configurations
        self.training_config = config.training_config
        self.echo_config = config.echo_training_config
        self.hardware_config = config.hardware_config
        
        # Setup device
        self.device = torch.device(
            self.hardware_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model.to(self.device)
        
        # Create data loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False)
        
        # Setup optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            self.training_config['optimizer']
        )
        
        self.scheduler = create_scheduler(
            self.optimizer,
            self.training_config['scheduler'],
            self.training_config['max_steps']
        )
        
        # Setup mixed precision training
        self.use_amp = self.training_config.get('mixed_precision', 'no') != 'no'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Checkpointing
        self.checkpoint_dir = Path(self.training_config['checkpoint'].get('save_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        self.early_stopping = self.training_config['early_stopping'].get('enable', False)
        self.patience = self.training_config['early_stopping'].get('patience', 10)
        self.patience_counter = 0
        
        # Progressive depth training
        self.progressive_depth = self.echo_config['progressive_depth'].get('enable', False)
        self.current_depth = self.echo_config['progressive_depth'].get('initial_depth', 1)
    
    def _create_dataloader(self, dataset: Any, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader from a dataset."""
        return DataLoader(
            dataset,
            batch_size=self.config.data_config['loader']['batch_size'],
            shuffle=shuffle,
            num_workers=self.config.data_config['loader'].get('num_workers', 4),
            pin_memory=self.config.data_config['loader'].get('pin_memory', True)
        )
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Total steps: {self.training_config['max_steps']}")
        print(f"Batch size: {self.config.data_config['loader']['batch_size']}")
        print(f"Gradient accumulation: {self.training_config['gradient_accumulation_steps']}")
        
        self.model.train()
        accumulation_steps = self.training_config['gradient_accumulation_steps']
        
        try:
            while self.global_step < self.training_config['max_steps']:
                self.epoch += 1
                self._train_epoch(accumulation_steps)
                
                # Evaluation
                if self.global_step % self.training_config['evaluation']['eval_steps'] == 0:
                    val_loss = self.evaluate()
                    
                    # Check for improvement
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.save_checkpoint('best_model')
                    else:
                        self.patience_counter += 1
                    
                    # Early stopping
                    if self.early_stopping and self.patience_counter >= self.patience:
                        print(f"Early stopping triggered after {self.global_step} steps")
                        break
                
                # Progressive depth update
                if self.progressive_depth:
                    self._update_tree_depth()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        print(f"Training completed! Final step: {self.global_step}")
        self.save_checkpoint('final_model')
    
    def _train_epoch(self, accumulation_steps: int):
        """Train for one epoch."""
        epoch_loss = 0
        n_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs['loss']
            else:
                outputs = self.model(**batch)
                loss = outputs['loss']
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                grad_clip_val = self.training_config.get('gradient_clipping', 0)
                if grad_clip_val and float(grad_clip_val) > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(grad_clip_val)
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Update metrics
                self.metrics.update('train_loss', loss.item() * accumulation_steps)
                self.metrics.update('learning_rate', self.scheduler.get_last_lr()[0])
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * accumulation_steps:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                    'step': self.global_step
                })
                
                # Checkpointing
                if self.global_step % self.training_config['checkpoint']['save_steps'] == 0:
                    self.save_checkpoint(f'checkpoint_{self.global_step}')
                
                # Check if we've reached max steps
                if self.global_step >= self.training_config['max_steps']:
                    break
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {self.epoch} completed. Average loss: {avg_epoch_loss:.4f}")
    
    def evaluate(self) -> float:
        """Evaluate the model on validation set."""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
                
                total_loss += outputs['loss'].item()
                n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        self.metrics.update('val_loss', avg_loss)
        
        print(f"Validation loss: {avg_loss:.4f}")
        
        self.model.train()
        return avg_loss
    
    def _update_tree_depth(self):
        """Update tree depth for progressive training."""
        increment_every = self.echo_config['progressive_depth']['increment_every']
        max_depth = self.echo_config['progressive_depth']['max_depth']
        
        if self.global_step % increment_every == 0 and self.current_depth < max_depth:
            self.current_depth += 1
            print(f"Increasing tree depth to {self.current_depth}")
            
            # Update model's tree depth (if model supports it)
            if hasattr(self.model, 'set_tree_depth'):
                self.model.set_tree_depth(self.current_depth)
    
    def save_checkpoint(self, name: str):
        """Save a model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'metrics': self.metrics.get_all()
        }
        
        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load a model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from step {self.global_step}, epoch {self.epoch}")