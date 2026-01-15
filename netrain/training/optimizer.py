"""
Optimizer and scheduler utilities for Deep Tree Echo training
"""

import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    LambdaLR
)
from typing import Dict, Any
import math


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create an optimizer based on configuration.
    
    Args:
        model: The model to optimize
        config: Optimizer configuration
        
    Returns:
        Configured optimizer
    """
    optimizer_type = config.get('type', 'AdamW').lower()
    lr = float(config.get('learning_rate', 2e-4))
    weight_decay = float(config.get('weight_decay', 0.01))
    
    # Get model parameters, potentially with weight decay exclusion
    params = get_parameter_groups(model, weight_decay)
    
    if optimizer_type == 'adamw':
        optimizer = AdamW(
            params,
            lr=lr,
            betas=(float(config.get('beta1', 0.9)), float(config.get('beta2', 0.999))),
            eps=float(config.get('eps', 1e-8)),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adam':
        optimizer = Adam(
            params,
            lr=lr,
            betas=(float(config.get('beta1', 0.9)), float(config.get('beta2', 0.999))),
            eps=float(config.get('eps', 1e-8)),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = SGD(
            params,
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def get_parameter_groups(model: torch.nn.Module, weight_decay: float):
    """
    Separate parameters into groups with and without weight decay.
    LayerNorm and bias parameters typically don't use weight decay.
    
    Args:
        model: The model
        weight_decay: Weight decay value
        
    Returns:
        Parameter groups for optimizer
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter should have weight decay
        if 'bias' in name or 'ln' in name or 'layernorm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    total_steps: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler based on configuration.
    
    Args:
        optimizer: The optimizer
        config: Scheduler configuration
        total_steps: Total training steps
        
    Returns:
        Configured scheduler
    """
    scheduler_type = config.get('type', 'cosine_with_warmup').lower()
    warmup_steps = config.get('warmup_steps', 2000)
    
    if scheduler_type == 'cosine_with_warmup':
        # Create warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Create cosine scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=config.get('min_lr_ratio', 0.1) * optimizer.param_groups[0]['lr']
        )
        
        # Combine with sequential scheduler
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=config.get('min_lr_ratio', 0.1) * optimizer.param_groups[0]['lr']
        )
    
    elif scheduler_type == 'cosine_with_restarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('restart_period', 10000),
            T_mult=config.get('restart_mult', 2),
            eta_min=config.get('min_lr_ratio', 0.1) * optimizer.param_groups[0]['lr']
        )
    
    elif scheduler_type == 'linear':
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.get('min_lr_ratio', 0.1),
            total_iters=total_steps
        )
    
    elif scheduler_type == 'custom_cosine_with_warmup':
        # Custom implementation with more control
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_with_warmup_lr_lambda(
                step,
                warmup_steps,
                total_steps,
                config.get('min_lr_ratio', 0.1)
            )
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def cosine_with_warmup_lr_lambda(
    current_step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
) -> float:
    """
    Calculate learning rate multiplier for cosine schedule with warmup.
    
    Args:
        current_step: Current training step
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum learning rate as ratio of initial
        
    Returns:
        Learning rate multiplier
    """
    if current_step < warmup_steps:
        # Linear warmup
        return current_step / warmup_steps
    else:
        # Cosine decay
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))