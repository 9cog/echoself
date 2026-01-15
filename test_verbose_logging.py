#!/usr/bin/env python3
"""
Test script to demonstrate verbose logging functionality.
This script creates a minimal training scenario to show the 1% progress updates.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    """Simple model for testing verbose logging."""
    def __init__(self, vocab_size=100, n_embd=128, n_layer=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([
            nn.Linear(n_embd, n_embd) for _ in range(n_layer)
        ])
        self.output = nn.Linear(n_embd, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x.mean(dim=1))

def simulate_verbose_training(max_iters=100, log_every_percent=1):
    """
    Simulate training with verbose logging every 1%.
    
    Args:
        max_iters: Total number of training iterations
        log_every_percent: Log progress every N percent (default 1%)
    """
    print(f"\n{'='*80}")
    print(f"🚀 VERBOSE TRAINING SIMULATION")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  • Max iterations: {max_iters:,}")
    print(f"  • Progress updates: Every {log_every_percent}%")
    print(f"  • Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'='*80}\n")
    
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Progress tracking
    start_time = time.time()
    last_progress_percent = -1
    progress_interval = max(1, max_iters // (100 // log_every_percent))
    
    # Training metrics
    running_loss = 0.0
    recent_losses = []
    
    print(f"Starting training loop...")
    print(f"Progress updates every {progress_interval} iterations ({log_every_percent}%)\n")
    
    for iteration in range(max_iters):
        # Simulate training step
        batch_size = 8
        seq_len = 32
        
        # Generate random data
        x = torch.randint(0, 100, (batch_size, seq_len)).to(device)
        y = torch.randint(0, 100, (batch_size,)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track loss
        running_loss += loss.item()
        recent_losses.append(loss.item())
        if len(recent_losses) > 20:
            recent_losses.pop(0)
        
        # Calculate progress
        current_progress_percent = ((iteration + 1) * 100) // max_iters
        
        # Verbose logging every 1%
        if current_progress_percent > last_progress_percent:
            elapsed_time = time.time() - start_time
            iterations_done = iteration + 1
            iterations_remaining = max_iters - iterations_done
            
            # Calculate ETA
            if iterations_done > 0:
                time_per_iter = elapsed_time / iterations_done
                eta_seconds = iterations_remaining * time_per_iter
                eta_minutes = int(eta_seconds // 60)
                eta_seconds = int(eta_seconds % 60)
                eta_str = f"{eta_minutes:02d}:{eta_seconds:02d}"
            else:
                eta_str = "--:--"
            
            # Calculate smoothed loss
            smoothed_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            
            # Calculate learning rate (with cosine schedule)
            lr = 1e-3 * (0.5 * (1 + math.cos(math.pi * iterations_done / max_iters)))
            
            print(f"\n{'='*80}")
            print(f"🔄 TRAINING PROGRESS: {current_progress_percent}% ({iterations_done:,}/{max_iters:,} iterations)")
            print(f"{'='*80}")
            print(f"📊 Metrics:")
            print(f"   • Loss (current): {loss.item():.6f}")
            print(f"   • Loss (smoothed): {smoothed_loss:.6f}")
            print(f"   • Learning Rate: {lr:.2e}")
            print(f"⏱️  Time:")
            print(f"   • Elapsed: {elapsed_time:.2f} seconds")
            print(f"   • ETA: {eta_str}")
            print(f"   • Speed: {iterations_done/elapsed_time:.2f} iter/s")
            print(f"   • Time per iter: {time_per_iter*1000:.2f}ms")
            
            if torch.cuda.is_available():
                print(f"💾 Memory:")
                print(f"   • GPU: {torch.cuda.memory_allocated()/1e6:.2f}MB allocated")
                print(f"   • GPU: {torch.cuda.max_memory_allocated()/1e6:.2f}MB peak")
            
            print(f"🧠 Model:")
            print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"   • Gradients: {sum(p.grad.numel() for p in model.parameters() if p.grad is not None):,}")
            print(f"{'='*80}")
            
            last_progress_percent = current_progress_percent
        
        # Simulate some work to make it more realistic
        time.sleep(0.01)  # Small delay to simulate computation
    
    # Final summary
    total_time = time.time() - start_time
    avg_loss = running_loss / max_iters
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"📊 Final Statistics:")
    print(f"   • Total iterations: {max_iters:,}")
    print(f"   • Final loss: {loss.item():.6f}")
    print(f"   • Average loss: {avg_loss:.6f}")
    print(f"   • Total time: {total_time:.2f} seconds")
    print(f"   • Average speed: {max_iters/total_time:.2f} iter/s")
    print(f"   • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*80}\n")

def main():
    """Main entry point for testing."""
    import argparse
    parser = argparse.ArgumentParser(description='Test verbose logging functionality')
    parser.add_argument('--max_iters', type=int, default=100,
                       help='Maximum number of iterations (default: 100)')
    parser.add_argument('--log_every', type=int, default=1,
                       help='Log progress every N percent (default: 1)')
    args = parser.parse_args()
    
    simulate_verbose_training(
        max_iters=args.max_iters,
        log_every_percent=args.log_every
    )

if __name__ == "__main__":
    main()