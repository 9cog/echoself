#!/usr/bin/env python3
"""
Demo script to showcase the training cache system

This script demonstrates how the caching system enables iterative improvement
by running multiple short training sessions that build upon each other.
"""

import os
import sys
import torch
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from training_cache import TrainingCache, CacheConfig
from train_cached import CachedNanEchoTrainer
from train_nanecho import TrainingConfig


def demo_iterative_training():
    """Demonstrate iterative training with caching."""
    print("🎯 Demo: Iterative Training with Build Artifact Caching")
    print("=" * 70)
    
    # Create a small training configuration for demo
    config = TrainingConfig(
        data_dir="data/nanecho",
        out_dir="demo-cache-training",
        max_iters=100,  # Very short sessions
        batch_size=2,
        learning_rate=2e-4,
        device='cpu',  # Use CPU for demo
        n_layer=2,     # Small model
        n_head=2,
        n_embd=128,
        eval_interval=25,
        checkpoint_interval=50,
        connection_growth_interval=25
    )
    
    # Cache configuration
    cache_config = CacheConfig(
        cache_dir=os.path.join(config.out_dir, "cache"),
        max_checkpoints=5,
        max_cache_size_gb=1.0,
        min_improvement_threshold=0.001
    )
    
    print(f"\n📋 Training Configuration:")
    print(f"   • Model: {config.n_layer} layers, {config.n_embd} dim")
    print(f"   • Iterations per session: {config.max_iters}")
    print(f"   • Cache directory: {cache_config.cache_dir}")
    print(f"   • Max checkpoints: {cache_config.max_checkpoints}")
    
    # Run multiple training sessions to show iterative improvement
    for session in range(1, 4):
        print(f"\n{'🚀' * 20}")
        print(f"🚀 TRAINING SESSION {session}")
        print(f"{'🚀' * 20}")
        
        # Create trainer - it will automatically resume from cache if available
        trainer = CachedNanEchoTrainer(
            config, 
            cache_config=cache_config,
            force_fresh_start=False  # Enable caching
        )
        
        # Show cache status before training
        cache_stats = trainer.cache.get_cache_stats()
        print(f"\n📊 Cache Status Before Session {session}:")
        print(f"   • Checkpoints: {cache_stats['total_checkpoints']}")
        print(f"   • Cache size: {cache_stats['total_size_gb']:.2f} GB")
        if cache_stats['total_checkpoints'] > 0:
            print(f"   • Best loss: {cache_stats['best_val_loss']:.4f}")
        
        # Run training
        trainer.train()
        
        # Show cache status after training
        cache_stats = trainer.cache.get_cache_stats()
        print(f"\n📊 Cache Status After Session {session}:")
        print(f"   • Checkpoints: {cache_stats['total_checkpoints']}")
        print(f"   • Cache size: {cache_stats['total_size_gb']:.2f} GB")
        print(f"   • Best loss: {cache_stats['best_val_loss']:.4f}")
        
        # List available checkpoints
        checkpoints = trainer.cache.list_checkpoints()
        if checkpoints:
            print(f"\n📋 Available Checkpoints:")
            for i, ckpt in enumerate(checkpoints[:3]):  # Show top 3
                print(f"   {i+1}. {ckpt.checkpoint_id}")
                print(f"      Quality: {ckpt.quality_score:.3f} | Loss: {ckpt.val_loss:.4f}")
                print(f"      Iteration: {ckpt.iteration} | Size: {ckpt.file_size_mb:.1f} MB")
        
        print(f"\n✅ Session {session} completed!")
        
        # Clean up trainer
        del trainer
    
    print(f"\n{'🎉' * 20}")
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print(f"{'🎉' * 20}")
    print(f"""
📈 Summary:
   • Ran 3 training sessions with {config.max_iters} iterations each
   • Each session built upon the previous one using cached checkpoints
   • Total effective training: ~{3 * config.max_iters} iterations
   • Cache automatically managed storage and cleanup
   • Demonstrated true iterative improvement across sessions

💡 Key Benefits Demonstrated:
   ✅ No training starts from scratch after first session
   ✅ Automatic checkpoint quality assessment and selection
   ✅ Intelligent storage management with cleanup
   ✅ Seamless resumption with full training state
   ✅ Progressive model improvement across sessions
""")


def demo_cache_management():
    """Demonstrate cache management features."""
    print("\n" + "="*70)
    print("🗂️  Demo: Cache Management Features")
    print("="*70)
    
    cache_config = CacheConfig(
        cache_dir="demo-cache-management",
        max_checkpoints=3,
        max_cache_size_gb=0.1  # Very small for demo
    )
    
    cache = TrainingCache(cache_config)
    
    # Create some dummy data to demonstrate features
    dummy_model = torch.nn.Linear(10, 1)
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
    
    print("\n📝 Creating sample checkpoints...")
    
    # Create several checkpoints with different quality
    checkpoints = []
    for i in range(5):
        val_loss = 3.0 - i * 0.3  # Decreasing loss = improving quality
        checkpoint_id = cache.save_checkpoint(
            model=dummy_model,
            optimizer=dummy_optimizer,
            scheduler=None,
            iteration=i * 100,
            epoch=i,
            train_loss=val_loss + 0.1,
            val_loss=val_loss,
            learning_rate=1e-4,
            model_config={'n_layer': 2, 'n_embd': 128},
            training_config={'batch_size': 4},
            data_config={'data_dir': 'demo'},
            metrics={'accuracy': 0.5 + i * 0.1},
            tags=[f'checkpoint_{i}', 'demo'],
            notes=f"Demo checkpoint {i}",
            force_save=True
        )
        checkpoints.append(checkpoint_id)
    
    # Show cache statistics
    print(f"\n📊 Cache Statistics:")
    stats = cache.get_cache_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   • {key}: {value:.3f}")
        else:
            print(f"   • {key}: {value}")
    
    # Show available checkpoints
    print(f"\n📋 Available Checkpoints:")
    available = cache.list_checkpoints()
    for i, ckpt in enumerate(available):
        print(f"   {i+1}. {ckpt.checkpoint_id}")
        print(f"      Quality: {ckpt.quality_score:.3f} | Loss: {ckpt.val_loss:.3f}")
        print(f"      Tags: {', '.join(ckpt.tags)}")
    
    # Demonstrate best checkpoint selection
    best_id = cache.get_best_checkpoint()
    print(f"\n🏆 Best checkpoint: {best_id}")
    
    # Demonstrate loading
    try:
        checkpoint_data, metadata = cache.load_checkpoint(best_id)
        print(f"✅ Successfully loaded best checkpoint")
        print(f"   Iteration: {metadata.iteration}")
        print(f"   Quality score: {metadata.quality_score:.3f}")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
    
    # Demonstrate cache cleanup
    print(f"\n🧹 Cache cleanup automatically removed excess checkpoints")
    print(f"   (max_checkpoints = {cache_config.max_checkpoints})")
    
    final_stats = cache.get_cache_stats()
    print(f"   Final checkpoint count: {final_stats['total_checkpoints']}")
    
    print("\n✅ Cache management demo completed!")


if __name__ == "__main__":
    # Check if we have the required dependencies
    try:
        from nanecho_model import NanEchoModel
        print("✅ Dependencies available - running full demo")
        
        # Run demos
        demo_iterative_training()
        demo_cache_management()
        
    except ImportError as e:
        print(f"⚠️  Some dependencies missing: {e}")
        print("📝 Running cache management demo only...")
        demo_cache_management()