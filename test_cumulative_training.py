#!/usr/bin/env python3
"""
Test for cumulative training with caching system.

This test validates that:
1. Connection masks are properly saved and restored
2. Training state is maintained across sessions
3. Connection growth is cumulative, not restarted
4. Model iteration tracking works correctly
"""

import os
import sys
import tempfile
import shutil
import torch
import numpy as np
from pathlib import Path

# Import the models and training components
from nanecho_model import NanEchoModel, NanEchoConfig
from training_cache import TrainingCache, CacheConfig
from train_cached import CachedNanEchoTrainer, TrainingConfig


def create_test_data(data_dir: str, block_size: int = 1024):
    """Create minimal test data for training."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Create simple repetitive data that's larger than block_size
    text_data = "Echo Self cognitive architecture adaptive attention recursive reasoning " * 100
    tokens = np.array([ord(c) % 256 for c in text_data], dtype=np.uint16)
    
    # Ensure we have enough data
    while len(tokens) <= block_size * 2:
        tokens = np.concatenate([tokens, tokens])
    
    # Save training and validation data
    train_tokens = tokens[:len(tokens) // 2]
    val_tokens = tokens[len(tokens) // 2:]
    
    train_tokens.tofile(os.path.join(data_dir, 'train.bin'))
    val_tokens.tofile(os.path.join(data_dir, 'val.bin'))
    
    print(f"✅ Created test data: train={len(train_tokens)}, val={len(val_tokens)}")


def test_connection_mask_state_dict():
    """Test that ConnectionMask properly saves and restores state."""
    print("🧪 Testing ConnectionMask state_dict functionality...")
    
    from nanecho_model import ConnectionMask
    
    # Create a connection mask
    shape = (64, 64)
    mask1 = ConnectionMask(shape, initial_ratio=0.1)
    
    # Grow connections
    mask1.grow_connections(0.2, 1.0)
    original_ratio = mask1.current_ratio
    original_mask = mask1.mask.clone()
    
    # Save state
    state_dict = mask1.state_dict()
    
    # Create new mask and load state
    mask2 = ConnectionMask(shape, initial_ratio=0.05)  # Different initial ratio
    mask2.load_state_dict(state_dict)
    
    # Check that state was restored correctly
    assert abs(mask2.current_ratio - original_ratio) < 1e-6, \
        f"Ratio not restored: {mask2.current_ratio} vs {original_ratio}"
    
    assert torch.allclose(mask2.mask, original_mask), \
        "Mask tensor not restored correctly"
    
    print(f"   ✅ Connection ratio restored: {original_ratio:.3f}")
    print(f"   ✅ Mask tensor restored correctly")
    return True


def test_model_state_preservation():
    """Test that model state including connection masks is preserved."""
    print("\n🧪 Testing model state preservation...")
    
    config = NanEchoConfig(
        vocab_size=256,
        n_embd=128,
        n_head=4,
        n_layer=2,
        block_size=64,
        dropout=0.0,
        initial_connections=0.1,
        connection_growth_rate=0.1,
        max_connections=0.5
    )
    
    # Create model
    model1 = NanEchoModel(config)
    model1.current_iteration = 1000
    
    # Grow connections
    original_ratio = model1.connection_ratio
    model1.grow_connections()
    new_ratio = model1.connection_ratio
    
    print(f"   Connection ratio grown: {original_ratio:.3f} -> {new_ratio:.3f}")
    
    # Save model state
    state_dict = model1.state_dict()
    
    # Create new model and load state
    model2 = NanEchoModel(config)
    model2.load_state_dict(state_dict)
    model2.current_iteration = 1000
    
    # Check that connection ratio was preserved
    assert abs(model2.connection_ratio - new_ratio) < 1e-6, \
        f"Model connection ratio not preserved: {model2.connection_ratio} vs {new_ratio}"
    
    # Check that masks were preserved in individual blocks
    for i, (block1, block2) in enumerate(zip(model1.blocks, model2.blocks)):
        mask1_ratio = block1.attn.connection_mask.current_ratio
        mask2_ratio = block2.attn.connection_mask.current_ratio
        
        assert abs(mask1_ratio - mask2_ratio) < 1e-6, \
            f"Block {i} attention mask ratio not preserved: {mask2_ratio} vs {mask1_ratio}"
        
        assert torch.allclose(block1.attn.connection_mask.mask, block2.attn.connection_mask.mask), \
            f"Block {i} attention mask tensor not preserved"
    
    print(f"   ✅ Model connection ratio preserved: {new_ratio:.3f}")
    print(f"   ✅ All block connection masks preserved")
    return True


def test_training_cache_functionality():
    """Test the training cache system."""
    print("\n🧪 Testing training cache functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "cache")
        
        # Create cache
        cache_config = CacheConfig(
            cache_dir=cache_dir,
            max_checkpoints=3,
            min_improvement_threshold=0.0
        )
        cache = TrainingCache(cache_config)
        
        # Create a simple model and optimizer
        config = NanEchoConfig(
            vocab_size=256,
            n_embd=64,
            n_head=2,
            n_layer=2,
            block_size=32,
            initial_connections=0.2
        )
        
        model = NanEchoModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Grow connections
        model.grow_connections()
        original_ratio = model.connection_ratio
        
        # Save checkpoint
        checkpoint_id = cache.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            iteration=500,
            epoch=1,
            train_loss=2.5,
            val_loss=2.8,
            learning_rate=0.001,
            model_config={'n_layer': config.n_layer, 'n_embd': config.n_embd, 'vocab_size': config.vocab_size},
            training_config={'learning_rate': 0.001},
            data_config={'data_dir': temp_dir},
            metrics={'accuracy': 0.7},
            force_save=True
        )
        
        assert checkpoint_id is not None, "Failed to save checkpoint"
        print(f"   ✅ Checkpoint saved: {checkpoint_id}")
        
        # Create new model and load checkpoint
        model2 = NanEchoModel(config)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        
        checkpoint_data, metadata = cache.load_checkpoint(
            checkpoint_id=checkpoint_id,
            model=model2,
            optimizer=optimizer2
        )
        
        # Verify restoration
        assert abs(model2.connection_ratio - original_ratio) < 1e-6, \
            f"Connection ratio not restored: {model2.connection_ratio} vs {original_ratio}"
        
        assert metadata.iteration == 500, f"Iteration not restored: {metadata.iteration}"
        assert abs(metadata.val_loss - 2.8) < 1e-6, f"Loss not restored: {metadata.val_loss}"
        
        print(f"   ✅ Checkpoint loaded successfully")
        print(f"   ✅ Connection ratio preserved: {original_ratio:.3f}")
        
        return True


def test_cumulative_training_session():
    """Test a complete cumulative training scenario."""
    print("\n🧪 Testing cumulative training session...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, "data")
        out_dir = os.path.join(temp_dir, "out")
        cache_dir = os.path.join(out_dir, "cache")
        
        # Create test data
        create_test_data(data_dir, block_size=64)
        
        # Create training configuration
        config = TrainingConfig(
            data_dir=data_dir,
            out_dir=out_dir,
            vocab_size=256,
            n_embd=64,
            n_head=2,
            n_layer=2,
            block_size=64,
            batch_size=2,
            max_iters=20,  # Very short training
            eval_interval=5,
            checkpoint_interval=10,
            connection_growth_interval=5,  # Grow connections frequently
            initial_connections=0.2,
            connection_growth_rate=0.1,
            max_connections=0.8,
            device='cpu'
        )
        
        cache_config = CacheConfig(
            cache_dir=cache_dir,
            max_checkpoints=5,
            min_improvement_threshold=0.0
        )
        
        print("   Starting first training session...")
        
        # First training session
        trainer1 = CachedNanEchoTrainer(
            config=config,
            cache_config=cache_config,
            force_fresh_start=True
        )
        
        # Check initial state
        initial_ratio = trainer1.model.connection_ratio
        print(f"   Initial connection ratio: {initial_ratio:.3f}")
        
        # Run short training
        trainer1.train()
        
        final_ratio_1 = trainer1.model.connection_ratio
        final_iteration_1 = trainer1.iteration
        
        print(f"   Session 1 completed:")
        print(f"     Final connection ratio: {final_ratio_1:.3f}")
        print(f"     Final iteration: {final_iteration_1}")
        
        # Second training session - should resume
        print("\n   Starting second training session (should resume)...")
        
        config.max_iters = 40  # Continue training for more iterations
        
        trainer2 = CachedNanEchoTrainer(
            config=config,
            cache_config=cache_config,
            force_fresh_start=False  # Should resume
        )
        
        # Check if it resumed
        assert trainer2.resumed_from_checkpoint, "Training should have resumed from checkpoint"
        
        starting_ratio_2 = trainer2.model.connection_ratio
        starting_iteration_2 = trainer2.starting_iteration
        
        print(f"   Session 2 resumed:")
        print(f"     Starting connection ratio: {starting_ratio_2:.3f}")
        print(f"     Starting iteration: {starting_iteration_2}")
        
        # Verify cumulative behavior
        assert abs(starting_ratio_2 - final_ratio_1) < 0.01, \
            f"Connection ratio not preserved across sessions: {starting_ratio_2} vs {final_ratio_1}"
        
        assert starting_iteration_2 >= final_iteration_1, \
            f"Starting iteration should be >= final iteration from session 1: {starting_iteration_2} vs {final_iteration_1}"
        
        # Run second session
        trainer2.train()
        
        final_ratio_2 = trainer2.model.connection_ratio
        final_iteration_2 = trainer2.iteration
        
        print(f"   Session 2 completed:")
        print(f"     Final connection ratio: {final_ratio_2:.3f}")
        print(f"     Final iteration: {final_iteration_2}")
        
        # Verify cumulative progress
        assert final_ratio_2 >= starting_ratio_2, \
            f"Connections should continue growing: {final_ratio_2} vs {starting_ratio_2}"
        
        assert final_iteration_2 > starting_iteration_2, \
            f"Iterations should continue from where left off: {final_iteration_2} vs {starting_iteration_2}"
        
        print("   ✅ Cumulative training behavior verified!")
        return True


def main():
    """Run all cumulative training tests."""
    print("🔍 Testing Cumulative Training with Caching")
    print("=" * 60)
    
    tests = [
        test_connection_mask_state_dict,
        test_model_state_preservation,
        test_training_cache_functionality,
        test_cumulative_training_session
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All cumulative training tests passed!")
        print("\n🎯 Summary of fixes:")
        print("   1. ✅ ConnectionMask now inherits from nn.Module for proper state saving")
        print("   2. ✅ Connection masks and ratios are preserved in state_dict")
        print("   3. ✅ Training cache saves and restores complete model state")
        print("   4. ✅ Connection growth is cumulative across sessions")
        print("   5. ✅ Training iterations continue from checkpoint")
        print("\n🚀 Cumulative training is now working correctly!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)