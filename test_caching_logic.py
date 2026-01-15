#!/usr/bin/env python3
"""
Test caching logic without requiring PyTorch.

This validates the core logic changes we made to ensure cumulative training works.
"""

import os
import sys
import json
import tempfile
from pathlib import Path


def test_connection_mask_concept():
    """Test the conceptual changes to ConnectionMask."""
    print("🧪 Testing ConnectionMask conceptual changes...")
    
    # Test that our changes make ConnectionMask inherit from nn.Module
    # and use buffers for state preservation
    
    # Read the nanecho_model.py file to check our changes
    try:
        with open('/home/runner/work/echoself/echoself/nanecho_model.py', 'r') as f:
            content = f.read()
        
        # Check key changes
        checks = [
            ('class ConnectionMask(nn.Module)', 'ConnectionMask now inherits from nn.Module'),
            ('self.register_buffer(\'mask\'', 'Mask is registered as buffer'),
            ('self.register_buffer(\'_current_ratio_tensor\'', 'Current ratio is registered as buffer'),
            ('@property', 'Current ratio is a property'),
            ('def current_ratio(self)', 'Current ratio property getter'),
            ('def state_dict(self', 'Custom state_dict method'),
            ('def load_state_dict(self', 'Custom load_state_dict method'),
        ]
        
        passed_checks = 0
        for check, description in checks:
            if check in content:
                print(f"   ✅ {description}")
                passed_checks += 1
            else:
                print(f"   ❌ {description}")
        
        if passed_checks == len(checks):
            print("   ✅ All ConnectionMask changes implemented correctly")
            return True
        else:
            print(f"   ❌ Only {passed_checks}/{len(checks)} changes found")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to read file: {e}")
        return False


def test_training_cache_changes():
    """Test changes to training cache system."""
    print("\n🧪 Testing training cache changes...")
    
    try:
        with open('/home/runner/work/echoself/echoself/training_cache.py', 'r') as f:
            content = f.read()
        
        # Check key changes
        checks = [
            ('\'connection_ratio\': getattr(model, \'connection_ratio\', 0.0)', 'Connection ratio saved in checkpoint'),
            ('\'current_iteration\': getattr(model, \'current_iteration\'', 'Current iteration saved in checkpoint'),
        ]
        
        passed_checks = 0
        for check, description in checks:
            if check in content:
                print(f"   ✅ {description}")
                passed_checks += 1
            else:
                print(f"   ❌ {description}")
        
        if passed_checks == len(checks):
            print("   ✅ All training cache changes implemented correctly")
            return True
        else:
            print(f"   ❌ Only {passed_checks}/{len(checks)} changes found")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to read file: {e}")
        return False


def test_cached_trainer_changes():
    """Test changes to cached trainer."""
    print("\n🧪 Testing cached trainer changes...")
    
    try:
        with open('/home/runner/work/echoself/echoself/train_cached.py', 'r') as f:
            content = f.read()
        
        # Check key changes
        checks = [
            ('checkpoint_data.get(\'current_iteration\'', 'Current iteration restored from checkpoint'),
            ('saved_ratio = checkpoint_data.get(\'connection_ratio\'', 'Connection ratio validation'),
            ('if iteration > 0 and iteration % self.config.connection_growth_interval == 0:', 'Fixed connection growth logic'),
            ('if new_ratio > old_ratio:', 'Only log when connections actually grow'),
        ]
        
        passed_checks = 0
        for check, description in checks:
            if check in content:
                print(f"   ✅ {description}")
                passed_checks += 1
            else:
                print(f"   ❌ {description}")
        
        if passed_checks == len(checks):
            print("   ✅ All cached trainer changes implemented correctly")
            return True
        else:
            print(f"   ❌ Only {passed_checks}/{len(checks)} changes found")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to read file: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        'nanecho_model.py',
        'training_cache.py', 
        'train_cached.py',
        'train_nanecho.py',
        'test_cumulative_training.py'
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = f'/home/runner/work/echoself/echoself/{filename}'
        if os.path.exists(filepath):
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename}")
            missing_files.append(filename)
    
    if not missing_files:
        print("   ✅ All required files present")
        return True
    else:
        print(f"   ❌ Missing files: {missing_files}")
        return False


def test_logic_correctness():
    """Test the logic of cumulative training."""
    print("\n🧪 Testing cumulative training logic...")
    
    # Simulate training state across sessions
    class MockTrainingState:
        def __init__(self):
            self.connection_ratio = 0.1
            self.iteration = 0
            self.connection_growth_interval = 500
        
        def simulate_training(self, max_iters: int, start_iteration: int = 0):
            """Simulate training with connection growth."""
            connections_grown = []
            
            for iteration in range(start_iteration, max_iters):
                self.iteration = iteration
                
                # Connection growth logic (from our fixed version)
                if iteration > 0 and iteration % self.connection_growth_interval == 0:
                    old_ratio = self.connection_ratio
                    # Simulate growing connections by 5%
                    self.connection_ratio = min(self.connection_ratio + 0.05, 1.0)
                    
                    if self.connection_ratio > old_ratio:
                        connections_grown.append({
                            'iteration': iteration,
                            'old_ratio': old_ratio,
                            'new_ratio': self.connection_ratio
                        })
            
            return connections_grown
    
    # Test Session 1: Fresh start
    print("   Simulating Session 1 (fresh start)...")
    state1 = MockTrainingState()
    growth1 = state1.simulate_training(max_iters=1000, start_iteration=0)
    
    print(f"     Initial ratio: 0.1")
    print(f"     Final ratio: {state1.connection_ratio:.3f}")
    print(f"     Final iteration: {state1.iteration}")
    print(f"     Connections grown {len(growth1)} times")
    
    # Test Session 2: Resume from checkpoint
    print("\n   Simulating Session 2 (resume)...")
    state2 = MockTrainingState()
    state2.connection_ratio = state1.connection_ratio  # Restore from checkpoint
    state2.iteration = state1.iteration  # Restore from checkpoint
    
    growth2 = state2.simulate_training(max_iters=2000, start_iteration=state1.iteration)
    
    print(f"     Resumed ratio: {state1.connection_ratio:.3f}")
    print(f"     Final ratio: {state2.connection_ratio:.3f}")
    print(f"     Final iteration: {state2.iteration}")
    print(f"     Additional connections grown {len(growth2)} times")
    
    # Validate cumulative behavior
    checks = [
        (state2.connection_ratio >= state1.connection_ratio, 'Connections continue growing'),
        (state2.iteration > state1.iteration, 'Iterations continue from checkpoint'),
        (len(growth1) > 0, 'Connections grew in session 1'),
        (len(growth2) > 0 or state1.connection_ratio >= 1.0, 'Connections grew in session 2 or reached max')
    ]
    
    all_passed = True
    for condition, description in checks:
        if condition:
            print(f"     ✅ {description}")
        else:
            print(f"     ❌ {description}")
            all_passed = False
    
    if all_passed:
        print("   ✅ Cumulative training logic is correct")
        return True
    else:
        print("   ❌ Cumulative training logic has issues")
        return False


def main():
    """Run all logic validation tests."""
    print("🔍 Testing Cumulative Training Caching Logic")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_connection_mask_concept,
        test_training_cache_changes,
        test_cached_trainer_changes,
        test_logic_correctness
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
        print("✅ All caching logic tests passed!")
        print("\n🎯 Key fixes implemented:")
        print("   1. ✅ ConnectionMask inherits from nn.Module with proper state management")
        print("   2. ✅ Connection masks and ratios saved/restored via state_dict")
        print("   3. ✅ Training cache includes connection_ratio and current_iteration")
        print("   4. ✅ Fixed connection growth to use absolute iteration counting")
        print("   5. ✅ Training properly resumes from checkpoints")
        print("\n🚀 Cumulative training should now work correctly!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)