#!/usr/bin/env python3
"""
Test script to verify the data size validation fix.

This script tests the scenario that was causing the original error:
- Creates a small dataset (smaller than block_size)
- Tests the enhanced get_batch function 
- Verifies that proper error messages are shown
"""

import os
import sys
import tempfile
import numpy as np
import torch

# Import our enhanced training functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import get_batch, validate_dataset_size

def create_test_dataset(size, data_dir):
    """Create a small test dataset for validation testing."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Create small datasets
    train_data = np.random.randint(0, 1000, size=size, dtype=np.uint16)
    val_data = np.random.randint(0, 1000, size=size, dtype=np.uint16)
    
    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')
    
    train_data.tofile(train_path)
    val_data.tofile(val_path)
    
    return train_path, val_path

def test_get_batch_validation():
    """Test the get_batch function with insufficient data."""
    print("🧪 Testing get_batch function validation...")
    
    # Test case 1: Data smaller than block_size (should fail)
    small_data = np.array([1, 2, 3, 4, 5], dtype=np.uint16)  # 5 tokens
    block_size = 10  # Larger than data
    batch_size = 2
    
    print(f"   Test 1: len(data)={len(small_data)}, block_size={block_size}")
    try:
        x, y = get_batch('train', small_data, block_size, batch_size, 'cpu', 'cpu')
        print("   ❌ FAIL: Expected ValueError but got success")
        return False
    except ValueError as e:
        print(f"   ✅ PASS: Correctly caught ValueError: {str(e)[:100]}...")
    
    # Test case 2: Data larger than block_size (should succeed)
    large_data = np.random.randint(0, 1000, size=50, dtype=np.uint16)  # 50 tokens
    block_size = 10  # Smaller than data
    
    print(f"   Test 2: len(data)={len(large_data)}, block_size={block_size}")
    try:
        x, y = get_batch('train', large_data, block_size, batch_size, 'cpu', 'cpu')
        print(f"   ✅ PASS: Successfully created batch with shapes x={x.shape}, y={y.shape}")
        return True
    except Exception as e:
        print(f"   ❌ FAIL: Unexpected error: {e}")
        return False

def test_dataset_validation():
    """Test the dataset validation function."""
    print("\n🧪 Testing dataset size validation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test case 1: Small datasets (should fail)
        print("   Test 1: Small datasets")
        create_test_dataset(size=100, data_dir=temp_dir)  # 100 tokens
        
        try:
            train_data, val_data = validate_dataset_size(temp_dir, block_size=1024)
            print("   ❌ FAIL: Expected ValueError but validation passed")
            return False
        except ValueError as e:
            print(f"   ✅ PASS: Correctly caught ValueError: {str(e)[:100]}...")
        
        # Test case 2: Large enough datasets (should succeed)
        print("   Test 2: Sufficient datasets")
        create_test_dataset(size=2000, data_dir=temp_dir)  # 2000 tokens
        
        try:
            train_data, val_data = validate_dataset_size(temp_dir, block_size=1024)
            print(f"   ✅ PASS: Validation successful with train={len(train_data)}, val={len(val_data)}")
            return True
        except Exception as e:
            print(f"   ❌ FAIL: Unexpected error: {e}")
            return False

def test_original_error_scenario():
    """Test the exact scenario from the original error."""
    print("\n🧪 Testing original error scenario...")
    
    # Simulate the original problem: len(data) - block_size < 0
    small_data = np.array(list(range(300)), dtype=np.uint16)  # 300 tokens
    block_size = 390  # Larger than data, would cause len(data) - block_size = -90
    
    print(f"   Original error scenario: len(data)={len(small_data)}, block_size={block_size}")
    print(f"   This would cause: len(data) - block_size = {len(small_data) - block_size}")
    
    try:
        # This would have caused: RuntimeError: random_ expects 'from' to be less than 'to', but got from=0 >= to=-90
        ix = torch.randint(len(small_data) - block_size, (4,))
        print("   ❌ FAIL: torch.randint should have failed but didn't")
        return False
    except RuntimeError as e:
        print(f"   ✅ Original error reproduced: {e}")
        
        # Now test our fix
        try:
            x, y = get_batch('train', small_data, block_size, 4, 'cpu', 'cpu')
            print("   ❌ FAIL: Our fix should have prevented this")
            return False
        except ValueError as e:
            print(f"   ✅ PASS: Our fix correctly prevents the error: {str(e)[:100]}...")
            return True

def main():
    """Run all validation tests."""
    print("🔍 Testing Data Size Validation Fix")
    print("="*50)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_get_batch_validation():
        tests_passed += 1
    
    if test_dataset_validation():
        tests_passed += 1
        
    if test_original_error_scenario():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! The data size validation fix is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())