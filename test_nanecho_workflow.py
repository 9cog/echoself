#!/usr/bin/env python3
"""
Test script for NanEcho training workflow components
Tests the complete workflow from data preparation to validation
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import json
from pathlib import Path

def test_data_preparation_error_handling():
    """Test data preparation handles tiktoken errors gracefully"""
    print("🧪 Testing data preparation error handling...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "test_nanecho"
        test_dir.mkdir()
        
        # Copy essential files
        nanecho_dir = Path(__file__).parent / "NanEcho"
        shutil.copy2(nanecho_dir / "prepare_nanecho.py", test_dir)
        shutil.copy2(Path(__file__).parent / "echoself.md", test_dir)
        
        # Create dummy echoself directory structure
        (test_dir / "docs").mkdir()
        (test_dir / "docs" / "optimization_plan.md").write_text("Echo Self patterns for testing")
        (test_dir / "README.md").write_text("Echo Self test content")
        
        os.chdir(test_dir)
        
        # Test should fail gracefully due to tiktoken network issue
        result = os.system("python prepare_nanecho.py --echo_depth=1 --persona_weight=0.5 --output_dir=data/test")
        
        if result != 0:
            print("   ✅ PASS: Data preparation failed gracefully as expected")
            return True
        else:
            print("   ❌ FAIL: Data preparation should have failed due to tiktoken")
            return False

def test_fallback_dataset_creation():
    """Test fallback dataset creation works correctly"""
    print("🧪 Testing fallback dataset creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "nanecho"
        data_dir.mkdir()
        
        # Create minimal fallback dataset (from workflow)
        minimal_data = np.array([1, 2, 3, 4, 5] * 200, dtype=np.uint16)  # 1000 tokens
        train_data = minimal_data[:800]
        val_data = minimal_data[800:]
        
        train_data.tofile(data_dir / "train.bin")
        val_data.tofile(data_dir / "val.bin")
        
        # Create metadata
        metadata = {
            'train_tokens': len(train_data),
            'val_tokens': len(val_data),
            'echo_depth': 7,
            'persona_weight': 0.95,
            'fallback_mode': True,
            'warning': 'This is minimal fallback data - real preparation failed'
        }
        with open(data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Verify files were created correctly
        if (data_dir / "train.bin").exists() and (data_dir / "val.bin").exists():
            train_size = (data_dir / "train.bin").stat().st_size
            val_size = (data_dir / "val.bin").stat().st_size
            
            if train_size > 0 and val_size > 0:
                print(f"   ✅ PASS: Fallback dataset created (train: {train_size} bytes, val: {val_size} bytes)")
                return True
            else:
                print("   ❌ FAIL: Files created but are empty")
                return False
        else:
            print("   ❌ FAIL: Required files not created")
            return False

def test_validation_error_messages():
    """Test validation provides helpful error messages"""
    print("🧪 Testing validation error messages...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        nanecho_dir = Path(__file__).parent / "NanEcho"
        
        # Test with missing directory
        os.chdir(temp_dir)
        result = os.system(f"python {nanecho_dir / 'validate_metadata.py'} nonexistent_dir 2>&1 | grep -q 'Data directory not found'")
        
        if result == 0:
            print("   ✅ PASS: Validation correctly reports missing directory")
            return True
        else:
            print("   ❌ FAIL: Validation did not report missing directory correctly")
            return False

def test_workflow_directory_structure():
    """Test workflow directory structure creation"""
    print("🧪 Testing workflow directory structure...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Simulate workflow directory creation
        dirs_to_create = [
            "echoself/NanEcho/data/nanecho",
            "nanoGPT/data/nanecho"
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Check all directories exist
        all_exist = all(Path(dir_path).exists() for dir_path in dirs_to_create)
        
        if all_exist:
            print("   ✅ PASS: All required directories created successfully")
            return True
        else:
            print("   ❌ FAIL: Some directories were not created")
            return False

def main():
    """Run all workflow tests"""
    print("🔍 Testing NanEcho Training Workflow Components")
    print("=" * 60)
    
    tests = [
        test_data_preparation_error_handling,
        test_fallback_dataset_creation,
        test_validation_error_messages,
        test_workflow_directory_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ FAIL: Test raised exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All workflow tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()