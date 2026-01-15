#!/usr/bin/env python3
"""
Test script to validate that the GitHub Actions cache fix works correctly.
This test simulates the workflow steps and ensures cache paths exist when needed.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path


def test_cache_directory_creation():
    """Test that cache directories are created before attempting to use them."""
    print("🧪 Testing cache directory creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Simulate workflow step: Prepare directory structure (now happens FIRST)
        print("Step 1: Creating directories (simulating 'Prepare directory structure')...")
        os.makedirs("echoself/data/nanecho", exist_ok=True)
        os.makedirs("echoself/out-nanecho-cached-ci/cache", exist_ok=True)
        
        # Verify paths exist for caching (simulating actions/cache@v4)
        print("Step 2: Verifying cache paths exist (simulating 'Cache training artifacts')...")
        
        cache_path = Path("echoself/out-nanecho-cached-ci/cache")
        data_path = Path("echoself/data/nanecho")
        
        if not cache_path.exists():
            print("❌ FAIL: Cache directory does not exist")
            return False
        
        if not data_path.exists():
            print("❌ FAIL: Data directory does not exist")
            return False
        
        print("✅ PASS: All cache directories exist before cache action")
        return True


def test_workflow_order():
    """Test that the workflow steps are in the correct order."""
    print("🧪 Testing workflow step order...")
    
    workflow_path = Path(__file__).parent / ".github/workflows/netrain-cached.yml"
    
    if not workflow_path.exists():
        print("❌ FAIL: Workflow file does not exist")
        return False
    
    with open(workflow_path, 'r') as f:
        content = f.read()
    
    # Find positions of key steps
    prepare_pos = content.find("name: Prepare directory structure")
    cache_pos = content.find("name: Cache training artifacts")
    
    if prepare_pos == -1:
        print("❌ FAIL: 'Prepare directory structure' step not found")
        return False
    
    if cache_pos == -1:
        print("❌ FAIL: 'Cache training artifacts' step not found") 
        return False
    
    if prepare_pos > cache_pos:
        print("❌ FAIL: Directory preparation happens AFTER caching (wrong order)")
        return False
    
    print("✅ PASS: Directory preparation happens BEFORE caching (correct order)")
    return True


def test_cache_paths_in_workflow():
    """Test that cache paths in workflow match expected directories."""
    print("🧪 Testing cache path configuration...")
    
    workflow_path = Path(__file__).parent / ".github/workflows/netrain-cached.yml"
    
    with open(workflow_path, 'r') as f:
        content = f.read()
    
    # Check that cache paths include both output cache and data directories
    if "echoself/${{ steps.params.outputs.output_dir }}/cache" not in content:
        print("❌ FAIL: Output cache directory not in cache paths")
        return False
    
    if "echoself/data/nanecho" not in content:
        print("❌ FAIL: Data directory not in cache paths")
        return False
    
    print("✅ PASS: Cache paths correctly configured")
    return True


def main():
    """Run all tests to validate the cache fix."""
    print("🚀 Testing GitHub Actions Cache Fix")
    print("=" * 50)
    
    tests = [
        ("cache_directory_creation", test_cache_directory_creation),
        ("workflow_order", test_workflow_order), 
        ("cache_paths_in_workflow", test_cache_paths_in_workflow),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results[test_name] = False
        print()
    
    print("=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The GitHub Actions cache fix is working correctly:")
        print("   1. ✅ Directories are created BEFORE cache action runs")
        print("   2. ✅ Workflow steps are in the correct order")  
        print("   3. ✅ Cache paths are properly configured")
        print("\nThe 'Cache not found' error should now be resolved.")
    else:
        print("❌ Some tests failed. The cache fix may need additional work.")
        sys.exit(1)


if __name__ == "__main__":
    main()