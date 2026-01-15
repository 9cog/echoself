#!/usr/bin/env python3
"""
Test script for training progress persistence in netrain-cached workflow.
Verifies that the workflow correctly sets up persistent storage and git operations.
"""

import os
import sys
import yaml
import json
from pathlib import Path


def test_workflow_structure():
    """Test that the workflow YAML has the required structure for persistence."""
    print("🧪 Testing workflow structure...")
    
    workflow_path = Path(__file__).parent / ".github/workflows/netrain-cached.yml"
    
    if not workflow_path.exists():
        print("   ❌ FAIL: Workflow file not found")
        return False
    
    with open(workflow_path) as f:
        workflow = yaml.safe_load(f)
    
    # Check permissions
    permissions = workflow.get('jobs', {}).get('train', {}).get('permissions', {})
    if permissions.get('contents') != 'write':
        print("   ❌ FAIL: Missing 'contents: write' permission")
        return False
    print("   ✅ Permissions include 'contents: write'")
    
    # Check for git configuration step
    steps = workflow['jobs']['train']['steps']
    step_names = [step['name'] for step in steps]
    
    if 'Configure Git' not in step_names:
        print("   ❌ FAIL: Missing 'Configure Git' step")
        return False
    print("   ✅ 'Configure Git' step found")
    
    # Check for commit and push step
    if 'Commit and push training progress' not in step_names:
        print("   ❌ FAIL: Missing 'Commit and push training progress' step")
        return False
    print("   ✅ 'Commit and push training progress' step found")
    
    # Verify checkout has token configured
    checkout_step = None
    for step in steps:
        if step['name'] == 'Checkout echoself repository':
            checkout_step = step
            break
    
    if checkout_step and 'with' in checkout_step:
        if 'token' in checkout_step['with']:
            print("   ✅ Checkout step includes token configuration")
        else:
            print("   ⚠️  WARNING: Checkout step may not have token configured")
    
    print("   ✅ PASS: Workflow structure is correct")
    return True


def test_training_progress_directory():
    """Test that .training-progress directory exists with proper structure."""
    print("\n🧪 Testing .training-progress directory...")
    
    progress_dir = Path(__file__).parent / ".training-progress"
    
    if not progress_dir.exists():
        print("   ❌ FAIL: .training-progress directory not found")
        return False
    print("   ✅ .training-progress directory exists")
    
    # Check for README
    readme = progress_dir / "README.md"
    if not readme.exists():
        print("   ❌ FAIL: README.md not found")
        return False
    print("   ✅ README.md exists")
    
    # Check for .gitignore
    gitignore = progress_dir / ".gitignore"
    if not gitignore.exists():
        print("   ❌ FAIL: .gitignore not found")
        return False
    
    # Verify .gitignore excludes large files but keeps metadata
    gitignore_content = gitignore.read_text()
    if '*.pt' in gitignore_content and '!**/metadata.json' in gitignore_content:
        print("   ✅ .gitignore properly configured")
    else:
        print("   ❌ FAIL: .gitignore not properly configured")
        return False
    
    print("   ✅ PASS: Directory structure is correct")
    return True


def test_output_directory_configuration():
    """Test that workflow uses .training-progress for output directories."""
    print("\n🧪 Testing output directory configuration...")
    
    workflow_path = Path(__file__).parent / ".github/workflows/netrain-cached.yml"
    
    with open(workflow_path) as f:
        content = f.read()
    
    # Check that output directories use .training-progress
    if '.training-progress/nanecho-cached-ci' not in content:
        print("   ❌ FAIL: CI output directory not using .training-progress")
        return False
    print("   ✅ CI output directory uses .training-progress")
    
    if '.training-progress/nanecho-cached-scheduled' not in content:
        print("   ❌ FAIL: Scheduled output directory not using .training-progress")
        return False
    print("   ✅ Scheduled output directory uses .training-progress")
    
    if '.training-progress/nanecho-cached-full' not in content:
        print("   ❌ FAIL: Full output directory not using .training-progress")
        return False
    print("   ✅ Full output directory uses .training-progress")
    
    # Ensure old out-* directories are not referenced
    if 'out-nanecho-cached-ci' in content:
        print("   ⚠️  WARNING: Old 'out-' directory references still present")
    
    print("   ✅ PASS: Output directories correctly configured")
    return True


def test_cache_configuration():
    """Test that GitHub Actions cache is properly configured."""
    print("\n🧪 Testing GitHub Actions cache configuration...")
    
    workflow_path = Path(__file__).parent / ".github/workflows/netrain-cached.yml"
    
    with open(workflow_path) as f:
        workflow = yaml.safe_load(f)
    
    steps = workflow['jobs']['train']['steps']
    
    # Find cache step
    cache_step = None
    for step in steps:
        if step['name'] == 'Cache training artifacts':
            cache_step = step
            break
    
    if not cache_step:
        print("   ❌ FAIL: Cache step not found")
        return False
    
    # Check cache uses actions/cache@v4
    if 'uses' not in cache_step or 'actions/cache@v4' not in cache_step['uses']:
        print("   ❌ FAIL: Not using actions/cache@v4")
        return False
    print("   ✅ Using actions/cache@v4")
    
    # Check cache paths
    cache_paths = cache_step['with']['path']
    if isinstance(cache_paths, str):
        cache_paths = [cache_paths]
    
    has_cache_dir = any('.training-progress' in path or 'cache' in path for path in cache_paths)
    if not has_cache_dir:
        print("   ❌ FAIL: Cache paths don't include cache directory")
        return False
    print("   ✅ Cache paths include cache directory")
    
    # Check for restore-keys
    if 'restore-keys' in cache_step['with']:
        print("   ✅ Cache has restore-keys for fallback")
    else:
        print("   ⚠️  WARNING: Cache doesn't have restore-keys")
    
    print("   ✅ PASS: Cache configuration is correct")
    return True


def test_documentation_exists():
    """Test that documentation has been updated."""
    print("\n🧪 Testing documentation...")
    
    # Check CACHING_SYSTEM_README.md
    caching_readme = Path(__file__).parent / "CACHING_SYSTEM_README.md"
    if not caching_readme.exists():
        print("   ❌ FAIL: CACHING_SYSTEM_README.md not found")
        return False
    
    content = caching_readme.read_text()
    if 'Git Repository Commits' in content or 'Hybrid Caching' in content:
        print("   ✅ CACHING_SYSTEM_README.md updated with new information")
    else:
        print("   ⚠️  WARNING: CACHING_SYSTEM_README.md may need updates")
    
    # Check implementation summary
    impl_summary = Path(__file__).parent / ".training-progress/IMPLEMENTATION_SUMMARY.md"
    if impl_summary.exists():
        print("   ✅ IMPLEMENTATION_SUMMARY.md exists")
    else:
        print("   ⚠️  INFO: IMPLEMENTATION_SUMMARY.md not found (optional)")
    
    print("   ✅ PASS: Documentation is present")
    return True


def test_gitignore_main():
    """Test that main .gitignore still excludes out-* directories."""
    print("\n🧪 Testing main .gitignore configuration...")
    
    gitignore = Path(__file__).parent / ".gitignore"
    if not gitignore.exists():
        print("   ❌ FAIL: .gitignore not found")
        return False
    
    content = gitignore.read_text()
    
    # Should still exclude old out-* directories
    if 'out-*/' in content:
        print("   ✅ Old out-* directories still excluded")
    
    # Should NOT exclude .training-progress
    if '.training-progress' in content:
        print("   ⚠️  WARNING: .training-progress is excluded in main .gitignore!")
        print("   This would prevent committing training progress.")
        return False
    else:
        print("   ✅ .training-progress is NOT excluded in main .gitignore")
    
    print("   ✅ PASS: Main .gitignore is correctly configured")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Training Progress Persistence - Test Suite")
    print("=" * 70)
    
    tests = [
        test_workflow_structure,
        test_training_progress_directory,
        test_output_directory_configuration,
        test_cache_configuration,
        test_documentation_exists,
        test_gitignore_main,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
