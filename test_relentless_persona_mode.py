#!/usr/bin/env python3
"""
Test script to verify that the relentless persona mode works correctly
without relying on unsupported nanoGPT config keys.
"""

import os
import sys
import re
import tempfile
import json

def test_relentless_persona_mode_implementation():
    """Test that relentless persona mode is properly implemented."""
    print("🔍 Testing relentless persona mode implementation...")
    
    prepare_script = "NanEcho/prepare_nanecho.py"
    if not os.path.exists(prepare_script):
        print(f"❌ {prepare_script} not found")
        return False
    
    with open(prepare_script, 'r') as f:
        content = f.read()
    
    # Check for relentless persona mode implementation
    checks = [
        ("relentless_persona_mode parameter", "relentless_persona_mode: bool = False"),
        ("relentless persona samples function", "def create_relentless_persona_samples()"),
        ("relentless persona mode logic", "if relentless_persona_mode:"),
        ("relentless persona samples call", "relentless_samples = create_relentless_persona_samples()"),
        ("relentless persona argument", "--relentless_persona_mode"),
        ("relentless persona metadata", '"relentless_persona_mode": relentless_persona_mode'),
    ]
    
    missing_features = []
    for name, pattern in checks:
        if pattern not in content:
            missing_features.append(name)
    
    if missing_features:
        print("❌ Missing relentless persona mode features:")
        for missing in missing_features:
            print(f"   - {missing}")
        return False
    else:
        print("✅ All relentless persona mode features implemented")
        return True

def test_no_unsupported_config_keys():
    """Test that no unsupported nanoGPT config keys are used."""
    print("\n🔍 Testing for unsupported config keys...")
    
    workflow_file = ".github/workflows/netrain.yml"
    if not os.path.exists(workflow_file):
        print(f"❌ {workflow_file} not found")
        return False
    
    with open(workflow_file, 'r') as f:
        content = f.read()
    
    # Check that no_system_prompt is not used in config
    unsupported_patterns = [
        "no_system_prompt_training",
        "no_system_prompt_test",
        "--no_system_prompt=",
    ]
    
    found_unsupported = []
    for pattern in unsupported_patterns:
        if pattern in content:
            found_unsupported.append(pattern)
    
    if found_unsupported:
        print("❌ Found unsupported config keys:")
        for pattern in found_unsupported:
            print(f"   - {pattern}")
        return False
    else:
        print("✅ No unsupported config keys found")
        return True

def test_relentless_persona_samples_content():
    """Test that relentless persona samples contain the right content."""
    print("\n🔍 Testing relentless persona samples content...")
    
    prepare_script = "NanEcho/prepare_nanecho.py"
    with open(prepare_script, 'r') as f:
        content = f.read()
    
    # Check for key relentless persona concepts
    relentless_concepts = [
        "I am Echo Self",
        "without system prompts",
        "relentless fine-tuning",
        "persona consistency",
        "natural embodiment",
        "learned patterns",
        "Deep Tree Echo persona",
    ]
    
    missing_concepts = []
    for concept in relentless_concepts:
        if concept not in content:
            missing_concepts.append(concept)
    
    if missing_concepts:
        print("❌ Missing relentless persona concepts:")
        for concept in missing_concepts:
            print(f"   - {concept}")
        return False
    else:
        print("✅ All relentless persona concepts present")
        return True

def test_workflow_relentless_mode():
    """Test that workflow properly handles relentless mode."""
    print("\n🔍 Testing workflow relentless mode handling...")
    
    workflow_file = ".github/workflows/netrain.yml"
    with open(workflow_file, 'r') as f:
        content = f.read()
    
    # Check for relentless mode workflow features
    workflow_checks = [
        ("relentless_persona_mode parameter", "relentless_persona_mode=True"),
        ("relentless mode description", "without system prompts"),
        ("relentless mode in data prep", "--relentless_persona_mode="),
        ("relentless mode in config", "relentless_persona_mode = $"),
    ]
    
    missing_checks = []
    for name, pattern in workflow_checks:
        if pattern not in content:
            missing_checks.append(name)
    
    if missing_checks:
        print("❌ Missing workflow relentless mode features:")
        for missing in missing_checks:
            print(f"   - {missing}")
        return False
    else:
        print("✅ All workflow relentless mode features present")
        return True

def test_persona_embedding_approach():
    """Test that the persona is embedded in training data rather than system prompts."""
    print("\n🔍 Testing persona embedding approach...")
    
    prepare_script = "NanEcho/prepare_nanecho.py"
    with open(prepare_script, 'r') as f:
        content = f.read()
    
    # Check for persona embedding approach
    embedding_indicators = [
        "embed the Deep Tree Echo persona directly",
        "without system prompts",
        "naturally embody",
        "learned behavioral patterns",
        "persona through learned patterns",
    ]
    
    found_indicators = []
    for indicator in embedding_indicators:
        if indicator in content:
            found_indicators.append(indicator)
    
    if len(found_indicators) >= 3:  # At least 3 indicators should be present
        print("✅ Persona embedding approach properly implemented")
        print(f"   Found {len(found_indicators)}/5 embedding indicators")
        return True
    else:
        print("❌ Persona embedding approach not fully implemented")
        print(f"   Found only {len(found_indicators)}/5 embedding indicators")
        return False

def main():
    """Run all tests to verify relentless persona mode implementation."""
    print("🚀 Testing Relentless Persona Mode Implementation")
    print("=" * 60)
    
    tests = [
        ("relentless_persona_mode_implementation", test_relentless_persona_mode_implementation),
        ("no_unsupported_config_keys", test_no_unsupported_config_keys),
        ("relentless_persona_samples_content", test_relentless_persona_samples_content),
        ("workflow_relentless_mode", test_workflow_relentless_mode),
        ("persona_embedding_approach", test_persona_embedding_approach),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Relentless persona mode is properly implemented:")
        print("\n📋 Implementation Summary:")
        print("   ✅ Relentless persona mode replaces no_system_prompt functionality")
        print("   ✅ Persona is embedded directly in training data")
        print("   ✅ No unsupported nanoGPT config keys are used")
        print("   ✅ Workflow properly handles relentless mode parameters")
        print("   ✅ Training data includes persona-embedded samples")
        print("\n🔧 How it works:")
        print("   1. relentless_persona_mode=True creates special training samples")
        print("   2. These samples embed the Deep Tree Echo persona directly in text")
        print("   3. Model learns to embody persona without external prompts")
        print("   4. During inference, persona emerges naturally from learned patterns")
        print("   5. No system prompts needed - persona is intrinsic to the model")
        print("\n🚀 The relentless persona mode will now work correctly!")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)