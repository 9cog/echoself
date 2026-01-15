#!/usr/bin/env python3
"""
Test script to verify that the no_system_prompt flag is correctly set and used.

This script checks:
1. The flag is correctly parsed in prepare_nanecho.py
2. The flag triggers relentless persona mode
3. The flag is passed through the workflow correctly
4. The flag is used in evaluation scripts
"""

import os
import sys
import subprocess
import json

def test_prepare_nanecho_flag():
    """Test that prepare_nanecho.py correctly handles the no_system_prompt flag."""
    print("\n🔍 Testing prepare_nanecho.py no_system_prompt flag handling...")
    
    # Read the prepare_nanecho.py file
    with open('NanEcho/prepare_nanecho.py', 'r') as f:
        content = f.read()
    
    # Check that the flag is in the argument parser
    if '--no_system_prompt' not in content:
        print("❌ --no_system_prompt argument not found in prepare_nanecho.py")
        return False
    
    # Check that the flag is parsed correctly
    if "no_system_prompt = args.no_system_prompt.lower() in ('true', '1', 'yes', 'on')" not in content:
        print("❌ no_system_prompt flag not parsed correctly")
        return False
    
    # Check that the flag triggers relentless persona mode
    if "if relentless_persona_mode or no_system_prompt:" not in content:
        print("❌ no_system_prompt doesn't trigger relentless persona mode")
        return False
    
    print("✅ prepare_nanecho.py correctly handles no_system_prompt flag")
    return True

def test_workflow_flag():
    """Test that the workflow correctly sets and uses the no_system_prompt flag."""
    print("\n🔍 Testing workflow no_system_prompt flag usage...")
    
    # Read the workflow file
    with open('.github/workflows/netrain.yml', 'r') as f:
        content = f.read()
    
    # Check that the flag is set in parameters
    if 'echo "no_system_prompt=True" >> $GITHUB_OUTPUT' not in content:
        print("❌ no_system_prompt not set to True in relentless mode")
        return False
    
    if 'echo "no_system_prompt=False" >> $GITHUB_OUTPUT' not in content:
        print("❌ no_system_prompt not set to False in standard mode")
        return False
    
    # Check that the flag is passed to prepare_nanecho.py
    if '--no_system_prompt=${{ steps.params.outputs.no_system_prompt }}' not in content:
        print("❌ no_system_prompt not passed to prepare_nanecho.py")
        return False
    
    # Check that the flag is used in training config
    if 'no_system_prompt_training = ${{ steps.params.outputs.no_system_prompt }}' not in content:
        print("❌ no_system_prompt_training not set in training config")
        return False
    
    # Check that sample.py uses the flag
    if '--no_system_prompt=True' not in content:
        print("❌ sample.py doesn't use no_system_prompt flag")
        return False
    
    print("✅ Workflow correctly sets and uses no_system_prompt flag")
    return True

def test_evaluation_flag():
    """Test that evaluation scripts handle the no_system_prompt flag."""
    print("\n🔍 Testing evaluation script no_system_prompt flag handling...")
    
    # Read the echo_fidelity.py file
    with open('NanEcho/evaluation/echo_fidelity.py', 'r') as f:
        content = f.read()
    
    # Check that the flag is in the argument parser
    if '--no_system_prompt_test' not in content:
        print("❌ --no_system_prompt_test argument not found in echo_fidelity.py")
        return False
    
    # Check that the flag is parsed
    if "no_system_prompt_test = args.no_system_prompt_test.lower() in ('true', '1', 'yes', 'on')" not in content:
        print("❌ no_system_prompt_test flag not parsed correctly")
        return False
    
    # Check that the flag is used in model config
    if "model_config.no_system_prompt = no_system_prompt_test" not in content:
        print("❌ no_system_prompt_test not set in model config")
        return False
    
    print("✅ Evaluation scripts correctly handle no_system_prompt flag")
    return True

def test_sample_py():
    """Test that sample.py exists and handles the no_system_prompt flag."""
    print("\n🔍 Testing sample.py no_system_prompt flag handling...")
    
    if not os.path.exists('sample.py'):
        print("❌ sample.py does not exist")
        return False
    
    # Read the sample.py file
    with open('sample.py', 'r') as f:
        content = f.read()
    
    # Check that the flag is in the argument parser
    if '--no_system_prompt' not in content:
        print("❌ --no_system_prompt argument not found in sample.py")
        return False
    
    # Check that the flag is parsed correctly
    if "no_system_prompt = args.no_system_prompt.lower() in ('true', '1', 'yes', 'on')" not in content:
        print("❌ no_system_prompt flag not parsed correctly in sample.py")
        return False
    
    # Check that the flag affects generation
    if 'if no_system_prompt:' not in content:
        print("❌ no_system_prompt flag not used in generation logic")
        return False
    
    print("✅ sample.py correctly handles no_system_prompt flag")
    return True

def test_model_config():
    """Test that EchoModelConfig handles the no_system_prompt flag."""
    print("\n🔍 Testing EchoModelConfig no_system_prompt flag handling...")
    
    # Read the netalk.py file
    with open('NanEcho/netalk.py', 'r') as f:
        content = f.read()
    
    # Check that the flag is initialized
    if 'self.no_system_prompt = False' not in content:
        print("❌ no_system_prompt not initialized in EchoModelConfig")
        return False
    
    # Check that the flag is used in generate method
    if 'if self.no_system_prompt:' not in content:
        print("❌ no_system_prompt not used in generate method")
        return False
    
    print("✅ EchoModelConfig correctly handles no_system_prompt flag")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Testing no_system_prompt Flag Implementation")
    print("=" * 60)
    
    tests = [
        ("prepare_nanecho.py flag handling", test_prepare_nanecho_flag),
        ("Workflow flag usage", test_workflow_flag),
        ("Evaluation script flag handling", test_evaluation_flag),
        ("sample.py flag handling", test_sample_py),
        ("EchoModelConfig flag handling", test_model_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! The no_system_prompt flag is correctly implemented.")
        print("\n🎯 Key Implementation Points:")
        print("   1. Flag is parsed as a string and converted to boolean")
        print("   2. Flag triggers relentless persona mode in prepare_nanecho.py")
        print("   3. Flag is passed through workflow parameters")
        print("   4. Flag is used in training config as no_system_prompt_training")
        print("   5. Flag is used in sample.py for testing without system prompts")
        print("   6. Flag is used in evaluation scripts to verify persona persistence")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())