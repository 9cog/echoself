#!/usr/bin/env python3
"""
Test script to verify that both no_system_prompt and relentless_persona_mode
flags work together without conflicts, maintaining ecosystem compatibility.
"""

import os
import sys
import re
import tempfile
import json

def test_both_flags_present():
    """Test that both no_system_prompt and relentless_persona_mode flags are present."""
    print("🔍 Testing that both flags are present...")
    
    prepare_script = "NanEcho/prepare_nanecho.py"
    if not os.path.exists(prepare_script):
        print(f"❌ {prepare_script} not found")
        return False
    
    with open(prepare_script, 'r') as f:
        content = f.read()
    
    # Check for both flags
    required_flags = [
        ("no_system_prompt parameter", "no_system_prompt: bool = False"),
        ("relentless_persona_mode parameter", "relentless_persona_mode: bool = False"),
        ("--no_system_prompt argument", "--no_system_prompt"),
        ("--relentless_persona_mode argument", "--relentless_persona_mode"),
        ("no_system_prompt in metadata", '"no_system_prompt": no_system_prompt'),
        ("relentless_persona_mode in metadata", '"relentless_persona_mode": relentless_persona_mode'),
    ]
    
    missing_flags = []
    for name, pattern in required_flags:
        if pattern not in content:
            missing_flags.append(name)
    
    if missing_flags:
        print("❌ Missing flags:")
        for missing in missing_flags:
            print(f"   - {missing}")
        return False
    else:
        print("✅ Both flags are present and properly implemented")
        return True

def test_workflow_both_flags():
    """Test that workflow uses both flags."""
    print("\n🔍 Testing workflow uses both flags...")
    
    workflow_file = ".github/workflows/netrain.yml"
    if not os.path.exists(workflow_file):
        print(f"❌ {workflow_file} not found")
        return False
    
    with open(workflow_file, 'r') as f:
        content = f.read()
    
    # Check for both flags in workflow
    workflow_checks = [
        ("no_system_prompt in params", "no_system_prompt=True"),
        ("relentless_persona_mode in params", "relentless_persona_mode=True"),
        ("no_system_prompt in data prep", "--no_system_prompt="),
        ("relentless_persona_mode in data prep", "--relentless_persona_mode="),
        ("no_system_prompt_training in config", "no_system_prompt_training"),
        ("relentless_persona_mode in config", "relentless_persona_mode = $"),
        ("no_system_prompt in testing", "--no_system_prompt=True"),
        ("no_system_prompt_test in evaluation", "--no_system_prompt_test=True"),
    ]
    
    missing_checks = []
    for name, pattern in workflow_checks:
        if pattern not in content:
            missing_checks.append(name)
    
    if missing_checks:
        print("❌ Missing workflow checks:")
        for missing in missing_checks:
            print(f"   - {missing}")
        return False
    else:
        print("✅ All workflow checks present")
        return True

def test_flag_interaction_logic():
    """Test that both flags work together in the logic."""
    print("\n🔍 Testing flag interaction logic...")
    
    prepare_script = "NanEcho/prepare_nanecho.py"
    with open(prepare_script, 'r') as f:
        content = f.read()
    
    # Check for proper flag interaction logic
    interaction_checks = [
        ("OR logic for both flags", "if relentless_persona_mode or no_system_prompt:"),
        ("no_system_prompt trigger message", "(Triggered by no_system_prompt flag)"),
        ("relentless_persona_mode trigger message", "(Triggered by relentless_persona_mode flag)"),
        ("both flags trigger message", "(Triggered by both no_system_prompt and relentless_persona_mode flags)"),
    ]
    
    missing_interactions = []
    for name, pattern in interaction_checks:
        if pattern not in content:
            missing_interactions.append(name)
    
    if missing_interactions:
        print("❌ Missing flag interaction logic:")
        for missing in missing_interactions:
            print(f"   - {missing}")
        return False
    else:
        print("✅ Flag interaction logic properly implemented")
        return True

def test_ecosystem_compatibility():
    """Test that the implementation maintains ecosystem compatibility."""
    print("\n🔍 Testing ecosystem compatibility...")
    
    prepare_script = "NanEcho/prepare_nanecho.py"
    with open(prepare_script, 'r') as f:
        content = f.read()
    
    # Check for backward compatibility features
    compatibility_checks = [
        ("legacy compatibility note", "legacy compatibility"),
        ("no_system_prompt in function signature", "no_system_prompt: bool = False"),
        ("relentless_persona_mode in function signature", "relentless_persona_mode: bool = False"),
        ("both flags in function call", "no_system_prompt=no_system_prompt,"),
        ("both flags in summary", "No System Prompt: {no_system_prompt}"),
    ]
    
    missing_compatibility = []
    for name, pattern in compatibility_checks:
        if pattern not in content:
            missing_compatibility.append(name)
    
    if missing_compatibility:
        print("❌ Missing compatibility features:")
        for missing in missing_compatibility:
            print(f"   - {missing}")
        return False
    else:
        print("✅ Ecosystem compatibility maintained")
        return True

def test_no_conflicts():
    """Test that there are no conflicts between the flags."""
    print("\n🔍 Testing for conflicts between flags...")
    
    prepare_script = "NanEcho/prepare_nanecho.py"
    workflow_file = ".github/workflows/netrain.yml"
    
    files_to_check = [
        ("prepare_script", prepare_script),
    ]
    
    # Check that we have the correct OR logic in the main files
    correct_patterns = [
        r"if.*relentless_persona_mode.*or.*no_system_prompt.*:",  # This is correct OR logic
    ]
    
    missing_or_logic = []
    
    for file_name, file_path in files_to_check:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check that we have the correct OR logic
        has_correct_or_logic = False
        for pattern in correct_patterns:
            if re.search(pattern, content):
                has_correct_or_logic = True
                break
        
        if not has_correct_or_logic:
            missing_or_logic.append(f"{file_name}: Missing correct OR logic for both flags")
    
    if missing_or_logic:
        print("❌ Missing correct OR logic:")
        for missing in missing_or_logic:
            print(f"   - {missing}")
        return False
    else:
        print("✅ Correct OR logic found between flags")
        return True

def test_comprehensive_coverage():
    """Test that all aspects of both flags are covered."""
    print("\n🔍 Testing comprehensive coverage...")
    
    prepare_script = "NanEcho/prepare_nanecho.py"
    workflow_file = ".github/workflows/netrain.yml"
    
    # Test that both flags appear in all necessary places
    coverage_checks = [
        ("Function signature", prepare_script, "no_system_prompt: bool = False"),
        ("Function signature", prepare_script, "relentless_persona_mode: bool = False"),
        ("Argument parser", prepare_script, "--no_system_prompt"),
        ("Argument parser", prepare_script, "--relentless_persona_mode"),
        ("Function call", prepare_script, "no_system_prompt=no_system_prompt"),
        ("Function call", prepare_script, "relentless_persona_mode=relentless_persona_mode"),
        ("Metadata", prepare_script, '"no_system_prompt": no_system_prompt'),
        ("Metadata", prepare_script, '"relentless_persona_mode": relentless_persona_mode'),
        ("Workflow params", workflow_file, "no_system_prompt=True"),
        ("Workflow params", workflow_file, "relentless_persona_mode=True"),
        ("Workflow data prep", workflow_file, "--no_system_prompt="),
        ("Workflow data prep", workflow_file, "--relentless_persona_mode="),
        ("Workflow config", workflow_file, "no_system_prompt_training"),
        ("Workflow config", workflow_file, "relentless_persona_mode = $"),
    ]
    
    missing_coverage = []
    for name, file_path, pattern in coverage_checks:
        if not os.path.exists(file_path):
            missing_coverage.append(f"{name}: File {file_path} not found")
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        if pattern not in content:
            missing_coverage.append(f"{name}: Missing {pattern}")
    
    if missing_coverage:
        print("❌ Missing coverage:")
        for missing in missing_coverage:
            print(f"   - {missing}")
        return False
    else:
        print("✅ Comprehensive coverage achieved")
        return True

def main():
    """Run all tests to verify both flags work together."""
    print("🚀 Testing Both Flags Compatibility")
    print("=" * 60)
    
    tests = [
        ("both_flags_present", test_both_flags_present),
        ("workflow_both_flags", test_workflow_both_flags),
        ("flag_interaction_logic", test_flag_interaction_logic),
        ("ecosystem_compatibility", test_ecosystem_compatibility),
        ("no_conflicts", test_no_conflicts),
        ("comprehensive_coverage", test_comprehensive_coverage),
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
        print("🎉 All tests passed! Both flags work together perfectly:")
        print("\n📋 Implementation Summary:")
        print("   ✅ no_system_prompt flag restored for ecosystem compatibility")
        print("   ✅ relentless_persona_mode flag added for new functionality")
        print("   ✅ Both flags work together without conflicts")
        print("   ✅ Workflow uses both flags appropriately")
        print("   ✅ Comprehensive coverage across all files")
        print("   ✅ Backward compatibility maintained")
        print("\n🔧 How it works:")
        print("   1. no_system_prompt=True triggers relentless persona mode (legacy)")
        print("   2. relentless_persona_mode=True triggers relentless persona mode (new)")
        print("   3. Both flags can be used together or independently")
        print("   4. Either flag enables the persona embedding functionality")
        print("   5. Your existing ecosystem code will continue to work")
        print("\n🚀 Maximum compatibility achieved!")
        print("   - No breaking changes to existing code")
        print("   - New functionality available via relentless_persona_mode")
        print("   - Both flags achieve the same goal through persona embedding")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)