#!/usr/bin/env python3
"""
Test to verify the ConnectionMask AttributeError fix.

This test specifically validates that the ConnectionMask initialization order
fix prevents the AttributeError that was occurring in the GitHub Actions workflow.
"""

import sys
import torch
sys.path.append('.')

from nanecho_model import ConnectionMask, create_nanecho_model

def test_connection_mask_initialization():
    """Test that ConnectionMask initializes without AttributeError."""
    print("🧪 Testing ConnectionMask initialization...")
    
    try:
        # This should not raise AttributeError anymore
        mask = ConnectionMask((768, 768), initial_ratio=0.1)
        print("   ✅ ConnectionMask created successfully")
        
        # Test property access
        ratio = mask.current_ratio
        print(f"   ✅ Current ratio accessible: {ratio:.1%}")
        
        # Test property setting
        mask.current_ratio = 0.2
        new_ratio = mask.current_ratio
        print(f"   ✅ Ratio updated successfully: {new_ratio:.1%}")
        
        # Test that _current_ratio_tensor exists
        assert hasattr(mask, '_current_ratio_tensor'), "Missing _current_ratio_tensor attribute"
        print("   ✅ _current_ratio_tensor buffer exists")
        
        return True
        
    except AttributeError as e:
        if '_current_ratio_tensor' in str(e):
            print(f"   ❌ FAILED: AttributeError still occurs: {e}")
            return False
        else:
            print(f"   ❌ FAILED: Unexpected AttributeError: {e}")
            return False
    except Exception as e:
        print(f"   ❌ FAILED: Unexpected error: {e}")
        return False

def test_model_creation():
    """Test that the full model can be created without errors."""
    print("🧪 Testing full NanEcho model creation...")
    
    try:
        model = create_nanecho_model()
        print("   ✅ NanEcho model created successfully")
        
        # Test that all connection masks are properly initialized
        mask_count = 0
        for name, module in model.named_modules():
            if isinstance(module, ConnectionMask):
                mask_count += 1
                assert hasattr(module, '_current_ratio_tensor'), f"Missing _current_ratio_tensor in {name}"
        
        print(f"   ✅ All {mask_count} ConnectionMasks properly initialized")
        return True
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

def test_state_dict_functionality():
    """Test that state_dict includes all connection mask buffers."""
    print("🧪 Testing state_dict functionality...")
    
    try:
        model = create_nanecho_model()
        state_dict = model.state_dict()
        
        # Count _current_ratio_tensor buffers in state_dict
        ratio_buffers = [k for k in state_dict.keys() if '_current_ratio_tensor' in k]
        print(f"   ✅ Found {len(ratio_buffers)} _current_ratio_tensor buffers in state_dict")
        
        # Verify they all have the correct initial value
        expected_value = 0.1  # default initial_connections
        for key in ratio_buffers:
            value = state_dict[key].item()
            assert abs(value - expected_value) < 1e-6, f"Wrong value in {key}: {value} vs {expected_value}"
        
        print(f"   ✅ All buffers have correct initial value: {expected_value}")
        return True
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Testing ConnectionMask AttributeError Fix")
    print("=" * 60)
    
    tests = [
        test_connection_mask_initialization,
        test_model_creation,
        test_state_dict_functionality,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("✅ All tests passed! ConnectionMask AttributeError is fixed.")
        return True
    else:
        print("❌ Some tests failed. The fix needs more work.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)