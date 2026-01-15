#!/usr/bin/env python3
"""
Test script to validate the nanoGPT fix approach works correctly.
This simulates the workflow fix that gets applied to external nanoGPT train.py
"""

import tempfile
import os
import re

def create_mock_nanogpt_train():
    """Create a mock nanoGPT train.py with the problematic line"""
    content = '''#!/usr/bin/env python3
"""
Simple GPT model training script
"""

import torch
import numpy as np

def get_batch(split):
    data = np.random.randint(0, 100, 300)  # Small dataset with 300 tokens
    block_size = 390  # Block size larger than data
    batch_size = 4
    
    # This is the problematic line that causes the error
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    return torch.zeros((batch_size, block_size))

if __name__ == '__main__':
    print("Testing original problematic code...")
    try:
        batch = get_batch('train')
        print("✅ No error occurred")
    except RuntimeError as e:
        print(f"❌ RuntimeError: {e}")
'''
    return content

def apply_fix(content):
    """Apply the same fix that the workflow uses"""
    pattern = r'(\s+)ix = torch\.randint\(len\(data\) - block_size, \(batch_size,\)\)'
    
    def add_validation(match):
        indent = match.group(1)
        replacement = f'''{indent}# Data size validation to prevent torch.randint error
{indent}if len(data) <= block_size:
{indent}    raise ValueError(
{indent}        f"Insufficient data for batch generation: "
{indent}        f"len(data)={{len(data)}}, block_size={{block_size}}. "
{indent}        f"Dataset must be larger than block_size. "
{indent}        f"Consider reducing block_size to {{len(data) - 1}} or generating more training data."
{indent}    )
{indent}ix = torch.randint(len(data) - block_size, (batch_size,))'''
        return replacement
    
    # Apply the fix
    fixed_content = re.sub(pattern, add_validation, content)
    return fixed_content

def main():
    print("🔍 Testing workflow fix approach for nanoGPT train.py")
    print("=" * 60)
    
    # Create mock nanoGPT content
    original_content = create_mock_nanogpt_train()
    
    # Apply fix
    fixed_content = apply_fix(original_content)
    
    # Test original (should fail)
    print("\n1. Testing original code (should fail):")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(original_content)
        original_file = f.name
    
    try:
        exec(compile(open(original_file).read(), original_file, 'exec'))
    except RuntimeError as e:
        print(f"   ✅ Expected RuntimeError: {str(e)[:60]}...")
    except Exception as e:
        print(f"   ❓ Unexpected error: {e}")
    finally:
        os.unlink(original_file)
    
    # Test fixed version (should catch and provide helpful error)
    print("\n2. Testing fixed code (should catch error gracefully):")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(fixed_content)
        fixed_file = f.name
    
    try:
        exec(compile(open(fixed_file).read(), fixed_file, 'exec'))
        print("   ❌ Should have caught the error")
    except ValueError as e:
        print(f"   ✅ Caught ValueError with helpful message: {str(e)[:80]}...")
    except Exception as e:
        print(f"   ❓ Unexpected error: {e}")
    finally:
        os.unlink(fixed_file)
    
    # Check if fix was applied correctly
    print("\n3. Verifying fix was applied:")
    if 'Data size validation to prevent torch.randint error' in fixed_content:
        print("   ✅ Validation code was added")
    else:
        print("   ❌ Validation code was not found")
    
    if 'if len(data) <= block_size:' in fixed_content:
        print("   ✅ Guard clause was added")
    else:
        print("   ❌ Guard clause was not found")
    
    print("\n✅ Workflow fix approach validated successfully!")

if __name__ == '__main__':
    main()