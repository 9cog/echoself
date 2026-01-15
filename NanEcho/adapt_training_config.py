#!/usr/bin/env python3
"""
Training Configuration Adaptation Script

This script adapts training configuration based on available data size.
It can automatically adjust block_size to prevent training failures.
"""

import os
import sys
import numpy as np
import re


def get_data_size(data_dir: str) -> tuple:
    """
    Get the size of training and validation data.
    
    Args:
        data_dir: Directory containing train.bin and val.bin
        
    Returns:
        tuple: (train_tokens, val_tokens)
    """
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    
    try:
        train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
        val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
        return len(train_data), len(val_data)
    except Exception as e:
        print(f"❌ Error reading data files: {e}")
        return 0, 0


def suggest_block_size(train_tokens: int, val_tokens: int) -> int:
    """
    Suggest an appropriate block_size based on available data.
    
    Args:
        train_tokens: Number of training tokens
        val_tokens: Number of validation tokens
        
    Returns:
        int: Suggested block_size
    """
    min_tokens = min(train_tokens, val_tokens)
    
    # Use 90% of the smaller dataset to be safe
    max_safe_block_size = int(min_tokens * 0.9)
    
    # Common block sizes in descending order
    common_sizes = [1024, 512, 256, 128]
    
    for size in common_sizes:
        if max_safe_block_size >= size:
            return size
    
    # If even 128 is too large, use a custom size
    return max(64, max_safe_block_size)


def adapt_config_file(config_path: str, new_block_size: int) -> bool:
    """
    Adapt a configuration file to use a new block_size.
    
    Args:
        config_path: Path to the configuration file
        new_block_size: New block size to use
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Replace block_size line
        pattern = r'block_size\s*=\s*\d+'
        replacement = f'block_size = {new_block_size}'
        
        if re.search(pattern, content):
            new_content = re.sub(pattern, replacement, content)
            
            with open(config_path, 'w') as f:
                f.write(new_content)
            
            print(f"✅ Updated {config_path} with block_size = {new_block_size}")
            return True
        else:
            print(f"⚠️ Could not find block_size setting in {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating config file: {e}")
        return False


def main():
    """Main configuration adaptation function."""
    if len(sys.argv) < 3:
        print("Usage: python adapt_training_config.py <data_directory> <config_file>")
        print("Example: python adapt_training_config.py data/nanecho config/train_nanecho_ci.py")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    config_file = sys.argv[2]
    
    print(f"🔧 Analyzing data and adapting training configuration...")
    print(f"   Data directory: {data_dir}")
    print(f"   Config file: {config_file}")
    
    # Get data sizes
    train_tokens, val_tokens = get_data_size(data_dir)
    
    if train_tokens == 0 or val_tokens == 0:
        print("❌ Could not read training data files")
        sys.exit(1)
    
    print(f"📊 Data Analysis:")
    print(f"   Training tokens: {train_tokens:,}")
    print(f"   Validation tokens: {val_tokens:,}")
    
    # Check if adaptation is needed
    min_required = 1024  # Default block_size from configs
    
    if min(train_tokens, val_tokens) <= min_required:
        print(f"⚠️ Data size requires block_size adaptation")
        
        suggested_block_size = suggest_block_size(train_tokens, val_tokens)
        print(f"💡 Suggested block_size: {suggested_block_size}")
        
        if os.path.exists(config_file):
            if adapt_config_file(config_file, suggested_block_size):
                print("✅ Configuration adapted successfully")
            else:
                print("❌ Failed to adapt configuration")
                sys.exit(1)
        else:
            print(f"❌ Configuration file not found: {config_file}")
            sys.exit(1)
    else:
        print("✅ No adaptation needed - data size is sufficient for default configuration")
    
    return 0


if __name__ == "__main__":
    main()