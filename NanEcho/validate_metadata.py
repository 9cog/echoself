#!/usr/bin/env python3
"""
Metadata Validation Script for NanEcho Training

This script validates that training data exists and has sufficient metadata
for the NanEcho training workflow. It replaces inline Python code in the
GitHub Actions workflow to avoid IndentationError issues.
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path


def validate_training_files(data_dir: str) -> bool:
    """
    Validate that required training files exist and are accessible.
    
    Args:
        data_dir: Directory containing training data
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin") 
    
    print("🔍 Verifying training data files exist...")
    print(f"   Checking: {train_path}")
    print(f"   Checking: {val_path}")
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found: {train_path}")
        print("💡 This indicates data preparation failed. Common causes:")
        print("   1. tiktoken failed to download GPT-2 vocabulary (network issue)")
        print("   2. Insufficient Echo Self content in source files")
        print("   3. Data preparation script exited with error before creating files")
        print("   4. File permissions or disk space issues")
        return False
    
    if not os.path.exists(val_path):
        print(f"❌ Validation data not found: {val_path}")
        print("💡 This indicates data preparation failed. Common causes:")
        print("   1. tiktoken failed to download GPT-2 vocabulary (network issue)")
        print("   2. Insufficient Echo Self content in source files") 
        print("   3. Data preparation script exited with error before creating files")
        print("   4. File permissions or disk space issues")
        return False
    
    # Check file sizes
    train_size = os.path.getsize(train_path)
    val_size = os.path.getsize(val_path)
    
    if train_size == 0:
        print(f"❌ Training data file is empty: {train_path}")
        return False
        
    if val_size == 0:
        print(f"❌ Validation data file is empty: {val_path}")
        return False
    
    print("✅ Both train.bin and val.bin found and accessible")
    print(f"   train.bin: {train_size:,} bytes")
    print(f"   val.bin: {val_size:,} bytes")
    return True


def validate_file_sizes(data_dir: str, min_block_size: int = 256, max_block_size: int = 1024, allow_adaptation: bool = False) -> bool:
    """
    Validate that training files have sufficient tokens for training.
    
    Args:
        data_dir: Directory containing training data
        min_block_size: Minimum acceptable tokens
        max_block_size: Maximum block size used in training
        allow_adaptation: If True, allows smaller datasets that can be adapted
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    
    print("📊 Validating dataset sizes...")
    
    try:
        # Load training data to check size
        train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
        val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
        
        train_tokens = len(train_data)
        val_tokens = len(val_data)
        
        print(f"   Training data: {train_tokens:,} tokens")
        print(f"   Validation data: {val_tokens:,} tokens")
        
        # Absolute minimum for any training
        absolute_min = 64
        
        # Check absolute minimum requirements
        if train_tokens <= absolute_min:
            print(f"❌ Training dataset too small: {train_tokens:,} tokens < {absolute_min}")
            print("💡 Cannot proceed with training. Please:")
            print("   - Increase echo_depth and persona_weight parameters")
            print("   - Check that Echo Self content exists in source files")
            print("   - Verify data preparation completed successfully")
            return False
        
        if val_tokens <= absolute_min:
            print(f"❌ Validation dataset too small: {val_tokens:,} tokens < {absolute_min}")
            print("💡 Cannot proceed with training. Please:")
            print("   - Increase echo_depth and persona_weight parameters")
            print("   - Check that Echo Self content exists in source files")
            print("   - Verify data preparation completed successfully")
            return False
        
        # Check normal minimum requirements
        if train_tokens <= min_block_size or val_tokens <= min_block_size:
            if allow_adaptation:
                print(f"⚠️  Dataset size requires adaptation:")
                print(f"   Training: {train_tokens:,} tokens, Validation: {val_tokens:,} tokens")
                print(f"   Will adapt block_size for available data")
                return True  # Allow adaptation to handle this
            else:
                print(f"❌ Training dataset too small: {train_tokens:,} tokens < {min_block_size}")
                print("💡 Cannot proceed with training. Please:")
                print("   - Increase echo_depth and persona_weight parameters")
                print("   - Check that Echo Self content exists in source files")
                print("   - Verify data preparation completed successfully")
                return False
        
        # Check against max block size (warnings only)
        if train_tokens <= max_block_size:
            print(f"⚠️  Warning: Training data ({train_tokens:,} tokens) smaller than typical block_size ({max_block_size})")
            print(f"   Consider reducing block_size to {train_tokens - 10} or generating more data")
        
        if val_tokens <= max_block_size:
            print(f"⚠️  Warning: Validation data ({val_tokens:,} tokens) smaller than typical block_size ({max_block_size})")
            print(f"   Consider reducing block_size to {val_tokens - 10} or generating more data")
        
        print("✅ Dataset size validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error validating dataset sizes: {e}")
        return False


def check_metadata(data_dir: str) -> None:
    """
    Check and display metadata for the training dataset.
    
    Args:
        data_dir: Directory containing training data
    """
    meta_pkl_path = os.path.join(data_dir, "meta.pkl")
    metadata_json_path = os.path.join(data_dir, "metadata.json")
    
    print("📊 Checking dataset metadata...")
    
    try:
        if os.path.exists(meta_pkl_path):
            with open(meta_pkl_path, 'rb') as f:
                meta = pickle.load(f)
            print(f"   Vocab size: {meta.get('vocab_size', 'unknown')}")
            print(f"   Characters: {meta.get('chars', 'unknown')}")
            
        elif os.path.exists(metadata_json_path):
            with open(metadata_json_path, 'r') as f:
                meta = json.load(f)
            print(f"   Total tokens: {meta.get('train_tokens', 0) + meta.get('val_tokens', 0):,}")
            print(f"   Echo depth: {meta.get('echo_depth', 'unknown')}")
            print(f"   Persona weight: {meta.get('persona_weight', 'unknown')}")
            print(f"   Synthetic samples: {meta.get('synthetic_samples', 'unknown')}")
            print(f"   Deep Tree Echo mode: {meta.get('deep_tree_echo_mode', 'unknown')}")
            
        else:
            print("   No metadata file found")
            
    except Exception as e:
        print(f"   Error reading metadata: {e}")


def suggest_block_size_adaptation(data_dir: str) -> int:
    """
    Suggest an appropriate block_size based on available data.
    
    Args:
        data_dir: Directory containing training data
        
    Returns:
        int: Suggested block_size, or -1 if data is insufficient
    """
    train_path = os.path.join(data_dir, "train.bin")
    
    try:
        train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
        train_tokens = len(train_data)
        
        if train_tokens <= 256:
            return -1  # Insufficient data
        
        # Suggest a safe block_size (90% of available data)
        suggested_block_size = min(1024, int(train_tokens * 0.9))
        
        # Round down to common block sizes
        common_sizes = [128, 256, 512, 1024]
        for size in reversed(common_sizes):
            if suggested_block_size >= size:
                return size
        
        return 128  # Minimum fallback
        
    except Exception as e:
        print(f"❌ Error calculating block_size suggestion: {e}")
        return -1


def main():
    """Main validation function."""
    if len(sys.argv) < 2:
        print("Usage: python validate_metadata.py <data_directory> [--allow-adaptation]")
        print("Example: python validate_metadata.py data/nanecho")
        print("         python validate_metadata.py data/nanecho --allow-adaptation")
        print("         python validate_metadata.py ../../nanoGPT/data/nanecho")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    allow_adaptation = "--allow-adaptation" in sys.argv
    
    # Convert to absolute path for better error messages
    abs_data_dir = os.path.abspath(data_dir)
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print(f"   Absolute path: {abs_data_dir}")
        print(f"💡 This usually means data preparation failed or the directory path is incorrect")
        print(f"   Common causes:")
        print(f"   1. Data preparation script failed due to tiktoken network issues")
        print(f"   2. Incorrect relative path (current dir: {os.getcwd()})")
        print(f"   3. Directory was not created properly during workflow setup")
        sys.exit(1)
    
    print(f"🔍 Validating NanEcho training data in: {data_dir}")
    print(f"   Absolute path: {abs_data_dir}")
    if allow_adaptation:
        print("🔧 Adaptation mode enabled - will allow smaller datasets")
    print("="*60)
    
    # Step 1: Check file existence
    if not validate_training_files(data_dir):
        sys.exit(1)
    
    # Step 2: Check file sizes
    if not validate_file_sizes(data_dir, allow_adaptation=allow_adaptation):
        print("\n🔧 Attempting to suggest block_size adaptation...")
        suggested_block_size = suggest_block_size_adaptation(data_dir)
        
        if suggested_block_size > 0:
            print(f"💡 Suggested block_size adaptation: {suggested_block_size}")
            print(f"   Add to your training config: block_size = {suggested_block_size}")
            # Still exit with error since data is insufficient for default config
            sys.exit(1)
        else:
            print("❌ Data is too small for any reasonable block_size")
            sys.exit(1)
    
    # Step 3: Display metadata
    check_metadata(data_dir)
    
    print("\n✅ All validation checks passed!")
    print("🚀 Training data is ready for NanEcho model training")


if __name__ == "__main__":
    main()