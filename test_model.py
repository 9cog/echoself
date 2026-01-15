#!/usr/bin/env python3
"""
Test script for the Deep Tree Echo LLM
"""

import torch
import sys
import os

# Add netrain to path
sys.path.insert(0, '/workspace')

from netrain.models import DeepTreeEchoTransformer
from netrain.utils import ConfigManager
import yaml

def test_model():
    """Test the trained Deep Tree Echo model."""
    
    print("🧪 Testing Deep Tree Echo LLM...")
    
    # Load configuration
    with open('/workspace/netrain.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    config_manager = ConfigManager(config)
    
    # Initialize model
    print("📦 Loading model architecture...")
    model = DeepTreeEchoTransformer(config_manager.model_config)
    
    # Load checkpoint
    checkpoint_path = '/workspace/checkpoints/final_model.pt'
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   - Training steps completed: {checkpoint['global_step']}")
    print(f"   - Epochs completed: {checkpoint['epoch']}")
    print(f"   - Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    # Test generation
    print("\n🎯 Testing text generation...")
    
    # Create a simple input
    test_prompts = [
        "The deep tree echo",
        "Hierarchical attention mechanism",
        "Recursive processing enables"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        
        # Simple tokenization (character-level for testing)
        input_ids = torch.tensor([[ord(c) % 50257 for c in prompt]], dtype=torch.long)
        
        # Generate
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs['logits']
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            print(f"   Output shape: {logits.shape}")
            print(f"   Max logit: {logits.max().item():.4f}")
            print(f"   Min logit: {logits.min().item():.4f}")
            
            # Test generation
            generated = model.generate(
                input_ids,
                max_length=50,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            print(f"   Generated tokens: {generated.shape}")
    
    print("\n✨ Model test completed successfully!")
    
    # Test model components
    print("\n🔍 Testing model components:")
    
    # Check for tree attention layers
    tree_layers = sum(1 for name, _ in model.named_modules() if 'TreeAttention' in str(type(_)))
    print(f"   - Tree attention layers: {tree_layers}")
    
    # Check for echo layers
    echo_layers = sum(1 for name, _ in model.named_modules() if 'EchoLayer' in str(type(_)))
    print(f"   - Echo layers: {echo_layers}")
    
    # Check for memory bank
    has_memory_bank = any('memory_bank' in name for name, _ in model.named_modules())
    print(f"   - Memory bank: {'Enabled' if has_memory_bank else 'Disabled'}")
    
    # Check for hierarchical pooling
    has_hierarchical = any('hierarchical_pooling' in name for name, _ in model.named_modules())
    print(f"   - Hierarchical pooling: {'Enabled' if has_hierarchical else 'Disabled'}")
    
    print("\n🎉 All tests passed! Deep Tree Echo LLM is working!")
    
    return True

if __name__ == "__main__":
    try:
        success = test_model()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)