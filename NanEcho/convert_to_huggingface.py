#!/usr/bin/env python3
"""
Convert NanEcho PyTorch checkpoint to HuggingFace format.

This script converts a trained NanEcho model to a HuggingFace-compatible
GPT-2 model format for easy deployment and sharing on the HuggingFace Hub.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np


def load_nanecho_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load a NanEcho checkpoint."""
    print(f"📦 Loading NanEcho checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state' in checkpoint:
        # Export format from cached training
        model_state = checkpoint['model_state']
        metadata = checkpoint.get('metadata', {})
    elif 'model' in checkpoint:
        # Standard training checkpoint
        model_state = checkpoint['model']
        metadata = checkpoint.get('config', {})
    else:
        # Raw model state
        model_state = checkpoint
        metadata = {}
    
    print(f"✅ Loaded checkpoint with {len(model_state)} state dict entries")
    if metadata:
        print(f"📊 Metadata: {json.dumps(metadata, indent=2)[:200]}...")
    
    return {
        'model_state': model_state,
        'metadata': metadata,
        'full_checkpoint': checkpoint
    }


def convert_to_hf_format(model_state: Dict[str, torch.Tensor], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert NanEcho model state to HuggingFace GPT-2 format.
    
    NanEcho uses a GPT-2-like architecture with some extensions (connection masks,
    adaptive attention, etc.). We extract the core transformer weights.
    """
    print("🔄 Converting to HuggingFace format...")
    
    hf_state = {}
    
    # Mapping from NanEcho to HuggingFace parameter names
    # NanEcho follows nanoGPT naming conventions
    name_mapping = {
        # Token embeddings
        'transformer.wte.weight': 'transformer.wte.weight',
        'transformer.wpe.weight': 'transformer.wpe.weight',
        
        # Layer norm
        'transformer.ln_f.weight': 'transformer.ln_f.weight',
        'transformer.ln_f.bias': 'transformer.ln_f.bias',
        
        # Language model head
        'lm_head.weight': 'lm_head.weight',
    }
    
    # Copy basic parameters
    for nanecho_name, hf_name in name_mapping.items():
        if nanecho_name in model_state:
            hf_state[hf_name] = model_state[nanecho_name]
            print(f"  ✓ Mapped {nanecho_name} -> {hf_name}")
    
    # Convert transformer blocks
    n_layers = config.get('n_layer', 12)
    for layer_idx in range(n_layers):
        layer_prefix = f'transformer.h.{layer_idx}'
        
        # Attention weights
        attn_mappings = {
            f'{layer_prefix}.attn.c_attn.weight': f'{layer_prefix}.attn.c_attn.weight',
            f'{layer_prefix}.attn.c_attn.bias': f'{layer_prefix}.attn.c_attn.bias',
            f'{layer_prefix}.attn.c_proj.weight': f'{layer_prefix}.attn.c_proj.weight',
            f'{layer_prefix}.attn.c_proj.bias': f'{layer_prefix}.attn.c_proj.bias',
        }
        
        # MLP weights
        mlp_mappings = {
            f'{layer_prefix}.mlp.c_fc.weight': f'{layer_prefix}.mlp.c_fc.weight',
            f'{layer_prefix}.mlp.c_fc.bias': f'{layer_prefix}.mlp.c_fc.bias',
            f'{layer_prefix}.mlp.c_proj.weight': f'{layer_prefix}.mlp.c_proj.weight',
            f'{layer_prefix}.mlp.c_proj.bias': f'{layer_prefix}.mlp.c_proj.bias',
        }
        
        # Layer norms
        ln_mappings = {
            f'{layer_prefix}.ln_1.weight': f'{layer_prefix}.ln_1.weight',
            f'{layer_prefix}.ln_1.bias': f'{layer_prefix}.ln_1.bias',
            f'{layer_prefix}.ln_2.weight': f'{layer_prefix}.ln_2.weight',
            f'{layer_prefix}.ln_2.bias': f'{layer_prefix}.ln_2.bias',
        }
        
        # Copy all layer parameters
        for nanecho_name, hf_name in {**attn_mappings, **mlp_mappings, **ln_mappings}.items():
            if nanecho_name in model_state:
                hf_state[hf_name] = model_state[nanecho_name]
    
    print(f"✅ Converted {len(hf_state)} parameters to HuggingFace format")
    return hf_state


def create_hf_config(nanecho_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create HuggingFace model configuration from NanEcho config."""
    print("⚙️ Creating HuggingFace configuration...")
    
    hf_config = {
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        
        # Core architecture
        "vocab_size": nanecho_config.get('vocab_size', 50257),
        "n_embd": nanecho_config.get('n_embd', 768),
        "n_head": nanecho_config.get('n_head', 12),
        "n_layer": nanecho_config.get('n_layer', 12),
        "n_positions": nanecho_config.get('block_size', 1024),
        
        # Regularization
        "embd_pdrop": nanecho_config.get('dropout', 0.1),
        "attn_pdrop": nanecho_config.get('dropout', 0.1),
        "resid_pdrop": nanecho_config.get('dropout', 0.1),
        
        # Other settings
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "bos_token_id": 0,
        "eos_token_id": 0,
        
        # Echo Self specific metadata (preserved as custom fields)
        "echo_self_version": "1.0",
        "echo_self_persona_dimensions": nanecho_config.get('persona_dimensions', []),
        "echo_self_adaptive_attention": nanecho_config.get('enable_adaptive_attention', True),
        "echo_self_recursive_reasoning": nanecho_config.get('enable_recursive_reasoning', True),
    }
    
    print(f"✅ Created HuggingFace config with {hf_config['n_layer']} layers, "
          f"{hf_config['n_embd']} embedding dim")
    return hf_config


def create_tokenizer_config() -> Dict[str, Any]:
    """Create tokenizer configuration for tiktoken-based tokenization."""
    return {
        "tokenizer_class": "GPT2Tokenizer",
        "model_max_length": 1024,
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    }


def create_model_card(
    metadata: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path
) -> None:
    """Create a model card with training details."""
    print("📝 Creating model card...")
    
    # Extract training info
    checkpoint_id = metadata.get('checkpoint_id', 'unknown')
    quality_score = metadata.get('quality_score', 'N/A')
    val_loss = metadata.get('val_loss', 'N/A')
    iteration = metadata.get('iteration', 'N/A')
    
    model_card = f"""---
language: en
tags:
- gpt2
- echo-self
- cognitive-architecture
- deep-tree-echo
license: mit
---

# EchoSelf NanEcho Model

## Model Description

This is a **Deep Tree Echo** cognitive architecture model trained using the EchoSelf framework.
The model implements adaptive attention mechanisms, persona dimensions, and recursive reasoning
capabilities inspired by cognitive science and AGI research.

## Model Architecture

- **Base Architecture**: GPT-2
- **Parameters**: {config.get('n_layer', 'N/A')} layers, {config.get('n_embd', 'N/A')} embedding dimensions
- **Vocabulary Size**: {config.get('vocab_size', 'N/A')}
- **Context Length**: {config.get('block_size', 'N/A')} tokens

## Training Details

- **Checkpoint ID**: {checkpoint_id}
- **Training Iteration**: {iteration}
- **Validation Loss**: {val_loss}
- **Quality Score**: {quality_score}

## Echo Self Features

This model incorporates several cognitive architecture features:

- **Adaptive Attention**: Dynamic threshold adjustment based on cognitive load
- **Persona Dimensions**: Multi-dimensional cognitive processing
  - Cognitive, Introspective, Adaptive, Recursive
  - Synergistic, Holographic, Neural-Symbolic, Dynamic
- **Recursive Reasoning**: Multi-level introspection capabilities
- **Hypergraph Patterns**: Neural-symbolic pattern encoding

## Usage

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("9cog/echoself-nanecho")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Generate text
inputs = tokenizer("Echo Self is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Training Data

The model was trained on:
- Echo Self documentation and cognitive architecture descriptions
- Hypergraph reasoning patterns
- Persona dimension examples
- Recursive introspection samples

## Limitations

This is a research model exploring cognitive architectures. It should not be used for:
- Production applications without further validation
- Tasks requiring factual accuracy
- Critical decision-making systems

## Citation

```bibtex
@misc{{echoself-nanecho,
  title={{EchoSelf NanEcho: Deep Tree Echo Cognitive Architecture}},
  author={{9cog}},
  year={{2026}},
  url={{https://github.com/9cog/echoself}}
}}
```

## More Information

- **Repository**: https://github.com/9cog/echoself
- **Documentation**: See repository README for detailed architecture information
"""
    
    model_card_path = output_dir / "README.md"
    with open(model_card_path, 'w') as f:
        f.write(model_card)
    
    print(f"✅ Model card saved to {model_card_path}")


def save_hf_model(
    hf_state: Dict[str, torch.Tensor],
    hf_config: Dict[str, Any],
    tokenizer_config: Dict[str, Any],
    metadata: Dict[str, Any],
    output_dir: str
) -> None:
    """Save model in HuggingFace format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving HuggingFace model to {output_path}")
    
    # Save model weights
    model_path = output_path / "pytorch_model.bin"
    torch.save(hf_state, model_path)
    print(f"  ✓ Saved model weights: {model_path}")
    
    # Save config
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(hf_config, f, indent=2)
    print(f"  ✓ Saved config: {config_path}")
    
    # Save tokenizer config
    tokenizer_path = output_path / "tokenizer_config.json"
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"  ✓ Saved tokenizer config: {tokenizer_path}")
    
    # Save metadata
    metadata_path = output_path / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved training metadata: {metadata_path}")
    
    # Create model card
    create_model_card(metadata, hf_config, output_path)
    
    print(f"\n✅ HuggingFace model saved successfully to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NanEcho checkpoint to HuggingFace format"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to NanEcho checkpoint (.pt file)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='hf-model',
        help='Output directory for HuggingFace model'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Optional: Path to model config JSON (if not in checkpoint)'
    )
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint_data = load_nanecho_checkpoint(args.checkpoint)
    model_state = checkpoint_data['model_state']
    metadata = checkpoint_data['metadata']
    
    # Get config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Try to extract from checkpoint metadata
        config = metadata.get('model_config', {})
        
        # Provide sensible defaults
        if not config:
            print("⚠️ No config found, using defaults based on model structure")
            # Try to infer from model state
            if 'transformer.wte.weight' in model_state:
                vocab_size, n_embd = model_state['transformer.wte.weight'].shape
                config = {
                    'vocab_size': vocab_size,
                    'n_embd': n_embd,
                    'n_layer': 12,
                    'n_head': 12,
                    'block_size': 1024,
                    'dropout': 0.1,
                }
                print(f"  Inferred config: {config}")
    
    # Convert to HuggingFace format
    hf_state = convert_to_hf_format(model_state, config)
    hf_config = create_hf_config(config)
    tokenizer_config = create_tokenizer_config()
    
    # Save HuggingFace model
    save_hf_model(hf_state, hf_config, tokenizer_config, metadata, args.output_dir)
    
    print("\n🎉 Conversion complete!")
    print(f"\nTo upload to HuggingFace Hub:")
    print(f"  1. Install: pip install huggingface_hub")
    print(f"  2. Login: huggingface-cli login")
    print(f"  3. Upload: huggingface-cli upload 9cog/echoself-nanecho {args.output_dir}")


if __name__ == '__main__':
    main()
