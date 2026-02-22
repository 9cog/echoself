#!/usr/bin/env python3
"""
Download and convert HuggingFace model to NanEcho checkpoint format.

This script downloads a trained model from HuggingFace Hub and converts it
back to NanEcho checkpoint format for continued training.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np


def download_from_hub(repo_id: str, output_dir: str, token: Optional[str] = None) -> Path:
    """Download model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ Error: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)
    
    print(f"📥 Downloading model from HuggingFace Hub: {repo_id}")
    
    model_dir = snapshot_download(
        repo_id=repo_id,
        token=token,
        local_dir=output_dir,
        local_dir_use_symlinks=False
    )
    
    print(f"✅ Downloaded to {model_dir}")
    return Path(model_dir)


def load_hf_model(model_dir: Path) -> Dict[str, Any]:
    """Load HuggingFace model files."""
    print(f"📦 Loading HuggingFace model from {model_dir}")
    
    # Load model weights
    model_path = model_dir / "pytorch_model.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    model_state = torch.load(model_path, map_location='cpu')
    print(f"  ✓ Loaded {len(model_state)} parameters")
    
    # Load config
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"  ✓ Loaded config")
    else:
        config = {}
        print(f"  ⚠️ No config.json found")
    
    # Load training metadata if available
    metadata_path = model_dir / "training_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"  ✓ Loaded training metadata")
    else:
        metadata = {}
        print(f"  ⚠️ No training metadata found")
    
    return {
        'model_state': model_state,
        'config': config,
        'metadata': metadata
    }


def convert_to_nanecho_format(
    hf_state: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert HuggingFace model to NanEcho checkpoint format.
    
    The model architecture is compatible, so this is mainly about
    ensuring the checkpoint structure matches what NanEcho expects.
    """
    print("🔄 Converting to NanEcho checkpoint format...")
    
    # Validate that essential components exist
    required_keys = ['transformer.wte.weight', 'transformer.wpe.weight', 'transformer.ln_f.weight']
    missing_keys = [key for key in required_keys if key not in hf_state]
    
    if missing_keys:
        raise ValueError(
            f"❌ Incompatible model: Missing required GPT-2 components: {missing_keys}\n"
            f"   Expected a GPT-2 style model with nanoGPT/NanEcho architecture.\n"
            f"   Please ensure the HuggingFace model is a GPT-2 variant.\n"
            f"   See NanEcho/HUGGINGFACE_README.md for supported model types."
        )
    
    # HuggingFace and NanEcho use compatible naming (both based on GPT-2)
    # so we can use the state dict mostly as-is
    nanecho_state = {}
    
    # Direct mapping - HF and nanoGPT/NanEcho use the same naming
    for name, param in hf_state.items():
        nanecho_state[name] = param
    
    # Create NanEcho checkpoint structure
    checkpoint = {
        'model': nanecho_state,
        'config': {
            'vocab_size': config.get('vocab_size', 50257),
            'n_embd': config.get('n_embd', 768),
            'n_head': config.get('n_head', 12),
            'n_layer': config.get('n_layer', 12),
            'block_size': config.get('n_positions', 1024),
            'dropout': config.get('embd_pdrop', 0.1),
            'bias': True,
            
            # Echo Self specific features (preserved from metadata)
            'enable_adaptive_attention': config.get('echo_self_adaptive_attention', True),
            'enable_persona_dimensions': True,
            'enable_recursive_reasoning': config.get('echo_self_recursive_reasoning', True),
            'enable_hypergraph_patterns': True,
            'persona_dimensions': config.get('echo_self_persona_dimensions', [
                'cognitive', 'introspective', 'adaptive', 'recursive',
                'synergistic', 'holographic', 'neural_symbolic', 'dynamic'
            ]),
        },
        'metadata': {
            'source': 'huggingface',
            'original_checkpoint_id': metadata.get('checkpoint_id', 'hf-download'),
            'quality_score': metadata.get('quality_score', 0.0),
            'val_loss': metadata.get('val_loss', 0.0),
            'iteration': metadata.get('iteration', 0),
        }
    }
    
    print(f"✅ Converted to NanEcho format with {len(nanecho_state)} parameters")
    return checkpoint


def save_nanecho_checkpoint(checkpoint: Dict[str, Any], output_path: str) -> None:
    """Save NanEcho checkpoint."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving NanEcho checkpoint to {output_file}")
    
    torch.save(checkpoint, output_file)
    
    # Also save config as JSON for easy inspection
    config_file = output_file.parent / "model_config.json"
    with open(config_file, 'w') as f:
        json.dump(checkpoint['config'], f, indent=2)
    
    print(f"  ✓ Saved checkpoint: {output_file}")
    print(f"  ✓ Saved config: {config_file}")
    print(f"\n✅ NanEcho checkpoint ready for training")


def main():
    parser = argparse.ArgumentParser(
        description="Download model from HuggingFace and convert to NanEcho format"
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default='9cog/echoself-nanecho',
        help='HuggingFace repository ID (default: 9cog/echoself-nanecho)'
    )
    parser.add_argument(
        '--output-checkpoint',
        type=str,
        default='out-nanecho/hf_init.pt',
        help='Output path for NanEcho checkpoint'
    )
    parser.add_argument(
        '--token',
        type=str,
        help='HuggingFace token (for private repos)',
        default=os.environ.get('HF_TOKEN')
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='/tmp/hf-download',
        help='Temporary directory for download'
    )
    
    args = parser.parse_args()
    
    # Download from HuggingFace Hub
    model_dir = download_from_hub(args.repo_id, args.cache_dir, args.token)
    
    # Load HuggingFace model
    hf_data = load_hf_model(model_dir)
    
    # Convert to NanEcho format
    nanecho_checkpoint = convert_to_nanecho_format(
        hf_data['model_state'],
        hf_data['config'],
        hf_data['metadata']
    )
    
    # Save checkpoint
    save_nanecho_checkpoint(nanecho_checkpoint, args.output_checkpoint)
    
    print("\n🎉 Download and conversion complete!")
    print(f"\nTo use this checkpoint for training:")
    print(f"  python train_cached.py --resume-from {args.output_checkpoint}")


if __name__ == '__main__':
    main()
