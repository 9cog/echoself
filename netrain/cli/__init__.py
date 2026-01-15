"""
NetRain CLI - Command Line Interface for Deep Tree Echo Training
"""

import argparse
import sys
import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

from ..models import DeepTreeEchoTransformer
from ..training import EchoTrainer
from ..data import EchoDataLoader
from ..utils import ConfigManager, setup_logging


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_command(args):
    """Build and train the Deep Tree Echo LLM."""
    print(f"🚀 Starting NetRain build process...")
    print(f"📁 Configuration: {args.config}")
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✅ Configuration loaded: {config['name']} v{config['version']}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(config.get('logging', {}))
    
    # Initialize configuration manager
    config_manager = ConfigManager(config)
    
    # Prepare data
    print("\n📊 Preparing data...")
    try:
        data_loader = EchoDataLoader(config_manager.data_config)
        train_dataset, val_dataset = data_loader.prepare_datasets()
        print(f"✅ Data prepared: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        sys.exit(1)
    
    # Initialize model
    print("\n🧠 Initializing Deep Tree Echo model...")
    try:
        model = DeepTreeEchoTransformer(config_manager.model_config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model initialized: {param_count:,} parameters")
        print(f"   - Layers: {config_manager.model_config['architecture']['n_layers']}")
        print(f"   - Hidden size: {config_manager.model_config['architecture']['n_embd']}")
        print(f"   - Tree depth: {config_manager.model_config['architecture']['tree_depth']}")
        print(f"   - Echo layers: {config_manager.model_config['architecture']['echo_layers']}")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        sys.exit(1)
    
    # Setup trainer
    print("\n🏋️ Setting up trainer...")
    try:
        trainer = EchoTrainer(
            model=model,
            config=config_manager,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        print(f"✅ Trainer ready")
        print(f"   - Max steps: {config_manager.training_config['max_steps']}")
        print(f"   - Learning rate: {config_manager.training_config['optimizer']['learning_rate']}")
        print(f"   - Batch size: {config_manager.data_config['loader']['batch_size']}")
    except Exception as e:
        print(f"❌ Trainer setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Start training
    print("\n🎯 Starting training...")
    try:
        trainer.train()
        print("\n✨ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer.save_checkpoint("interrupted_checkpoint")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 NetRain build completed successfully!")
    return 0


def test_command(args):
    """Test the trained model."""
    print("🧪 Testing Deep Tree Echo model...")
    
    # Load configuration
    config = load_config(args.config)
    config_manager = ConfigManager(config)
    
    # Load model
    model = DeepTreeEchoTransformer(config_manager.model_config)
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run inference test
    model.eval()
    test_prompt = args.prompt or "The deep tree echo architecture"
    
    print(f"\nTest prompt: {test_prompt}")
    print("Generating response...")
    
    # Simple generation (placeholder - implement actual generation)
    print("Response: [Model inference would go here]")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NetRain - Deep Tree Echo LLM Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build and train the model')
    build_parser.add_argument('config', help='Path to configuration file (netrain.yml)')
    build_parser.add_argument('--resume', help='Resume from checkpoint', default=None)
    build_parser.set_defaults(func=build_command)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the trained model')
    test_parser.add_argument('config', help='Path to configuration file')
    test_parser.add_argument('--checkpoint', help='Path to model checkpoint')
    test_parser.add_argument('--prompt', help='Test prompt for generation')
    test_parser.set_defaults(func=test_command)
    
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()