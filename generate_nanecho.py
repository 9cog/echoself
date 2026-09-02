#!/usr/bin/env python3
"""
NanEcho Generation Script

Generate text using a trained NanEcho model with Echo Self capabilities.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from nanecho_model import NanEchoModel, NanEchoConfig

ROOT = Path(__file__).resolve().parent
if str(ROOT / "NanEcho") not in sys.path:
    sys.path.insert(0, str(ROOT / "NanEcho"))
from runtime import NanEchoRuntime  # noqa: E402


class NanEchoGenerator:
    """Text generator for NanEcho models, preferring the shared production runtime."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        self.iteration = 0
        self.connection_ratio = 0.0
        self.runtime = None
        try:
            self.runtime = NanEchoRuntime.load(
                checkpoint_path,
                device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu",
            )
            self.model = self.runtime.model
            self.config = self.runtime.config
            self.iteration = self.runtime.metadata["iteration"]
            self.connection_ratio = self.model.connection_ratio
            print(f"✅ Runtime loaded from {checkpoint_path}")
            print(f"   • Iteration: {self.iteration:,}")
            print(f"   • Connection ratio: {self.connection_ratio:.1%}")
        except Exception as exc:
            print(f"⚠️  NanEchoRuntime unavailable ({exc}); using legacy checkpoint loader")
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        print(f"📂 Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract configuration
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            self.config = NanEchoConfig(
                vocab_size=config_dict.get('vocab_size', 50257),
                n_embd=config_dict.get('n_embd', 768),
                n_head=config_dict.get('n_head', 12),
                n_layer=config_dict.get('n_layer', 12),
                block_size=config_dict.get('block_size', 1024),
                dropout=0.0,  # No dropout for generation
                bias=config_dict.get('bias', True),
                initial_connections=config_dict.get('initial_connections', 0.1),
                connection_growth_rate=config_dict.get('connection_growth_rate', 0.05),
                max_connections=config_dict.get('max_connections', 1.0)
            )
        else:
            self.config = NanEchoConfig()
        
        # Create model
        self.model = NanEchoModel(self.config).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set connection ratio
        if 'connection_ratio' in checkpoint:
            self.model.connection_ratio = checkpoint['connection_ratio']
            self.connection_ratio = checkpoint['connection_ratio']
        
        # Get iteration
        if 'iteration' in checkpoint:
            self.iteration = checkpoint['iteration']
        
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   • Iteration: {self.iteration:,}")
        print(f"   • Connection ratio: {self.connection_ratio:.1%}")
        print(f"   • Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def generate(
        self,
        prompt: str = "",
        max_length: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        num_samples: int = 1
    ) -> list:
        """Generate text from prompt."""
        if self.runtime is not None:
            if not prompt:
                raise ValueError("prompt must not be empty when using NanEchoRuntime")
            prompt_tokens = len(self.runtime.encode(prompt))
            max_new_tokens = max(0, max_length - prompt_tokens)
            return [
                self.runtime.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                for _ in range(num_samples)
            ]

        # Legacy path for checkpoints the production runtime cannot load
        if prompt:
            input_ids = torch.tensor([[ord(c) % self.config.vocab_size for c in prompt]], 
                                    device=self.device)
        else:
            input_ids = torch.randint(0, self.config.vocab_size, (1, 1), device=self.device)
        
        results = []
        
        for i in range(num_samples):
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample
                )
            text = ''.join([chr(min(t.item(), 127)) for t in generated[0]])
            results.append(text)
        
        return results
    
    def interactive_generate(self):
        """Interactive generation mode."""
        print(f"""
╔══════════════════════════════════════════════════════════╗
║         NanEcho Interactive Generation                    ║
╚══════════════════════════════════════════════════════════╝

🤖 Model loaded with {self.connection_ratio:.1%} active connections
   Type 'quit' to exit, 'help' for options

""")
        
        while True:
            try:
                prompt = input("📝 Enter prompt (or command): ").strip()
                
                if prompt.lower() == 'quit':
                    break
                elif prompt.lower() == 'help':
                    self.show_help()
                    continue
                elif prompt.startswith('/'):
                    self.handle_command(prompt)
                    continue
                
                # Generate text
                print("\n🔮 Generating...\n")
                results = self.generate(
                    prompt=prompt,
                    max_length=200,
                    temperature=0.8,
                    num_samples=1
                )
                
                print("=" * 60)
                print(results[0])
                print("=" * 60)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show help message."""
        print("""
Commands:
  /temp <value>    Set temperature (0.1-2.0)
  /length <value>  Set max length (10-1000)
  /topk <value>    Set top-k sampling (0-100)
  /topp <value>    Set top-p sampling (0.0-1.0)
  /echo            Generate Echo Self description
  /persona         Generate persona dimension text
  /hypergraph      Generate hypergraph pattern
  /recursive       Generate recursive reasoning
  /adaptive        Generate adaptive attention text
  help             Show this help
  quit             Exit
""")
    
    def handle_command(self, command: str):
        """Handle special commands."""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/echo':
            prompt = "Echo Self is"
            results = self.generate(prompt, max_length=150)
            print("\n🔮 Echo Self Generation:\n")
            print("=" * 60)
            print(results[0])
            print("=" * 60)
        
        elif cmd == '/persona':
            prompt = "The cognitive dimension"
            results = self.generate(prompt, max_length=150)
            print("\n🔮 Persona Dimension Generation:\n")
            print("=" * 60)
            print(results[0])
            print("=" * 60)
        
        elif cmd == '/hypergraph':
            prompt = "Hypergraph node"
            results = self.generate(prompt, max_length=150)
            print("\n🔮 Hypergraph Pattern Generation:\n")
            print("=" * 60)
            print(results[0])
            print("=" * 60)
        
        elif cmd == '/recursive':
            prompt = "Level 1: Analyzing"
            results = self.generate(prompt, max_length=150)
            print("\n🔮 Recursive Reasoning Generation:\n")
            print("=" * 60)
            print(results[0])
            print("=" * 60)
        
        elif cmd == '/adaptive':
            prompt = "Attention threshold:"
            results = self.generate(prompt, max_length=150)
            print("\n🔮 Adaptive Attention Generation:\n")
            print("=" * 60)
            print(results[0])
            print("=" * 60)
        
        else:
            print(f"Unknown command: {cmd}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate text with NanEcho model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default="",
                       help='Generation prompt')
    parser.add_argument('--max_length', type=int, default=200,
                       help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top-p sampling')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to generate')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive generation mode')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = NanEchoGenerator(args.checkpoint, args.device)
    
    if args.interactive:
        # Interactive mode
        generator.interactive_generate()
    else:
        # Single generation
        results = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_samples=args.num_samples
        )
        
        for i, text in enumerate(results):
            if args.num_samples > 1:
                print(f"\n=== Sample {i+1} ===")
            print(text)


if __name__ == "__main__":
    main()