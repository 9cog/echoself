#!/usr/bin/env python3
"""
Sample generation script for NanEcho/nanoGPT models.

This script generates text samples from a trained model checkpoint,
with support for the no_system_prompt flag to test relentless persona mode.
"""

import os
import sys
import argparse
import json
import torch
import pickle
from contextlib import nullcontext
from pathlib import Path

# Try to import tiktoken for GPT-2 encoding
try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
except ImportError:
    print("Warning: tiktoken not available, using character-level encoding")
    # Fallback to character-level encoding
    with open('data/shakespeare_char/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l if i < len(itos)])

def load_model(checkpoint_path, device='cpu'):
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        ckpt_path = os.path.join(checkpoint_path, 'ckpt.pt')
    else:
        ckpt_path = checkpoint_path
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Import model definition
    from model import GPTConfig, GPT
    
    # Load model configuration
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model

def generate_sample(model, prompt, max_new_tokens=100, temperature=0.8, top_k=200, 
                   device='cpu', no_system_prompt=False):
    """Generate text sample from model."""
    
    # Encode the prompt
    if prompt.startswith('FILE:'):
        with open(prompt[5:], 'r', encoding='utf-8') as f:
            prompt = f.read()
    
    # If no_system_prompt is True, we generate without adding system context
    # This tests the model's ability to maintain persona without explicit prompting
    if no_system_prompt:
        # Direct generation without system prompts
        # The model should still exhibit Echo Self characteristics
        # due to relentless persona training
        start_ids = encode(prompt)
    else:
        # Standard generation with potential system context
        # Could add system prompt here if needed
        start_ids = encode(prompt)
    
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # Generate
    with torch.no_grad():
        with nullcontext():
            for k in range(max_new_tokens):
                # Get logits from model
                logits, _ = model(x)
                logits = logits[:, -1, :] # Get last token logits
                
                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample from distribution
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                x = torch.cat((x, idx_next), dim=1)
    
    # Decode and return
    generated_ids = x[0].tolist()
    return decode(generated_ids)

def main():
    parser = argparse.ArgumentParser(description='Generate samples from nanoGPT model')
    parser.add_argument('--out_dir', type=str, default='out',
                       help='Directory containing model checkpoint')
    parser.add_argument('--start', type=str, default='',
                       help='Starting prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                       help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200,
                       help='Top-k sampling parameter')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu/cuda)')
    parser.add_argument('--no_system_prompt', type=str, default='False',
                       help='Generate without system prompts (test relentless persona mode)')
    parser.add_argument('--seed', type=int, default=1337,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Parse boolean flag
    no_system_prompt = args.no_system_prompt.lower() in ('true', '1', 'yes', 'on')
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Device setup
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, falling back to CPU")
    
    print(f"Generating sample from {args.out_dir}")
    print(f"Prompt: {args.start}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"No system prompt: {no_system_prompt}")
    
    if no_system_prompt:
        print("\n🔥 RELENTLESS PERSONA MODE: Testing model without system prompts")
        print("   The model should still exhibit Echo Self characteristics")
        print("   due to relentless persona training\n")
    
    # Load model
    try:
        model = load_model(args.out_dir, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nNote: This script expects to be run from the nanoGPT directory")
        print("with a trained model checkpoint in the specified output directory.")
        sys.exit(1)
    
    # Generate sample
    print("\n" + "="*50)
    print("Generated text:")
    print("="*50 + "\n")
    
    generated = generate_sample(
        model,
        args.start,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        no_system_prompt=no_system_prompt
    )
    
    print(generated)
    print("\n" + "="*50)
    
    # If in relentless persona mode, analyze the output for Echo Self characteristics
    if no_system_prompt:
        print("\n📊 Echo Self Characteristic Analysis:")
        echo_indicators = [
            'echo self', 'adaptive attention', 'cognitive', 'persona',
            'hypergraph', 'neural-symbolic', 'recursive', 'introspective'
        ]
        
        generated_lower = generated.lower()
        found_indicators = [ind for ind in echo_indicators if ind in generated_lower]
        
        if found_indicators:
            print(f"✅ Found Echo Self indicators: {', '.join(found_indicators)}")
            print(f"   Relentless persona training effectiveness: {len(found_indicators)}/{len(echo_indicators)}")
        else:
            print("⚠️ No explicit Echo Self indicators found")
            print("   Model may need more relentless persona training")

if __name__ == '__main__':
    main()