"""
Generate text from a trained General Relevance Transformer.
Fixes the membrane state shape mismatch by resetting states each generation.
"""
import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from netrain.models.experimental.general_relevance import GeneralRelevanceTransformer


class CharTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = {int(v): k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, text: str, max_len: int = 256):
        tokens = [self.char_to_idx.get('<bos>', 2)]
        for c in text[:max_len - 2]:
            tokens.append(self.char_to_idx.get(c, 1))
        tokens.append(self.char_to_idx.get('<eos>', 3))
        return tokens
    
    def decode(self, indices):
        chars = []
        for idx in indices:
            c = self.idx_to_char.get(idx, '')
            if c in ('<pad>', '<bos>', '<eos>', '<unk>'):
                continue
            chars.append(c)
        return ''.join(chars)


def generate(model, tokenizer, prompt, max_len=200, temperature=0.8, device='cpu'):
    """Generate text, resetting membrane states for each call."""
    model.eval()
    
    tokens = tokenizer.encode(prompt, max_len=64)
    # Remove the EOS token from the prompt encoding
    if tokens[-1] == tokenizer.char_to_idx.get('<eos>', 3):
        tokens = tokens[:-1]
    
    generated = list(tokens)
    
    with torch.no_grad():
        for step in range(max_len):
            # Always use full generated sequence (up to 128 chars)
            context_tokens = generated[-128:]
            context = torch.tensor([context_tokens], dtype=torch.long, device=device)
            
            # Fresh membrane states each forward pass (no state carryover bug)
            outputs = model(context, membrane_states=None)
            
            # Sample from last position
            logits = outputs['logits'][:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            
            # Stop at EOS or pad
            if next_token in (tokenizer.char_to_idx.get('<eos>', 3), 
                             tokenizer.char_to_idx.get('<pad>', 0)):
                break
    
    return tokenizer.decode(generated)


def main():
    checkpoint_dir = 'checkpoints/general_relevance'
    
    # Load config
    with open(f'{checkpoint_dir}/config.json', 'r') as f:
        config = json.load(f)
    
    print(f"Loading model: {config['total_params']:,} params")
    print(f"  d_model={config['d_model']}, n_heads={config['n_heads']}, "
          f"layers={config['num_layers']}, membranes={config['num_membranes']}")
    
    # Load tokenizer
    tokenizer = CharTokenizer()
    tokenizer.load(f'{checkpoint_dir}/tokenizer.json')
    
    # Load model
    model = GeneralRelevanceTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_membranes=config['num_membranes'],
        num_layers=config['num_layers'],
        matula_numbers=config['matula_numbers'],
        dropout=0.0  # No dropout during inference
    )
    
    checkpoint = torch.load(f'{checkpoint_dir}/best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded best model (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
    
    # Generate samples
    prompts = [
        "Deep Tree Echo",
        "The Ricci flow",
        "Identity is",
        "Curvature and",
        "The manifold",
        "Attention",
        "Relevance",
        "The cognitive",
        "Memory",
        "Evolution"
    ]
    
    print(f"\n{'='*60}")
    print(f"  GENERATION SAMPLES (temperature=0.8)")
    print(f"{'='*60}")
    
    for prompt in prompts:
        output = generate(model, tokenizer, prompt, max_len=200, temperature=0.8)
        print(f"\n  [{prompt}] → {output[:300]}")
    
    # Interactive mode
    print(f"\n{'='*60}")
    print(f"  INTERACTIVE MODE (type 'quit' to exit)")
    print(f"{'='*60}")
    
    while True:
        try:
            prompt = input("\n  > ")
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
            output = generate(model, tokenizer, prompt, max_len=300, temperature=0.8)
            print(f"  → {output}")
        except (EOFError, KeyboardInterrupt):
            break
    
    print("\n  Done.")


if __name__ == '__main__':
    main()
