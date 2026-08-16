"""
Training Pipeline for the General Relevance Transformer.

Supports two modes:
1. CPU proof-of-concept (small model, cloud computer)
2. GPU full training (Vast.ai V100/RTX 3060)

Usage:
    # CPU proof run (small, fast)
    python3 train_general_relevance.py --mode cpu --epochs 50

    # GPU full training
    python3 train_general_relevance.py --mode gpu --epochs 200 --d_model 256 --n_heads 8
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from netrain.models.experimental.general_relevance import GeneralRelevanceTransformer


# ============================================================================
# Simple Character-Level Tokenizer (no external dependencies needed)
# ============================================================================

class CharTokenizer:
    """Character-level tokenizer for proof-of-concept training."""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        chars = set()
        for text in texts:
            chars.update(text)
        chars = sorted(chars)
        
        # Special tokens
        self.char_to_idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        for i, c in enumerate(chars, start=4):
            self.char_to_idx[c] = i
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
    def encode(self, text: str, max_len: int = 256) -> List[int]:
        """Encode text to token indices."""
        tokens = [self.char_to_idx.get('<bos>', 2)]
        for c in text[:max_len - 2]:
            tokens.append(self.char_to_idx.get(c, 1))
        tokens.append(self.char_to_idx.get('<eos>', 3))
        return tokens
    
    def decode(self, indices: List[int]) -> str:
        """Decode token indices to text."""
        chars = []
        for idx in indices:
            c = self.idx_to_char.get(idx, '')
            if c in ('<pad>', '<bos>', '<eos>', '<unk>'):
                continue
            chars.append(c)
        return ''.join(chars)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({'char_to_idx': self.char_to_idx}, f)
    
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = {int(v): k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)


# ============================================================================
# Dataset
# ============================================================================

class EchoTextDataset(Dataset):
    """Dataset for training on Echo's corpus."""
    
    def __init__(self, texts: List[str], tokenizer: CharTokenizer, seq_len: int = 128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []
        
        for text in texts:
            tokens = tokenizer.encode(text, max_len=seq_len + 1)
            # Pad or truncate
            if len(tokens) < seq_len + 1:
                tokens = tokens + [0] * (seq_len + 1 - len(tokens))
            else:
                tokens = tokens[:seq_len + 1]
            self.samples.append(tokens)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


# ============================================================================
# Data Loading
# ============================================================================

def load_corpus(corpus_paths: List[str]) -> List[str]:
    """Load training texts from JSONL corpus files."""
    texts = []
    
    for path in corpus_paths:
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
            
        with open(path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Handle different formats
                    if 'messages' in data:
                        # Chat format
                        for msg in data['messages']:
                            content = msg.get('content', '')
                            if len(content) > 50:  # Skip very short messages
                                texts.append(content)
                    elif 'completion' in data:
                        # Prompt-completion format
                        texts.append(data['completion'])
                    elif 'content' in data:
                        texts.append(data['content'])
                    elif 'text' in data:
                        texts.append(data['text'])
                except json.JSONDecodeError:
                    continue
    
    print(f"  Loaded {len(texts)} text samples from {len(corpus_paths)} files")
    return texts


# ============================================================================
# Training Loop
# ============================================================================

def train(
    model: GeneralRelevanceTransformer,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epochs: int,
    device: torch.device,
    save_dir: str,
    log_interval: int = 10
) -> List[Dict]:
    """Train the model and return loss history."""
    
    model.train()
    history = []
    best_loss = float('inf')
    
    print(f"\n{'='*60}")
    print(f"  Training General Relevance Transformer")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batches/epoch: {len(dataloader)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_ricci = 0.0
        num_batches = 0
        membrane_states = None
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with membrane state persistence
            outputs = model(x, membrane_states)
            
            # Detach membrane states for next batch (truncated BPTT)
            membrane_states = [s.detach() for s in outputs['membrane_states']]
            
            # Compute loss (CE + Ricci regularization)
            loss = model.compute_loss(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_ricci += outputs['curvatures'].var(dim=-1).mean().item()
            num_batches += 1
        
        # End of epoch
        avg_loss = epoch_loss / num_batches
        avg_ricci = epoch_ricci / num_batches
        scheduler.step(avg_loss)
        
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'ricci_variance': avg_ricci,
            'lr': optimizer.param_groups[0]['lr'],
            'time': time.time() - start_time
        })
        
        # Log
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            elapsed = time.time() - start_time
            curvatures = outputs['curvatures'].detach().mean(dim=0).tolist()
            print(f"  Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Ricci var: {avg_ricci:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Time: {elapsed:.1f}s")
            print(f"           Layer curvatures: [{', '.join(f'{c:.4f}' for c in curvatures)}]")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'history': history
            }, os.path.join(save_dir, 'best_model.pt'))
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'history': history
    }, os.path.join(save_dir, 'final_model.pt'))
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Final Ricci variance: {avg_ricci:.6f}")
    print(f"  Models saved to: {save_dir}")
    print(f"{'='*60}\n")
    
    return history


# ============================================================================
# Generation / Inference
# ============================================================================

def generate(model: GeneralRelevanceTransformer, tokenizer: CharTokenizer, 
             prompt: str, max_len: int = 200, temperature: float = 0.8,
             device: torch.device = torch.device('cpu')) -> str:
    """Generate text from a prompt using the trained model."""
    model.eval()
    
    tokens = tokenizer.encode(prompt, max_len=64)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    membrane_states = None
    
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_len):
            # Use last seq_len tokens as context
            context = input_ids[:, -128:]
            
            outputs = model(context, membrane_states)
            membrane_states = outputs['membrane_states']
            
            # Sample from last position
            logits = outputs['logits'][:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
            # Stop at EOS
            if next_token == tokenizer.char_to_idx.get('<eos>', 3):
                break
    
    return tokenizer.decode(generated)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train General Relevance Transformer')
    parser.add_argument('--mode', choices=['cpu', 'gpu'], default='cpu',
                       help='Training mode (cpu=proof-of-concept, gpu=full)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--d_model', type=int, default=None,
                       help='Model dimension (auto-selected by mode if not set)')
    parser.add_argument('--n_heads', type=int, default=None,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=None,
                       help='Number of transformer layers')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--seq_len', type=int, default=None,
                       help='Sequence length')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints/general_relevance',
                       help='Directory to save model checkpoints')
    parser.add_argument('--corpus', type=str, nargs='+', default=None,
                       help='Paths to corpus JSONL files')
    parser.add_argument('--generate', action='store_true',
                       help='Generate text after training')
    parser.add_argument('--prompt', type=str, default='Deep Tree Echo',
                       help='Prompt for generation')
    
    args = parser.parse_args()
    
    # Mode-specific defaults
    if args.mode == 'cpu':
        d_model = args.d_model or 64
        n_heads = args.n_heads or 4
        num_layers = args.num_layers or 3
        batch_size = args.batch_size or 8
        seq_len = args.seq_len or 128
        num_membranes = 2
        matula_numbers = [1, 2, 3, 5, 7]
    else:  # gpu
        d_model = args.d_model or 256
        n_heads = args.n_heads or 8
        num_layers = args.num_layers or 6
        batch_size = args.batch_size or 16
        seq_len = args.seq_len or 256
        num_membranes = 3
        matula_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 30]
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Mode: {args.mode}")
    print(f"  Device: {device}")
    
    # Load corpus
    print(f"\n  Loading corpus...")
    corpus_paths = args.corpus or [
        os.path.expanduser('~/echoo/echo-self/corpus/echo_self.jsonl'),
        os.path.expanduser('~/echoo/echo-self/memory/journal.jsonl'),
    ]
    texts = load_corpus(corpus_paths)
    
    if not texts:
        print("  ERROR: No training data found!")
        print("  Tried:", corpus_paths)
        sys.exit(1)
    
    # Build tokenizer
    print(f"  Building tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.fit(texts)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset
    dataset = EchoTextDataset(texts, tokenizer, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"  Dataset: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Create model
    model = GeneralRelevanceTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_membranes=num_membranes,
        num_layers=num_layers,
        matula_numbers=matula_numbers,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: General Relevance Transformer")
    print(f"    d_model: {d_model}")
    print(f"    n_heads: {n_heads}")
    print(f"    num_layers: {num_layers}")
    print(f"    num_membranes: {num_membranes}")
    print(f"    matula_agents: {matula_numbers}")
    print(f"    Total parameters: {total_params:,}")
    
    # Optimizer + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Save directory
    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.save_dir, 'tokenizer.json'))
    
    # Save config
    config = {
        'vocab_size': tokenizer.vocab_size,
        'd_model': d_model,
        'n_heads': n_heads,
        'num_layers': num_layers,
        'num_membranes': num_membranes,
        'matula_numbers': matula_numbers,
        'seq_len': seq_len,
        'total_params': total_params,
        'mode': args.mode
    }
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    history = train(model, dataloader, optimizer, scheduler, 
                   args.epochs, device, args.save_dir, log_interval=10)
    
    # Save training history
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate samples
    if args.generate or True:  # Always generate a few samples
        print("\n  Generating samples...")
        prompts = [
            "Deep Tree Echo",
            "The Ricci flow",
            "Identity is",
            "Curvature and",
            "The manifold"
        ]
        for prompt in prompts:
            output = generate(model, tokenizer, prompt, max_len=150, 
                            temperature=0.8, device=device)
            print(f"\n  Prompt: '{prompt}'")
            print(f"  Output: {output[:200]}")
    
    print(f"\n  Training complete. Checkpoints at: {args.save_dir}")
    print(f"  To generate: python3 train_general_relevance.py --generate --prompt 'your prompt'")


if __name__ == '__main__':
    main()
