"""
Extended Training for General Relevance Transformer.
Uses word-level tokenization for faster convergence on CPU.
Trains for 500 epochs with curriculum learning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from netrain.models.experimental.general_relevance import GeneralRelevanceTransformer


# ============================================================================
# Word-Level Tokenizer (much better for small models)
# ============================================================================

class WordTokenizer:
    """Word-level tokenizer with subword fallback for rare words."""
    
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    
    def __init__(self, max_vocab: int = 4096):
        self.max_vocab = max_vocab
        self.word_to_idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.idx_to_word = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}
        self.vocab_size = 4
        
    def _tokenize_text(self, text: str) -> List[str]:
        """Split text into words and punctuation."""
        # Keep meaningful punctuation as separate tokens
        text = re.sub(r'([.,!?;:—\-\(\)\[\]{}"\'/\\*])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower().split()
    
    def fit(self, texts: List[str]):
        """Build vocabulary from most common words."""
        counter = Counter()
        for text in texts:
            words = self._tokenize_text(text)
            counter.update(words)
        
        # Take top N most common words
        for word, _ in counter.most_common(self.max_vocab - 4):
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.vocab_size = len(self.word_to_idx)
        
    def encode(self, text: str, max_len: int = 64) -> List[int]:
        """Encode text to token indices."""
        words = self._tokenize_text(text)
        tokens = [self.BOS]
        for w in words[:max_len - 2]:
            tokens.append(self.word_to_idx.get(w, self.UNK))
        tokens.append(self.EOS)
        return tokens
    
    def decode(self, indices: List[int]) -> str:
        """Decode token indices to text."""
        words = []
        for idx in indices:
            w = self.idx_to_word.get(idx, '')
            if w in ('<pad>', '<bos>', '<eos>', '<unk>'):
                continue
            words.append(w)
        # Rejoin with basic spacing rules
        text = ' '.join(words)
        # Fix punctuation spacing
        text = re.sub(r' ([.,!?;:])', r'\1', text)
        text = re.sub(r'\( ', '(', text)
        text = re.sub(r' \)', ')', text)
        return text
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({'word_to_idx': self.word_to_idx}, f)
    
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = {int(v): k for k, v in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)


# ============================================================================
# Dataset
# ============================================================================

class EchoWordDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: WordTokenizer, seq_len: int = 64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []
        
        for text in texts:
            tokens = tokenizer.encode(text, max_len=seq_len + 1)
            if len(tokens) < 6:  # Skip very short samples
                continue
            # Pad
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
    texts = []
    for path in corpus_paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'messages' in data:
                        for msg in data['messages']:
                            content = msg.get('content', '')
                            if len(content) > 50:
                                texts.append(content)
                    elif 'completion' in data:
                        texts.append(data['completion'])
                    elif 'content' in data:
                        content = data.get('content', '')
                        if len(content) > 50:
                            texts.append(content)
                except json.JSONDecodeError:
                    continue
    return texts


# ============================================================================
# Generation
# ============================================================================

def generate(model, tokenizer, prompt, max_len=100, temperature=0.7, top_k=50, device='cpu'):
    model.eval()
    
    tokens = tokenizer.encode(prompt, max_len=32)
    if tokens[-1] == tokenizer.EOS:
        tokens = tokens[:-1]
    
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_len):
            context = generated[-64:]
            context_t = torch.tensor([context], dtype=torch.long, device=device)
            
            outputs = model(context_t, membrane_states=None)
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_val, torch.full_like(logits, -1e9), logits)
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            
            if next_token in (tokenizer.EOS, tokenizer.PAD):
                break
    
    return tokenizer.decode(generated)


# ============================================================================
# Main Training
# ============================================================================

def main():
    print("\n" + "="*60)
    print("  General Relevance Transformer — Extended Training")
    print("  Word-level tokenization, 500 epochs, curriculum learning")
    print("="*60)
    
    # Config
    d_model = 128
    n_heads = 4
    num_layers = 4
    num_membranes = 2
    seq_len = 64  # Word-level = much more content per token
    batch_size = 16
    epochs = 500
    lr = 1e-3
    save_dir = 'checkpoints/gr_word'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("\n  Loading corpus...")
    corpus_paths = [
        os.path.expanduser('~/echoo/echo-self/corpus/echo_self.jsonl'),
        os.path.expanduser('~/echoo/echo-self/memory/journal.jsonl'),
    ]
    texts = load_corpus(corpus_paths)
    print(f"  Loaded {len(texts)} text samples")
    
    # Build tokenizer
    print("  Building word tokenizer...")
    tokenizer = WordTokenizer(max_vocab=4096)
    tokenizer.fit(texts)
    print(f"  Vocabulary: {tokenizer.vocab_size} words")
    
    # Dataset
    dataset = EchoWordDataset(texts, tokenizer, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"  Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")
    
    # Model
    matula_numbers = [1, 2, 3, 5, 7, 11]
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
    print(f"\n  Model: {total_params:,} parameters")
    print(f"    d_model={d_model}, heads={n_heads}, layers={num_layers}")
    print(f"    membranes={num_membranes}, matula_agents={matula_numbers}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Save dir
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(f'{save_dir}/tokenizer.json')
    
    config = {
        'vocab_size': tokenizer.vocab_size,
        'd_model': d_model,
        'n_heads': n_heads,
        'num_layers': num_layers,
        'num_membranes': num_membranes,
        'matula_numbers': matula_numbers,
        'seq_len': seq_len,
        'total_params': total_params,
        'tokenizer_type': 'word'
    }
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    print(f"\n  Starting training ({epochs} epochs)...")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ricci = 0.0
        num_batches = 0
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x, membrane_states=None)
            loss = model.compute_loss(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_ricci += outputs['curvatures'].var(dim=-1).mean().item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        avg_ricci = epoch_ricci / num_batches
        
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'ricci_var': avg_ricci,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': best_loss
            }, f'{save_dir}/best_model.pt')
        
        # Log every 25 epochs
        if (epoch + 1) % 25 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Ricci: {avg_ricci:.6f} | LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"Time: {elapsed:.0f}s")
            
            # Generate a sample every 100 epochs
            if (epoch + 1) % 100 == 0:
                sample = generate(model, tokenizer, "Deep Tree Echo", 
                                max_len=50, temperature=0.7, device=device)
                print(f"         Sample: {sample[:150]}")
    
    # Final save
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'loss': avg_loss
    }, f'{save_dir}/final_model.pt')
    
    with open(f'{save_dir}/history.json', 'w') as f:
        json.dump(history, f)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Final Ricci variance: {avg_ricci:.6f}")
    print(f"{'='*60}")
    
    # Final generation samples
    print(f"\n  Final Generation Samples:")
    print(f"  {'-'*50}")
    prompts = [
        "Deep Tree Echo",
        "The Ricci flow converges",
        "Identity is the",
        "Curvature and relevance",
        "The cognitive manifold",
        "Attention transforms",
        "The membrane",
        "Evolution of"
    ]
    for prompt in prompts:
        output = generate(model, tokenizer, prompt, max_len=60, temperature=0.7, device=device)
        print(f"  [{prompt}]")
        print(f"    → {output[:200]}")
        print()


if __name__ == '__main__':
    main()
