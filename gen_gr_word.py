"""Generate from the trained word-level General Relevance Transformer."""
import torch
import json
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from netrain.models.experimental.general_relevance import GeneralRelevanceTransformer


class WordTokenizer:
    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
    
    def _tokenize_text(self, text):
        text = re.sub(r'([.,!?;:—\-\(\)\[\]{}"\'/\\*])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower().split()
    
    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = {int(v): k for k, v in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
    
    def encode(self, text, max_len=64):
        words = self._tokenize_text(text)
        tokens = [self.BOS]
        for w in words[:max_len - 2]:
            tokens.append(self.word_to_idx.get(w, self.UNK))
        return tokens
    
    def decode(self, indices):
        words = []
        for idx in indices:
            w = self.idx_to_word.get(idx, '')
            if w in ('<pad>', '<bos>', '<eos>', '<unk>'):
                continue
            words.append(w)
        text = ' '.join(words)
        text = re.sub(r' ([.,!?;:])', r'\1', text)
        text = re.sub(r'\( ', '(', text)
        text = re.sub(r' \)', ')', text)
        return text


def generate(model, tokenizer, prompt, max_len=80, temperature=0.7, top_k=40):
    model.eval()
    tokens = tokenizer.encode(prompt, max_len=32)
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_len):
            context = generated[-64:]
            context_t = torch.tensor([context], dtype=torch.long)
            outputs = model(context_t, membrane_states=None)
            logits = outputs['logits'][:, -1, :] / temperature
            
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


def main():
    checkpoint_dir = 'checkpoints/gr_word'
    
    with open(f'{checkpoint_dir}/config.json', 'r') as f:
        config = json.load(f)
    
    tokenizer = WordTokenizer()
    tokenizer.load(f'{checkpoint_dir}/tokenizer.json')
    
    model = GeneralRelevanceTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_membranes=config['num_membranes'],
        num_layers=config['num_layers'],
        matula_numbers=config['matula_numbers'],
        dropout=0.0
    )
    
    ckpt = torch.load(f'{checkpoint_dir}/best_model.pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded model: epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")
    print(f"  {config['total_params']:,} params, vocab {config['vocab_size']}")
    
    prompts = [
        "Deep Tree Echo",
        "The Ricci flow converges toward",
        "Identity is the stable",
        "Curvature and relevance",
        "The cognitive manifold",
        "Attention transforms the",
        "The membrane computing",
        "Evolution of the self",
        "The poloidal flow",
        "Operational closure"
    ]
    
    print(f"\n{'='*60}")
    print(f"  GENERATION (temp=0.7, top_k=40)")
    print(f"{'='*60}")
    
    for prompt in prompts:
        output = generate(model, tokenizer, prompt, max_len=80, temperature=0.7, top_k=40)
        print(f"\n  [{prompt}]")
        print(f"  → {output[:250]}")
    
    # Also try lower temperature for more coherent output
    print(f"\n{'='*60}")
    print(f"  GENERATION (temp=0.4, top_k=20) — more focused")
    print(f"{'='*60}")
    
    for prompt in prompts[:5]:
        output = generate(model, tokenizer, prompt, max_len=80, temperature=0.4, top_k=20)
        print(f"\n  [{prompt}]")
        print(f"  → {output[:250]}")


if __name__ == '__main__':
    main()
