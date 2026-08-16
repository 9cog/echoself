#!/usr/bin/env python3
"""Train the 2-3-5 TernaryQuinary Transformer on the cognitive cycle corpus."""
import sys
sys.path.insert(0, '/workspace/echoself')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import time
from tqdm import tqdm

from netrain.models.ternary_quinary import TernaryQuinaryConfig, TernaryQuinaryTransformer

# --- Configuration ---
CORPUS_FILE = "/workspace/echoself/data/deep_echo/corpus_235.txt"
CHECKPOINT_DIR = "/workspace/echoself/checkpoints_235"
BATCH_SIZE = 8
BLOCK_SIZE = 512
MAX_STEPS = 10000
EVAL_INTERVAL = 500
SAVE_INTERVAL = 2000
LR = 3e-4
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1


class CognitiveCorpusDataset(Dataset):
    """Dataset that loads the 2-3-5 cognitive cycle corpus."""

    def __init__(self, text_file: str, block_size: int = 512):
        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Add cognitive phase tokens
        phase_tokens = [
            "<|perceive|>", "<|feel|>", "<|think|>", "<|remember|>",
            "<|interpret|>", "<|strategize|>", "<|evaluate|>",
            "<|gesture|>", "<|speak|>"
        ]
        self.tokenizer.add_tokens(phase_tokens)

        # Tokenize the entire corpus
        with open(text_file, 'r') as f:
            text = f.read()

        print(f"Tokenizing corpus ({len(text)} chars)...")
        self.tokens = self.tokenizer.encode(text)
        print(f"Total tokens: {len(self.tokens):,}")

        self.block_size = block_size

    def __len__(self):
        return max(1, (len(self.tokens) - self.block_size) // (self.block_size // 2))

    def __getitem__(self, idx):
        start = idx * (self.block_size // 2)
        end = start + self.block_size + 1
        chunk = self.tokens[start:end]

        # Pad if necessary
        if len(chunk) < self.block_size + 1:
            chunk = chunk + [self.tokenizer.eos_token_id] * (self.block_size + 1 - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr_ratio=0.1):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * decay_ratio))
    return max_lr * min_lr_ratio + coeff * (max_lr - max_lr * min_lr_ratio)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Initialize model
    config = TernaryQuinaryConfig(block_size=BLOCK_SIZE)
    model = TernaryQuinaryTransformer(config).to(device)

    # Dataset
    dataset = CognitiveCorpusDataset(CORPUS_FILE, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95)
    )

    # Training loop
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    best_loss = float('inf')
    losses = []
    start_time = time.time()

    print(f"\nStarting training: {MAX_STEPS} steps, batch_size={BATCH_SIZE}")
    print(f"Dataset: {len(dataset)} examples, {len(dataset.tokens):,} tokens")

    while step < MAX_STEPS:
        for x, y in dataloader:
            if step >= MAX_STEPS:
                break

            x, y = x.to(device), y.to(device)

            # Update learning rate
            lr = get_lr(step, WARMUP_STEPS, MAX_STEPS, LR)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                output = model(x, targets=y)
                loss = output['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            step += 1

            # Logging
            if step % 50 == 0:
                avg_loss = sum(losses[-50:]) / len(losses[-50:])
                elapsed = time.time() - start_time
                tokens_per_sec = (step * BATCH_SIZE * BLOCK_SIZE) / elapsed
                print(f"Step {step}/{MAX_STEPS} | Loss: {avg_loss:.4f} | LR: {lr:.6f} | "
                      f"Tokens/s: {tokens_per_sec:.0f} | Elapsed: {elapsed/60:.1f}m")

            # Save checkpoint
            if step % SAVE_INTERVAL == 0:
                ckpt_path = Path(CHECKPOINT_DIR) / f"ckpt_step_{step}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'config': config.__dict__,
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_path = Path(CHECKPOINT_DIR) / "best_model.pt"
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'config': config.__dict__,
                        'loss': best_loss,
                    }, best_path)
                    print(f"  New best model! Loss: {best_loss:.4f}")

    # Final save
    final_path = Path(CHECKPOINT_DIR) / "final_model.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__,
        'loss': losses[-1] if losses else 0,
        'all_losses': losses,
    }, final_path)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total steps: {step}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()
