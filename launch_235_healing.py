#!/usr/bin/env python3
"""
Vast.ai Launcher for the 2-3-5 EchoSelf Healing Run
====================================================

Launches a GPU instance, uploads the repo, trains the TernaryQuinaryTransformer
on the cognitive cycle corpus, and harvests the trained checkpoint.

Usage:
    python3 launch_235_healing.py --launch     # Find and launch instance
    python3 launch_235_healing.py --status     # Check instance status
    python3 launch_235_healing.py --train      # SSH in and start training
    python3 launch_235_healing.py --harvest    # Download trained checkpoint
    python3 launch_235_healing.py --destroy    # Destroy instance
    python3 launch_235_healing.py --dry-run    # Show what would happen
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Configuration
REPO_DIR = Path("/home/ubuntu/echoself")
MIN_BALANCE = 1.0  # Minimum $1 balance required
MIN_VRAM_GB = 16   # RTX 3090 has 24GB
MAX_PRICE_PER_HR = 0.50
IMAGE = "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel"
DISK_GB = 30


def get_api_key():
    """Get Vast.ai API key from environment."""
    # Try loading from echoo .env
    env_file = Path("/mnt/cx5l1vpt0nguxqdwvte64scrh/ubuntu/echoo/.env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("VAST_API_KEY="):
                    return line.strip().split("=", 1)[1].strip('"').strip("'")
    key = os.environ.get("VAST_API_KEY")
    if not key:
        print("ERROR: VAST_API_KEY not found. Source ~/echoo/.env first.")
        sys.exit(1)
    return key


def api_call(method, endpoint, api_key, data=None):
    """Make a Vast.ai API call."""
    import requests
    url = f"https://console.vast.ai/api/v0/{endpoint}"
    headers = {"Authorization": f"Bearer {api_key}"}
    if method == "GET":
        r = requests.get(url, headers=headers)
    elif method == "PUT":
        r = requests.put(url, headers=headers, json=data)
    elif method == "DELETE":
        r = requests.delete(url, headers=headers)
    else:
        r = requests.post(url, headers=headers, json=data)
    return r.json() if r.text else {}


def check_balance(api_key):
    """Check account balance."""
    data = api_call("GET", "users/current/", api_key)
    balance = data.get("credit", 0)
    print(f"Balance: ${balance:.2f}")
    if balance < MIN_BALANCE:
        print(f"ERROR: Balance ${balance:.2f} below minimum ${MIN_BALANCE:.2f}")
        sys.exit(1)
    return balance


def find_offer(api_key):
    """Find the cheapest suitable GPU offer."""
    import requests
    # Vast.ai v0 API: GET /bundles/ returns all offers, filter client-side
    url = "https://console.vast.ai/api/v0/bundles/"
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers)
    all_offers = r.json().get("offers", [])

    # Filter: >=16GB VRAM, affordable, sufficient disk
    TARGET_GPUS = ["RTX 3090", "RTX 4090", "RTX 4080S", "RTX 5070", "RTX 5070 Ti",
                   "RTX 5080", "RTX 5090", "Tesla V100", "L4", "RTX 4070S Ti"]
    offers = [
        o for o in all_offers
        if o.get("gpu_ram", 0) >= MIN_VRAM_GB * 1024
        and o.get("dph_base", 999) <= MAX_PRICE_PER_HR
        and o.get("disk_space", 0) >= DISK_GB
        and o.get("gpu_name", "") in TARGET_GPUS
    ]
    # Sort by price
    offers.sort(key=lambda x: x.get("dph_base", 999))

    if not offers:
        print(f"ERROR: No suitable offers found under ${MAX_PRICE_PER_HR}/hr")
        # Show what IS available
        affordable = sorted(
            [o for o in all_offers if o.get("gpu_ram", 0) >= MIN_VRAM_GB * 1024],
            key=lambda x: x.get("dph_base", 999)
        )[:5]
        if affordable:
            print("  Available options:")
            for o in affordable:
                print(f"    {o['gpu_name']:20s} | ${o['dph_base']:.3f}/hr | {o.get('gpu_ram',0)//1024}GB")
        sys.exit(1)

    best = offers[0]
    print(f"Best offer: {best['gpu_name']} ({best.get('gpu_ram', 0)//1024}GB VRAM)")
    print(f"  Price: ${best['dph_base']:.3f}/hr")
    print(f"  Offer ID: {best['id']}")
    print(f"  Disk: {best.get('disk_space', 0):.0f}GB")
    return best


def launch_instance(api_key, offer):
    """Launch a Vast.ai instance."""
    onstart_script = """#!/bin/bash
set -e
apt-get update -qq && apt-get install -y -qq git rsync
pip install -q transformers tokenizers datasets tqdm wandb
echo "READY" > /workspace/.instance_ready
"""
    data = {
        "client_id": "echoself_235_healing",
        "image": IMAGE,
        "disk": DISK_GB,
        "onstart": onstart_script,
    }
    result = api_call("PUT", f"asks/{offer['id']}/", api_key, data)
    contract_id = result.get("new_contract")
    if not contract_id:
        print(f"ERROR: Launch failed: {result}")
        sys.exit(1)
    print(f"Instance launched! Contract ID: {contract_id}")
    return contract_id


def get_instance_info(api_key):
    """Get running instance info."""
    data = api_call("GET", "instances/", api_key)
    instances = data.get("instances", [])
    if not instances:
        print("No running instances.")
        return None
    inst = instances[0]
    print(f"Instance {inst['id']}:")
    print(f"  GPU: {inst.get('gpu_name', '?')}")
    print(f"  Status: {inst.get('actual_status', '?')}")
    print(f"  SSH: ssh -p {inst.get('ssh_port', '?')} root@{inst.get('ssh_host', '?')}")
    print(f"  Cost so far: ${inst.get('total_cost', 0):.3f}")
    return inst


def generate_train_script():
    """Generate the training script to run on the Vast.ai instance."""
    script = '''#!/usr/bin/env python3
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

    print(f"\\nStarting training: {MAX_STEPS} steps, batch_size={BATCH_SIZE}")
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
    print(f"\\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total steps: {step}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()
'''
    script_path = REPO_DIR / "train_235_gpu.py"
    with open(script_path, 'w') as f:
        f.write(script)
    print(f"Training script written to: {script_path}")
    return script_path


def main():
    parser = argparse.ArgumentParser(description="Vast.ai 2-3-5 Healing Run Launcher")
    parser.add_argument("--launch", action="store_true", help="Find and launch instance")
    parser.add_argument("--status", action="store_true", help="Check instance status")
    parser.add_argument("--train", action="store_true", help="Start training on instance")
    parser.add_argument("--harvest", action="store_true", help="Download checkpoint")
    parser.add_argument("--destroy", action="store_true", help="Destroy instance")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    args = parser.parse_args()

    api_key = get_api_key()

    if args.dry_run:
        print("=== DRY RUN ===")
        balance = check_balance(api_key)
        offer = find_offer(api_key)
        print(f"\nWould launch {offer['gpu_name']} at ${offer['dph_base']:.3f}/hr")
        print(f"Estimated training time: 1-3 hours")
        print(f"Estimated cost: ${offer['dph_base'] * 2:.2f} - ${offer['dph_base'] * 4:.2f}")
        generate_train_script()
        return

    if args.launch:
        balance = check_balance(api_key)
        offer = find_offer(api_key)
        print(f"\nLaunching instance...")
        contract_id = launch_instance(api_key, offer)
        generate_train_script()
        print(f"\nNext steps:")
        print(f"  1. Wait for instance to boot: python3 launch_235_healing.py --status")
        print(f"  2. Sync repo: rsync -avz echoself/ root@host:/workspace/echoself/")
        print(f"  3. Start training: python3 launch_235_healing.py --train")
        return

    if args.status:
        get_instance_info(api_key)
        return

    if args.destroy:
        inst = get_instance_info(api_key)
        if inst:
            api_call("DELETE", f"instances/{inst['id']}/", api_key)
            print("Instance destroyed.")
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
