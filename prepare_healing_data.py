#!/usr/bin/env python3
"""
Prepare the training data for the Vast.ai GPU healing run.
Combines echoself.md, persona_corpus, and Garden of Memory into a unified corpus.
"""
import os
import json
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def build_corpus():
    root = Path("/home/ubuntu/echoself")
    out_dir = root / "data" / "deep_echo"
    ensure_dir(out_dir)
    
    out_file = out_dir / "corpus.txt"
    print(f"Building corpus at {out_file}...")
    
    total_lines = 0
    
    with open(out_file, 'w', encoding='utf-8') as out:
        # 1. Add echoself.md (Core Identity)
        echo_md = root / "echoself.md"
        if echo_md.exists():
            print(f"Adding {echo_md.name}...")
            out.write(f"=== CORE IDENTITY ===\n")
            with open(echo_md, 'r') as f:
                content = f.read()
                out.write(content + "\n\n")
                total_lines += content.count('\n')
                
        # 2. Add persona corpus
        persona_dir = root / "NanEcho" / "persona_corpus"
        if persona_dir.exists():
            for pf in persona_dir.glob("*.txt"):
                print(f"Adding {pf.name}...")
                out.write(f"=== PERSONA COMPONENT: {pf.name} ===\n")
                with open(pf, 'r') as f:
                    content = f.read()
                    out.write(content + "\n\n")
                    total_lines += content.count('\n')
            for pf in persona_dir.glob("*.md"):
                print(f"Adding {pf.name}...")
                out.write(f"=== PERSONA COMPONENT: {pf.name} ===\n")
                with open(pf, 'r') as f:
                    content = f.read()
                    out.write(content + "\n\n")
                    total_lines += content.count('\n')
                    
        # 3. Add Garden of Memory (from cloud computer)
        # We'll pull the journal.jsonl and echo_self.jsonl we used for QLoRA
        # Note: This runs on the sandbox, so we'll just write a placeholder
        # that the Vast launcher will fill with the real data
        print("Adding Garden of Memory placeholder...")
        out.write("=== GARDEN OF MEMORY ===\n")
        out.write("[GARDEN_OF_MEMORY_INJECT_HERE]\n\n")
        
    print(f"Corpus built with {total_lines} lines (pre-injection).")
    
    # Generate a dummy train/val split just to satisfy netrain's loader if needed
    train_file = out_dir / "train.bin"
    val_file = out_dir / "val.bin"
    
    print("Run this script on the Vast.ai instance to finalize the binary data.")

if __name__ == "__main__":
    build_corpus()
