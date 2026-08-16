#!/usr/bin/env python3
"""Diagnose the echoself repository health."""
import ast
import os
import sys
from pathlib import Path

root = Path(".")
issues = []
ok = []

# 1. Check all Python files for syntax errors
print("=== SYNTAX CHECK ===")
py_files = list(root.rglob("*.py"))
py_files = [f for f in py_files if '.git' not in str(f) and 'node_modules' not in str(f)]
syntax_errors = []
for f in py_files:
    try:
        with open(f) as fh:
            ast.parse(fh.read())
    except SyntaxError as e:
        syntax_errors.append((str(f), str(e)))
        
if syntax_errors:
    print(f"  FAIL: {len(syntax_errors)} files with syntax errors:")
    for path, err in syntax_errors:
        print(f"    - {path}: {err}")
    issues.append(f"{len(syntax_errors)} syntax errors")
else:
    print(f"  OK: All {len(py_files)} Python files parse cleanly")
    ok.append("syntax")

# 2. Check if training data exists
print("\n=== TRAINING DATA ===")
data_dirs = ["data/deep_echo", "NanEcho/persona_corpus", "echoself/data"]
for d in data_dirs:
    p = root / d
    if p.exists():
        files = list(p.rglob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        print(f"  {d}: {len(files)} files, {total_size/1024:.1f} KB")
    else:
        print(f"  {d}: MISSING")
        issues.append(f"Missing data dir: {d}")

# 3. Check import chains
print("\n=== IMPORT CHAIN CHECK ===")
critical_imports = [
    ("netrain.models", "netrain/models/__init__.py"),
    ("netrain.training", "netrain/training/__init__.py"),
    ("netrain.data", "netrain/data/__init__.py"),
    ("NanEcho", "NanEcho/__init__.py"),
]
for mod_name, init_path in critical_imports:
    if (root / init_path).exists():
        print(f"  {mod_name}: OK ({init_path} exists)")
    else:
        print(f"  {mod_name}: MISSING {init_path}")
        issues.append(f"Missing module: {mod_name}")

# 4. Check model file completeness
print("\n=== MODEL ARCHITECTURE ===")
model_files = [
    "netrain/models/deep_tree_echo.py",
    "netrain/models/layers.py",
    "nanecho_model.py",
    "train.py",
    "train_nanecho.py",
]
for mf in model_files:
    p = root / mf
    if p.exists():
        size = p.stat().st_size
        with open(p) as f:
            lines = f.readlines()
        # Check for placeholder/stub indicators
        content = "".join(lines)
        has_pass_only = content.count("pass") > content.count("def ") * 0.5
        has_todo = "TODO" in content or "FIXME" in content or "NotImplementedError" in content
        status = "OK"
        notes = []
        if has_todo:
            notes.append("has TODOs")
        if has_pass_only:
            notes.append("many pass stubs")
        note_str = f" ({', '.join(notes)})" if notes else ""
        print(f"  {mf}: {len(lines)} lines, {size/1024:.1f} KB{note_str}")
    else:
        print(f"  {mf}: MISSING")
        issues.append(f"Missing model file: {mf}")

# 5. Check netrain.yml config
print("\n=== CONFIGURATION ===")
config_path = root / "netrain.yml"
if config_path.exists():
    import yaml
    with open(config_path) as f:
        try:
            config = yaml.safe_load(f)
            hw = config.get('hardware', {})
            model = config.get('model', {}).get('architecture', {})
            print(f"  Device: {hw.get('device', 'unknown')}")
            print(f"  Model: {model.get('n_layers', '?')} layers, {model.get('n_embd', '?')} embd, {model.get('n_heads', '?')} heads")
            print(f"  Block size: {model.get('block_size', '?')}")
            print(f"  Max steps: {config.get('training', {}).get('max_steps', '?')}")
            if hw.get('device') == 'cpu':
                issues.append("Config set to CPU - needs GPU for real training")
        except Exception as e:
            print(f"  PARSE ERROR: {e}")
            issues.append("netrain.yml parse error")
else:
    print("  netrain.yml: MISSING")
    issues.append("Missing netrain.yml")

# 6. Check GitHub Actions workflows
print("\n=== GITHUB ACTIONS ===")
wf_dir = root / ".github" / "workflows"
if wf_dir.exists():
    for wf in sorted(wf_dir.iterdir()):
        if wf.suffix == '.yml':
            with open(wf) as f:
                content = f.read()
            uses_gpu = "gpu" in content.lower() or "cuda" in content.lower()
            uses_vast = "vast" in content.lower()
            runs_on = "self-hosted" if "self-hosted" in content else "ubuntu-latest"
            print(f"  {wf.name}: runs-on={runs_on}, GPU={uses_gpu}, Vast={uses_vast}")
else:
    print("  No workflows found")

# 7. Summary
print("\n" + "=" * 60)
print("DIAGNOSIS SUMMARY")
print("=" * 60)
if issues:
    print(f"\n  ISSUES FOUND: {len(issues)}")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")
else:
    print("\n  All checks passed!")

print(f"\n  Total Python files: {len(py_files)}")
print(f"  Total repo size: {sum(f.stat().st_size for f in root.rglob('*') if f.is_file() and '.git' not in str(f)) / 1024 / 1024:.1f} MB")
