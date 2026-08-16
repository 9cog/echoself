# Training Progress Log

This directory contains progress logs from Agent-Neuro supervised training sessions.

## 📈 Cumulative Training Progress

**Total Iterations Completed**: 200
**Total Sessions**: 1
**Best Validation Loss**: 1.9703
**Current Model Scale**: n_layer=4, n_embd=256

### How Cumulative Training Works

Training progress accumulates across multiple sessions:

1. **Session 1**: iterations 0 → 200 (model: 4 layers)
2. **Session 2**: iterations 200 → 400 (scales up if threshold reached)
3. **Session 3**: iterations 400 → 600 (continues accumulating)
4. ...and so on indefinitely

### Model Scaling Schedule

Model parameters automatically scale based on total cumulative iterations:

| Total Iterations | n_layer | n_embd | Learning Rate | Batch Size |
| ---------------- | ------- | ------ | ------------- | ---------- |
| 0 - 500          | 4       | 256    | 2e-4          | 2          |
| 500 - 2000       | 6       | 384    | 1e-4          | 4          |
| 2000 - 10000     | 8       | 512    | 6e-5          | 6          |
| 10000 - 50000    | 12      | 768    | 3e-5          | 8          |
| 50000+           | 16      | 1024   | 1e-5          | 12         |

## Latest Session

- **Orchestrator**: Agent-Neuro (Chaotic Cognitive VTuber Framework)
- **Persona Enforced**: Deep Tree Echo
- **Training Mode**: Standard Training
- **Output Directory**: out-nanecho-ci
- **Timestamp**: 2026-06-11 11:32:44 UTC

## Supervision Phases

1. Data Preparation - Supervised ✓
2. Training - Supervised ✓
3. Evaluation - Supervised ✓

## Files

### Critical Files

- `cumulative_progress.json` - **CRITICAL**: Tracks total iterations across ALL sessions
- `cumulative_progress_manager.py` - Python utility for managing cumulative progress

### Session Logs

- `data_prep_supervision.json` - Data preparation phase supervision log
- `training_supervision.json` - Training phase supervision log
- `evaluation_supervision.json` - Evaluation phase supervision log
- `session_summary.json` - Complete session summary

## Usage

### Check Current Progress

```bash
python .training-progress/cumulative_progress_manager.py summary
```

### Get Next Session Parameters

```bash
python .training-progress/cumulative_progress_manager.py next_session --session-iters 200
```

### Get GitHub Actions Outputs

```bash
python .training-progress/cumulative_progress_manager.py github_outputs --session-iters 200
```

### Manual Progress Update

```bash
python .training-progress/cumulative_progress_manager.py update \
  --iterations 200 \
  --val-loss 1.97 \
  --train-loss 2.22 \
  --workflow agent-neuro-train.yml \
  --trigger manual
```
