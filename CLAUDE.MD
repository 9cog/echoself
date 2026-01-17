# CLAUDE.MD - EchoSelf Development Guide

This file provides essential context for Claude Code when working with the EchoSelf/Deep Tree Echo project.

## Project Overview

**EchoSelf** is an advanced distributed AGI cognitive architecture featuring:

- **Deep Tree Echo**: Primary AI persona with adaptive cognition
- **NanEcho**: GPT-2-based model trained on persona-specific data
- **Echo State Networks**: Temporal pattern recognition and adaptive learning
- **Hypergraph Pattern Encoding**: Neural-symbolic cognitive associations
- **Toroidal Dual-Persona Processing**: Interdependent cognitive streams
- **EchoLayla**: Character-based AI assistant integration

## Quick Reference

### Key Directories

```
/home/user/echoself/
├── NanEcho/              # GPT-2 training system (primary)
│   ├── prepare_nanecho.py    # Data preparation with persona weighting
│   ├── server.py             # Inference server
│   └── evaluation/           # Fidelity testing
├── src/services/         # Core TypeScript services
│   ├── deepTreeEchoService.ts
│   ├── toroidalCognitiveService.ts
│   ├── adaptiveFeedbackService.ts
│   └── feedback/             # Hypergraph feedback system
├── app/                  # Remix web application
│   ├── routes/               # Application routes
│   └── services/             # App-level services (echolayla, etc.)
├── .github/
│   ├── workflows/        # 12 CI/CD workflows
│   └── agents/           # 80+ agent personality definitions
├── echo/                 # Echo State Network modules
├── chatbot/              # Standalone web chatbot
├── .training-progress/   # Training session logs
└── .maintenance-logs/    # Automated maintenance reports
```

### Essential Commands

```bash
# Development
npm run dev              # Start development server
npm run build            # Production build (verified: ~7s)
npm start                # Run production server

# Code Quality
npm run lint             # ESLint check (13 warnings acceptable)
npm run lint:fix         # Auto-fix ESLint issues
npm run format           # Prettier formatting
npm run format:check     # Check formatting
npm run typecheck        # TypeScript compilation

# Training (NanEcho)
cd NanEcho
python prepare_nanecho.py --echo_depth=5 --persona_weight=0.9
python server.py         # Start inference server

# Dependency Management
npm run audit:deps       # Run dependency audit
npm audit                # Check for vulnerabilities
```

### Repair Sequence

When the project needs repair (e.g., after clone or dependency issues):

```bash
# 1. Install dependencies
npm install

# 2. Verify lint (should show 13 warnings, 0 errors)
npm run lint

# 3. Verify TypeScript compilation (should pass)
npm run typecheck

# 4. Verify build (should complete in ~7s)
npm run build
```

### Configuration Files

| File                  | Purpose                        |
| --------------------- | ------------------------------ |
| `nanecho_config.json` | NanEcho model hyperparameters  |
| `package.json`        | Node.js dependencies           |
| `pyproject.toml`      | Python package config          |
| `.env.example`        | Environment variables template |
| `.eslintrc.cjs`       | ESLint configuration (legacy)  |
| `tsconfig.json`       | TypeScript configuration       |
| `vite.config.ts`      | Vite bundler configuration     |

## Architecture Concepts

### Deep Tree Echo Persona Dimensions

1. **Cognitive** - Analytical reasoning (weight: 0.15)
2. **Introspective** - Self-examination (weight: 0.15)
3. **Adaptive** - Threshold adjustment (weight: 0.15)
4. **Recursive** - Multi-level processing (weight: 0.15)
5. **Synergistic** - Emergent properties (weight: 0.10)
6. **Holographic** - Comprehensive modeling (weight: 0.10)
7. **Neural-Symbolic** - Hybrid reasoning (weight: 0.10)
8. **Dynamic** - Continuous evolution (weight: 0.10)

### Adaptive Attention Formula

```
threshold = 0.5 + (cognitive_load * 0.3) - (recent_activity * 0.2)
```

### Training Modes

- **CI Mode**: Quick validation (4 layers, 200 iterations)
- **Full Mode**: Complete training (12 layers, 50000 iterations)
- **Relentless Mode**: Continuous persona reinforcement (scheduled every 4 hours)

## Cumulative Training Safeguards

**CRITICAL**: Training is designed to NEVER start from scratch accidentally.

### Checkpoint Guardian System

The project implements a multi-layer checkpoint protection system:

1. **Checkpoint Restoration Priority**:
   - `.training-progress/checkpoints/latest_checkpoint.pt` (highest priority)
   - Downloaded artifacts from previous workflow runs
   - GitHub Actions cache
   - Any valid checkpoint in backup locations

2. **Multi-Location Model Backup**:
   Every training run saves checkpoints to multiple locations:
   - `.training-progress/checkpoints/` - Committed to repository
   - `.training-progress/cache/` - GitHub Actions cache
   - `/tmp/nanecho-checkpoint-backup/` - Uploaded as artifact
   - Workflow artifacts with 30-90 day retention

3. **Fresh Start Protection**:
   - `force_fresh_start` requires explicit confirmation string
   - Data preparation failures do NOT create minimal fallback data
   - If no checkpoint exists AND data prep fails, workflow errors out
   - This prevents accidental loss of training progress

### Backup Manifest

Each checkpoint backup includes a manifest with:
```json
{
  "timestamp": "20260117_143022",
  "iteration": 5000,
  "val_loss": 2.3456,
  "workflow_run": "123",
  "backup_count": 3,
  "locations": [
    ".training-progress/checkpoints/latest_checkpoint.pt",
    ".training-progress/cache/ckpt.pt",
    "artifacts"
  ]
}
```

### Using Checkpoint Guardian Script

```bash
# Restore best checkpoint for cumulative training
python scripts/checkpoint_guardian.py --output-dir out-nanecho --action restore

# Backup a checkpoint to multiple locations
python scripts/checkpoint_guardian.py --output-dir out-nanecho --action backup \
  --checkpoint out-nanecho/ckpt.pt --iteration 5000 --val-loss 2.345

# Verify checkpoint availability
python scripts/checkpoint_guardian.py --output-dir out-nanecho --action verify

# Clean up old checkpoints (keeps 5 best)
python scripts/checkpoint_guardian.py --output-dir out-nanecho --action cleanup --keep-count 5

# DANGEROUS: Allow fresh start (only for first-ever training)
python scripts/checkpoint_guardian.py --output-dir out-nanecho --action restore --allow-fresh-start
```

## Workflow Reference

### Training Workflows

| Workflow                        | Purpose                         |
| ------------------------------- | ------------------------------- |
| `netrain.yml`                   | Primary NanEcho training        |
| `agent-neuro-train.yml`         | Agent-Neuro supervised training |
| `netrain-cached.yml`            | Cached/incremental training     |
| `nctrain.yml`                   | NanCog training                 |
| `ncrun.yml`                     | NanCog run/inference            |
| `nerun.yml`                     | NanEcho run/inference           |
| `test-training-persistence.yml` | Training persistence tests      |

### Quality Workflows

| Workflow                    | Purpose                       |
| --------------------------- | ----------------------------- |
| `automated-quality.yml`     | Linting, formatting, security |
| `deno.yml`                  | Deno-specific checks          |
| `disk-space-management.yml` | CI disk management            |
| `deploy-chatbot.yml`        | Deploy standalone chatbot     |

## Current Project State

### Quality Metrics (as of last repair)

- **TypeScript**: Zero errors (clean compilation)
- **ESLint**: 13 warnings (acceptable technical debt - `any` types for external APIs)
- **Build**: Successful (~7s)
- **Security**: 35 vulnerabilities (mostly dev-only, monitored)

### Known Acceptable Warnings

The 13 ESLint warnings are intentional `any` types in:
- `hypergraphSchemeCore.ts` (6 instances) - Dynamic cognitive patterns
- `mem0aiService.ts` (3 instances) - OpenAI SDK compatibility
- `toroidalCognitiveService.ts` (2 instances) - Flexible cognitive typing
- `adaptiveFeedbackService.ts` (1 instance) - Feedback system
- `tests.ts` (1 instance) - Test fixtures

## Common Issues & Solutions

### ESLint Flat Config Error

If you see "Invalid option '--ignore-path'" error:
```bash
# The fix is already in package.json (ESLINT_USE_FLAT_CONFIG=false)
# If not, run:
ESLINT_USE_FLAT_CONFIG=false npm run lint
```

### tiktoken Network Failures

```python
# Pre-cache GPT-2 vocabulary
import tiktoken
enc = tiktoken.get_encoding('gpt2')
```

### Insufficient Training Data

Increase parameters:
```bash
python prepare_nanecho.py --echo_depth=7 --persona_weight=0.95
```

### Git Push Conflicts in CI

Workflows use 4-layer fallback: pull-rebase → retry → force-lease → backup-branch

### TypeScript Definite Assignment

If you see "has no initializer" errors for properties initialized in `initialize()` methods, use the definite assignment assertion:
```typescript
private property!: Type;  // Note the ! operator
```

## Environment Variables

Required in `.env`:

```
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_key
OPENAI_API_KEY=your_openai_key  # Optional
```

## Development Patterns

### When Modifying Training

1. Update `NanEcho/prepare_nanecho.py` for data changes
2. Update `nanecho_config.json` for model architecture
3. Test with CI mode first: `--echo_depth=3 --persona_weight=0.7`
4. Check `.training-progress/` for session logs

### When Modifying Services

1. Services are in `src/services/`
2. Use TypeScript strict mode
3. Follow existing hypergraph pattern encoding conventions
4. Maintain persona dimension consistency
5. Use `!` definite assignment for properties initialized in `initialize()` methods

### When Modifying Workflows

1. Test changes in fork first
2. Use `continue-on-error: true` for non-critical steps
3. Log to `.maintenance-logs/` for transparency
4. Respect the 4-layer git push fallback pattern

## Quality Standards

### Automated Checks

- TypeScript: Zero errors required
- ESLint: 13 warnings acceptable (documented)
- Security: Dev-only vulnerabilities tracked
- Formatting: Prettier enforced

### Maintenance Logs

All automated changes logged in `.maintenance-logs/`:

- `latest-report.md` - Current status
- `dependency-audit.md` - Package analysis
- Historical logs with timestamps

## Agent Personalities

Primary agents in `.github/agents/`:

- **DEEP_TREE_ECHO.md** - Primary persona
- **NANECHO.md** - GPT-2 trained persona
- **agent-neuro.md** - Chaotic cognitive VTuber supervisor
- **Marduk-ML-AI-Workbench.md** - AI scientist personality
- **AUTOGNOSIS.md** - Self-knowledge system

## Cognitive Architecture Principles

This project implements:

- **4E Cognition**: Embodied, Embedded, Enacted, Extended
- **CogPrime**: Hypergraph-based cognitive architecture
- **Relevance Realization**: Dynamic attention allocation
- **Echo State Networks**: Reservoir computing
- **Self-Morphing Networks**: Adaptive topology with identity preservation

## Git Workflow

- Branch naming: `feature/*`, `fix/*`, `claude/*`
- Commit messages: Descriptive with context
- PRs: Required for main branch changes
- CI must pass before merge

## Testing Guide

```bash
# Unit tests
npm test

# Training validation
cd NanEcho && python prepare_nanecho.py --echo_depth=3

# Build verification
npm run build

# Full quality check
npm run lint && npm run typecheck && npm run build
```

## Troubleshooting

### Dependencies Not Installed

```bash
npm install
```

### Training Data Not Found

1. Check `echoself.md` exists
2. Verify tiktoken installation
3. Increase echo_depth parameter

### TypeScript Errors

1. Run `npm run typecheck`
2. Check `.eslintrc.cjs` for rules
3. Some `any` types required for OpenAI SDK
4. Use `!` for properties with delayed initialization

### Workflow Failures

1. Check `.maintenance-logs/latest-report.md`
2. Review GitHub Actions artifacts
3. Check for disk space issues on runners

### Build Failures

1. Ensure `npm install` has been run
2. Check for TypeScript errors first
3. Review Remix/Vite warnings (most are informational)

---

_Last updated: 2026-01-17_
_This CLAUDE.MD provides essential context for AI-assisted development. For detailed documentation, see the specific README files in each subdirectory._
