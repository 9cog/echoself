# CLAUDE.MD - EchoSelf Development Guide

This file provides essential context for Claude Code when working with the EchoSelf/Deep Tree Echo project.

## Project Overview

**EchoSelf** is an advanced distributed AGI cognitive architecture featuring:

- **Deep Tree Echo**: Primary AI persona with adaptive cognition
- **NanEcho**: GPT-2-based model trained on persona-specific data
- **Echo State Networks**: Temporal pattern recognition and adaptive learning
- **Hypergraph Pattern Encoding**: Neural-symbolic cognitive associations
- **Toroidal Dual-Persona Processing**: Interdependent cognitive streams

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
│   └── adaptiveFeedbackService.ts
├── .github/
│   ├── workflows/        # 10 CI/CD workflows
│   └── agents/           # 80+ agent personality definitions
├── echo/                 # Echo State Network modules
├── app/                  # Remix web application
└── .training-progress/   # Training session logs
```

### Essential Commands

```bash
# Development
npm run dev              # Start development server
npm run build            # Production build
npm start                # Run production server

# Training (NanEcho)
cd NanEcho
python prepare_nanecho.py --echo_depth=5 --persona_weight=0.9
python server.py         # Start inference server

# Code Quality
npm run lint             # ESLint check
npm run format           # Prettier formatting
deno lint               # Deno linting (for .ts files)

# Testing
npm test                 # Run test suite
```

### Configuration Files

| File                  | Purpose                        |
| --------------------- | ------------------------------ |
| `nanecho_config.json` | NanEcho model hyperparameters  |
| `package.json`        | Node.js dependencies           |
| `pyproject.toml`      | Python package config          |
| `.env.example`        | Environment variables template |

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

## Workflow Reference

### Training Workflows

| Workflow                        | Purpose                         |
| ------------------------------- | ------------------------------- |
| `netrain.yml`                   | Primary NanEcho training        |
| `agent-neuro-train.yml`         | Agent-Neuro supervised training |
| `netrain-cached.yml`            | Cached/incremental training     |
| `test-training-persistence.yml` | Training persistence tests      |

### Quality Workflows

| Workflow                    | Purpose                       |
| --------------------------- | ----------------------------- |
| `automated-quality.yml`     | Linting, formatting, security |
| `deno.yml`                  | Deno-specific checks          |
| `disk-space-management.yml` | CI disk management            |

## Common Issues & Solutions

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

### When Modifying Workflows

1. Test changes in fork first
2. Use `continue-on-error: true` for non-critical steps
3. Log to `.maintenance-logs/` for transparency
4. Respect the 4-layer git push fallback pattern

## Quality Standards

### Automated Checks

- TypeScript: Zero errors required
- ESLint: 13 warnings acceptable (documented)
- Security: 8 dev-only vulnerabilities tracked
- Formatting: Prettier enforced

### Maintenance Logs

All automated changes logged in `.maintenance-logs/`:

- `latest-report.md` - Current status
- `dependency-audit.md` - Package analysis
- Historical logs with timestamps

## Agent Personalities

Primary agents in `.github/agents/`:

- **deep-tree-echo.md** - Primary persona
- **agent-neuro.md** - Chaotic cognitive VTuber supervisor
- **marduk.md** - AI scientist personality
- **echolayla.md** - Character-based AI integration

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
```

## Troubleshooting

### Training Data Not Found

1. Check `echoself.md` exists
2. Verify tiktoken installation
3. Increase echo_depth parameter

### TypeScript Errors

1. Run `npm run typecheck`
2. Check `.eslintrc.cjs` for rules
3. Some `any` types required for OpenAI SDK

### Workflow Failures

1. Check `.maintenance-logs/latest-report.md`
2. Review GitHub Actions artifacts
3. Check for disk space issues on runners

---

_This CLAUDE.MD provides essential context for AI-assisted development. For detailed documentation, see the specific README files in each subdirectory._
