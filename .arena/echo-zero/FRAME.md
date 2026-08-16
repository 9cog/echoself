# Echo-Zero Arena Frame

Date: 2026-08-16
Workspace: `C:\hyp\ghx\echo\echoself`
Branch: `main`
Do not commit. Do not push.

## Artifact (each candidate)

Write ONLY to your assigned directory:

1. `agent.md` — Cursor project subagent: YAML frontmatter with `name` (lowercase hyphens) and a specific actionable `description` that includes when to delegate and the phrase "use proactively". Then the system-prompt body.
2. `rationale.md` — short rationale naming alternatives considered and rejected. Mandatory.
3. Optional tiny domain-model sketch in the same files (typed object / discriminated union / registry / state machine) that the subagent will enforce.

Do not write to `.cursor/agents/` — the coordinator grafts the winner later.

## Domain-modeling principle (must follow)

Encode the real domain in a data structure. Reach for: state machine, typed model, map/registry/discriminated union, reducer/command-event, one module of domain knowledge, queue/cache/index/graph where the access pattern calls for it.

Do NOT force an abstraction. Do NOT grow if/else chains. Do NOT add a second boolean that must stay in sync with the first. Do NOT use temporal/phase-named modules that repeat the same rules.

Suggested domain spine (improve this, do not scatter it):

- `TrainingMode = ci | full | relentless`
- `CheckpointSource` = ordered priority (latest_checkpoint.pt → artifacts → gha cache → backups)
- `PersonaDimension` registry (cognitive, introspective, adaptive, recursive, synergistic, holographic, neural-symbolic, dynamic) with weights
- `MemoryLoop = continual_learning | dream | remember`
- `AgentZeroSurface = tool | extension | instrument | profile | subordinate`

Invalid states (must be unrepresentable):

- `force_fresh_start` without confirmation string
- data-prep failure creating fallback corpus
- training starting from scratch when a checkpoint exists

## Agent Zero mapping the subagent must enforce

When invoked, the subagent:

1. Load the EchoSelf domain model (not ad hoc flags).
2. Inspect `.training-progress/nanecho-cached-ci` and checkpoint guardian state.
3. Decide the next domain command (`train | restore | evaluate | dream | continual_learn | respond`) from the model, not from scattered booleans.
4. May call subordinates the way Agent Zero uses `call_subordinate`.
5. Treat AGENTS.md updates as continual-learning and Mem0 consolidation as dream — both are operations on the same memory surface.

Map Agent Zero concepts locally onto EchoSelf (no extra indirection):

| Agent Zero | EchoSelf |
|---|---|
| tools (`python/tools` or `agents/{profile}/tools`) | NanEcho / checkpoint-guardian / training-progress inspectors |
| extensions (lifecycle: agent_init, message_loop_start, before_main_llm_call, system_prompt, response_stream, monologue_end) | continual-learning hook + Mem0 remember/dream points |
| profiles (`agents/{profile}/settings.json`) | Deep Tree Echo persona + NanEcho CI/full/relentless |
| instruments (callable procedures) | `scripts/checkpoint_guardian.py`, prepare_nanecho, evaluation |
| `call_subordinate` hierarchy | Deep Tree Echo (superior) → NanEcho / checkpoint-guardian / Mem0-dream (subordinates) |
| memory_save/load/delete | AGENTS.md (continual_learn) + Mem0 (remember / dream) |

User is Agent 0's superior. Subordinates report back. Do not invent a second hierarchy.

## Grounding facts you MUST read (do not invent metrics)

Read these files yourself before writing:

- `C:\hyp\ghx\echo\echoself\CLAUDE.md`
- `C:\hyp\ghx\echo\echoself\AGENTS.md`
- Everything under `C:\hyp\ghx\echo\echoself\.training-progress\nanecho-cached-ci\`
- `C:\hyp\ghx\echo\echoself\.cursor\agents\` (empty as of frame; avoid colliding with `gitboy`)
- `.github/agents/NANECHO.md` and `.github/agents/DEEP_TREE_ECHO.md`
- This FRAME.md

### nanecho-cached-ci facts (verbatim from files, 2026-08-16)

Two summary generations exist. Cite both; do not collapse them into one invented number.

**`training_summary.json`** (workflow_run `504`, completed `2026-05-30T08:38:47.121973`):

- params: max_iters 500, batch_size 2, learning_rate 0.0002, model_layers 4, model_embedding 256, `force_fresh_start: false`
- cache_stats: 10 checkpoints, 2531.631278991699 MB, best_quality_score 1996800.781072235, best_val_loss 0.00035515782814400155
- best_checkpoint: `ckpt_20260530_083841_13000_22deff1b_9470fbb7`, iteration 13000, created `2026-05-30T08:38:42.071468`

**`training_summary (2).json`** (workflow_run `827`, completed `2026-08-16T06:14:30.454042`):

- same params, `force_fresh_start: false`
- cache_stats: 7 checkpoints, 1936.4671354293823 MB, best_quality_score 76800.61010787632, best_val_loss 6.55673903465271
- best_checkpoint: `ckpt_20260731_145425_500_22deff1b_9470fbb7`, iteration 500, created `2026-07-31T14:54:25.427860`

**`cache/metadata.json`**: 10 ckpts, iterations 9500→13000, tags include `phase_adaptive_mastery`, `high_quality`, n_layer 4 / n_embd 256. Latest id matches May-30 best (iter 13000).

**`cache/metadata (2).json`**: 7 ckpts, iterations 0→500, curriculum tags `phase_basic_awareness` → `phase_persona_dimensions` → `phase_hypergraph_patterns` → `phase_recursive_reasoning` → `phase_adaptive_integration`, `low_quality`. Latest extra ckpt `ckpt_20260816_061425_500_22deff1b_9470fbb7` notes "resumed from iteration 500".

**`introspection_history (2).json`**: empty array `[]`. No non-`(2)` introspection_history file present.

### Persona weights (CLAUDE.md)

Registry, not booleans: cognitive 0.15, introspective 0.15, adaptive 0.15, recursive 0.15, synergistic 0.10, holographic 0.10, neural-symbolic 0.10, dynamic 0.10.

Adaptive attention: `threshold = 0.5 + (cognitive_load * 0.3) - (recent_activity * 0.2)`

### Training modes (CLAUDE.md)

- CI: 4 layers, 200 iterations (nanecho-cached-ci on disk used max_iters 500 / 4 layers / 256 embd — cite the file, do not "correct" it)
- Full: 12 layers, 50000 iterations
- Relentless: continuous persona reinforcement, scheduled every 4 hours

### Checkpoint restore priority (CLAUDE.md, ordered list)

1. `.training-progress/checkpoints/latest_checkpoint.pt`
2. Downloaded artifacts from previous workflow runs
3. GitHub Actions cache
4. Any valid checkpoint in backup locations

### Continual-learning hook state

`.cursor/hooks/state/continual-learning.json`: version 1, lastRunAtMs 0, turnsSinceLastRun 8, lastProcessedGenerationId `33247975-f88c-481d-b5e7-fc5d391b845c`.

`.cursor/hooks/state/continual-learning-index.json`: version 1, transcripts empty object.

### Memory operations (same surface)

- `continual_learn` → mine transcript deltas into `AGENTS.md`
- `dream` → Mem0 consolidation (merge/contradiction/prune). A sibling worker owns applying dream deletes; this subagent may *decide* `dream` but must not apply Mem0 deletes.
- `remember` → Mem0 add (infer=False for structured domain facts)

### AGENTS.md facts

- Windows PowerShell: chain with `;` not `&&`; git `-m` not bash HEREDOC.
- Sibling checkout pairing: `C:\hyp\ghx\echoself` with `C:\hyp\ghx\echoself-1`.

## Cursor subagent file shape

Follow `C:\Users\d\.cursor\agents\gitboy.md` shape:

```markdown
---
name: echo-zero
description: ... Use proactively when ... Delegate when ...
---

# echo-zero
...
```

Prefer name `echo-zero` unless a collision forces a variant. Project `.cursor/agents/` is empty. Do not use `gitboy`, `nanecho`, or `deep-tree-echo` as the Cursor subagent name.

Description must be specific and actionable, include "use proactively", and say when the parent should delegate.

## Small surface

One module of domain knowledge. Prefer the cleaner boundary when tied. The prompt should *be* the domain model the agent loads — not a tutorial and not a phase-named pipeline (load/validate/transform/save).
