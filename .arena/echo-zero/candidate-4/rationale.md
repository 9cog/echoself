# Rationale

echo-zero is one command-event module: typed `EchoSelf` + decision registry. Invalid states are missing constructors, not runtime checks.

## Chosen spine

- Discriminated unions for `TrainingOrigin`, `FreshStart`, `Prep`, `Command`, `Event`
- Ordered `CheckpointSource` ranks (not a boolean)
- `PersonaRegistry` with fixed weights (not eight flags)
- `Lineage` keyed by workflow `504` | `827` (two records, never merged)
- `MemoryOp` as named commands on one surface (`continual_learn` | `dream` | `remember`)
- Agent Zero mapped locally: tools / extensions / profiles / instruments / `call_subordinate`

## Alternatives considered and rejected

1. **Load/validate/transform/save pipeline** — repeats the same guardian rules under temporal phase names. FRAME forbids it; continual_learn/dream would become steps instead of commands.

2. **`has_checkpoint` + `force_fresh_start` booleans** — two flags that must stay in sync. Makes "fresh start while a checkpoint exists" representable. Replaced by `TrainingOrigin` / `FreshStart` unions (`confirmed` requires `absent: NoCheckpoint`).

3. **Fallback corpus on prep failure** — a `failed` variant that still holds data. Unrepresentable: `Prep.failed` has no corpus field; `train` requires `prep: ready`.

4. **Collapse 504 and 827 into one best_*** — FRAME and the files disagree on "current" quality (iter 13000 / val_loss ~3.55e-4 vs iter 500 / val_loss ~6.56). A merged scalar invents a metric. `Lineage` keeps both.

5. **Curriculum as a phase machine the agent walks** — NANECHO.md lists five training phases; `(2)` metadata tags them on checkpoints. Walking `basic_awareness → … → adaptive_mastery` as modules re-encodes the same rules temporally. Tags stay on `Generation` records.

6. **"Correct" CI to 200 (or 100) iters** — CLAUDE.md says 200, NANECHO.md says 100, on-disk summaries say `max_iters` 500 / 4 layers / 256 embd. Overwriting the files would invent a number. Cite all three.

7. **Name `nanecho` or `deep-tree-echo`** — reserved persona/agent ids; FRAME forbids them as the Cursor subagent name. `.cursor/agents/` is empty, so `echo-zero` has no collision (`gitboy` is user-level and a different domain).

8. **Two memory systems** — AGENTS.md vs Mem0 as separate pipelines. They are one `MemoryOp` surface with different sinks. Dream/remember are decided here; a sibling applies Mem0 mutations.

9. **Second hierarchy** (User → DTE Agent 0 → echo-zero → …) — FRAME: user is Agent 0's superior; DTE is the superior **profile**. One line: User → echo-zero → {NanEcho, checkpoint-guardian, Mem0-dream}.

10. **Generic cognitive-architecture essay** (4E, CogPrime, OEIS, EchoLayla, Remix) — second module. Out of surface. Those stay in CLAUDE.md / DEEP_TREE_ECHO.md unless a `respond` command needs a citation.

11. **Applying Mem0 writes/deletes from `dream`/`remember`** — arena rule and FRAME: this agent may decide those ops; it must not apply them.

12. **Treating `*(2)*` Explorer copies as source of truth** — AGENTS.md says not to. Exception: the two nanecho-cached-ci summary/metadata generations are cited as distinct `Lineage` records, not merged and not committed.
