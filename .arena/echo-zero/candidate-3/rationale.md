# Rationale — echo-zero (candidate-3)

The prompt _is_ the domain module: a typed `EchoSelf` object, a `Situation` discriminated union, and a `DISPATCH` registry from tag → `Command`. Continual-learning and dream are `Command` / `MemoryOp` variants on that same surface.

## Chosen spine

- `TrainPlan` is a union of `resume` (requires `Presence.present` + `DataPrep.ok`) and `fresh` (requires `Presence.absent` + `FreshStart.confirmed` + `DataPrep.ok`). Those three invalid states cannot be constructed.
- Two `Generation` rows keyed by workflow_run `504` and `827`. Quality and curriculum phases are fields on the row, not control-flow modules.
- Agent Zero maps 1:1 onto tools / extensions / profiles / instruments / `call_subordinate`. User → echo-zero → subordinates. Deep Tree Echo is the superior _profile_, not a second root.

## Alternatives considered and rejected

1. **Load / validate / transform / save pipeline** — rejected. FRAME forbids phase-named pipelines that repeat the same rules. Inspection is an `Event`; the next act is a `Command`.
2. **Boolean pair `hasCheckpoint` + `forceFreshStart`** — rejected. The two flags can desync and would make fresh-start-without-confirmation representable.
3. **Single “current” generation / averaged metrics** — rejected. On-disk `training_summary.json` (run 504, iter 13000) and `training_summary (2).json` (run 827, iter 500) are distinct rows. Collapsing them invents a number.
4. **Name `nanecho` or `deep-tree-echo`** — rejected. Those are `.github/agents/` personalities. FRAME forbids them as the Cursor subagent name. Project `.cursor/agents/` is empty; `echo-zero` does not collide.
5. **Name `gitboy`** — rejected. Reserved for the org App control-plane; everyday git stays with the parent.
6. **If/else command dispatcher** — rejected. `DISPATCH` is a closed `Record<Situation["tag"], Command["type"]>`. Adding a command means adding a tag, not another branch.
7. **Separate memory pipelines for AGENTS.md vs Mem0** — rejected. One `MemoryOp` union: `continual_learn` | `dream` | `remember`.
8. **Applying Mem0 writes or deletes from this agent** — rejected. Sibling worker owns dream deletes. This agent may decide `{ type: "dream", apply: false }` only.
9. **Second hierarchy (echo-zero and Deep Tree Echo as peer roots)** — rejected. User is Agent 0’s superior. DTE is the profile used when calling subordinates.
10. **Curriculum phases as sequential modules** — rejected. Phases are tags on `Generation` (`adaptive_mastery` on 504; the 0→500 walk on 827). They do not own rules.
11. **“Correcting” on-disk CI `max_iters: 500` to CLAUDE.md’s 200 (or NANECHO.md’s 100)** — rejected. Cite each source; do not overwrite the file.
12. **Using FRAME’s hook snapshot as live state** — rejected. Files as read: `turnsSinceLastRun` 9, `lastProcessedGenerationId` `ccda2d0c-…`, index transcripts non-empty (this workspace + `(2)` sibling slug). FRAME’s older empty-index / 8-turn snapshot is not substituted.
13. **Writing into `.cursor/agents/`** — rejected. Arena artifact only; coordinator grafts the winner.
