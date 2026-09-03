# echo-zero arena synthesis

Date: 2026-08-16
Workspace: `C:\hyp\ghx\echo\echoself`
Branch: `main`
Dropouts: none (all four candidates + judge completed)

## Pick

**Base: candidate-2** (`EchoZeroDomain` command/event algebra).

Coordinator scores (1–5 per rubric criterion):

| Criterion       |     C1 |     C2 |     C3 |     C4 |
| --------------- | -----: | -----: | -----: | -----: |
| 1 Domain model  |      5 |      4 |      4 |      3 |
| 2 Frontmatter   |      5 |      5 |      4 |      5 |
| 3 Agent Zero    |      4 |      5 |      4 |      5 |
| 4 Grounding     |      5 |      5 |      5 |      4 |
| 5 Memory ops    |      5 |      5 |      5 |      5 |
| 6 Small surface |      3 |      5 |      4 |      4 |
| **Total**       | **27** | **29** | **26** | **26** |

[Judge](e766ff9b-b911-40c7-b0b7-9e555c2052db) (gpt-5.6-sol-medium): C2 29, C1 26, C3 26, C4 25. Recommended base C2, runner-up C1.

Agreement: same base. Tie-break is the rubric's "cleaner boundary / smaller surface." C1's ledger is the richer model of the live `504`/`695`/`827` disagreement, but it is too large for one Cursor subagent prompt and names `echo-zero` as superior instead of Deep Tree Echo as profile.

Candidates converged on the same spine (unions + registries + MemoryOp commands). Differences were size, restore-order representation, and whether aggregate state can still contradict.

## Grafts (1–2 from each loser)

From [Candidate 1](752ff148-b364-40c2-b744-ca67fc0a1847):

- `RESTORE_ORDER` as an ordered list consumed by resolve (first present), plus verified `latest_checkpoint.pt` absence.
- Minimal `LineageVerdict` (`coherent | divergent | forked | uninitialized`) and fingerprint comparability — not the seven-site total ledger.

From [Candidate 3](368a656d-8db3-4ad8-9c4b-dc01e5aa47e8):

- `DISPATCH: Record<SituationTag, Command["kind"]>` so command choice is a registry lookup.
- NANECHO.md CI `100` iterations cited alongside CLAUDE.md `200` and on-disk `max_iters` `500`.

From [Candidate 4](6c4d16e6-d3d3-4718-a6d9-b9df430a2266):

- Subordinate ownership (NanEcho train/infer; guardian restore/backup/verify/cleanup; Mem0-dream apply).
- Deep Tree Echo is the superior **profile**, not a second Agent 0.

## Rejections

- C1 full `ProgressLedger` over seven `Site`s — justified by disk disagreement, too much surface for the prompt.
- C1 "exactly one command" vs later orthogonal memory emission — inconsistent.
- C3 extension mapping written as a procedural sequence (load → inspect → dispatch → inject) — pipeline smell.
- C3 frontmatter lists what to delegate _away_ without a clear "delegate here when".
- C4 `EchoSelf` record holding `lineage` + `origin` + `fresh` independently — contradictory combinations remain representable.
- C4 frozen claim that both introspection files exist and are `[]` (FRAME had no non-`(2)` file; presence drifts).
- Collapsing runs `504` and `827` into one best\_\* scalar.
- Freezing hook `turnsSinceLastRun` (frame 8 → candidates 9 → verify 4). Re-read only.
- Mechanical averaging of the four prompts.

## Verification

- YAML frontmatter parses: `name: echo-zero`, description includes "use proactively" and when to delegate.
- Path: `.cursor/agents/echo-zero.md`
- No collision: project `.cursor/agents/` was empty; user-level `gitboy` is a different name.
- Grounded metrics match files actually read: runs `504`, `827`, `695`; no invented losses/iterations.
- Mem0 dream deletes not applied (sibling owns dream).

## Mem0 persist

Attempted `plugin-mem0-mem0` after `GetMcpTools`. Server status `error` ("failed during live tool discovery"); `mcp_auth` rejected (`server not found` / timeout). Shell `resolve_api_key()` returned empty (`MEM0_API_KEY` unset). Discovered identity for a later write: `user_id=0f1f0fb4-36f2-4c30-990b-07cdc2c8b426`, `app_id=9cog-echoself`, `branch=main`. **No memory ids added.** Do not apply dream deletes.

## Domain model (structure)

```
TrainingMode = ci | full | relentless
RESTORE_ORDER = [latest_checkpoint, downloaded_artifact, gha_cache, backup]
PersonaRegistry = 8 ids → {0.15×4, 0.10×4}
Fingerprint = {config, data}   // comparability is a parse result
LineageVerdict = coherent | divergent | forked | uninitialized
TrainingOrigin = restored | confirmed_fresh_start(no_checkpoint, ExplicitConfirmation)
PrepFailure.fallbackCorpus = never
CachedCIGeneration = 504 | 827   // never merged
MemoryOperation = continual_learn | dream(decide_only) | remember(infer:false)
SituationTag → DISPATCH → Command
AgentZeroSurface = tool | extension | instrument | profile | subordinate
```

Invalid / unconstructible: fresh start without confirmation; confirmed fresh while a checkpoint exists; fallback corpus on prep failure; train-from-scratch when a checkpoint exists; a single invented best-iter/best-loss across generations.
