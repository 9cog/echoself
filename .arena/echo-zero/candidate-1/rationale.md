# echo-zero — candidate-1 rationale

## The fact that chose the structure

Three files in this repo claim a different training head for the same lineage `22deff1b/9470fbb7`:

| site                                          | workflow_run | iteration | val_loss               |
| --------------------------------------------- | ------------ | --------- | ---------------------- |
| `nanecho-cached-ci/training_summary.json`     | 504          | 13000     | 0.00035515782814400155 |
| `checkpoints/backup_manifest.json`            | 695          | 200       | 2.0612                 |
| `nanecho-cached-ci/training_summary (2).json` | 827          | 500       | 6.55673903465271       |

The newest run (827, completed 2026-08-16) reports the _worst_ loss and a _lower_ iteration than the oldest. And `checkpoints/latest_checkpoint.pt` — restore priority 1 in `CLAUDE.md` — does not exist, though `backup_manifest.json` asserts `backup_count: 3`.

So any model with a scalar `currentIteration` is false on arrival. That single observation is what forces the ledger + verdict shape; the structure is not decoration over a config file.

## What was rejected

**A state machine over training phases** (`idle → preparing → training → evaluating → dreaming`). Rejected: the frame explicitly warns against temporal/phase-named modules that repeat the same rules, and the repo already has a phase vocabulary (`phase_basic_awareness` … `phase_adaptive_mastery`) that belongs to _checkpoints_, not to the agent. A second phase machine would be two phase concepts needing to stay in sync. Curriculum phase survives as a tag you read; agent control flow is a decision table instead.

**A `TrainingState` record with `hasCheckpoint` / `forceFreshStart` / `corpusReady` booleans.** Rejected outright: three booleans that must agree is exactly the "second boolean that must stay in sync" failure. Replaced by `ResumeIntent`, where `{kind:"fresh"}` cannot be constructed without an `Attestation`, and the `Attestation`'s `witness` slot admits only `{kind:"uninitialized"}`. The frame's first two invalid states stop being validated and start being unconstructible.

**Scalar "latest checkpoint" with regression detection.** Rejected: that is a boolean (`isRegressed`) plus a comparison, and it presumes checkpoints are totally ordered. They are not — ordering is only defined inside a `Fingerprint`. Comparability became a parse result of the `ckpt_<ts>_<iter>_<config>_<data>` id, which is why `forked` is a first-class verdict rather than a crash.

**A `CorpusOrigin.fallback` variant with a "never use in CI" comment.** Rejected: `CLAUDE.md` says data-prep failure must not create minimal fallback data. A variant you must remember not to construct is a comment; deleting the variant is a type. `{kind:"absent"}` reduces only to `Halt`.

**`load / validate / transform / save` pipeline stages, and a `MemoryPipeline` wrapping continual-learning.** Rejected by the frame's small-surface rule. `ContinualLearn`, `Dream`, and `Remember` are three named commands on one `MemorySurface`, sitting in the same `Command` union as `Train` and `Restore` — the memory operations and the training operations are peers, not a subsystem.

**Modelling the Mem0 delete guardrail as a permission check (`if (!canDelete) throw`).** Rejected: that is the if/else chain the frame forbids, and it is bypassable. Instead `Dream` emits `DreamProposed` and the agent simply has no `Mem0Deleted` event constructor. The sibling worker owns application; the boundary is structural.

**Deriving persona weights at runtime from cognitive load.** Rejected: the eight weights are fixed data in `CLAUDE.md` summing to 1.00. Only the attention _threshold_ is computed. `PERSONA` stays a total registry so allocation is over all eight or none — no per-dimension enable flags.

**Collapsing the `(2)` duplicate files into one canonical reading.** Tempting and wrong: the disagreement _is_ the domain signal. They are registered as distinct `Site`s and an operating rule forbids deleting them to make the ledger agree.

**Names `nanecho`, `deep-tree-echo`, `echoself-trainer`.** Rejected: the frame reserves the first two, and `echoself-trainer` misdescribes an agent whose most common correct answer today is `Restore`, not train. `echo-zero` also carries the Agent Zero mapping in the name — Agent 0 with the user as superior.

## Why the decision table over `decide()` with guards

Four `LineageVerdict` variants × two `CorpusOrigin` variants is six meaningful rows (`divergent` and `forked` short-circuit corpus). Six rows fit in one table, are exhaustive by construction, and adding a restore source means appending to `RESTORE_ORDER` rather than editing control flow. The current repo state (`divergent` → `Restore`) falls out of a normal row — it is not a special case, which was the test I held the design to.

## Known cost

`ProgressLedger` must be total over `Site`, so every invocation re-reads seven paths. That is deliberate: `continual-learning.json` already drifted between the arena frame's snapshot (`turnsSinceLastRun 8`, generation `33247975-…`) and the live file (`9`, `ccda2d0c-…`) during this session. Caching would have made the agent quote a stale number confidently.
