---
title: NanEcho Surface Adapter - Plan
type: feat
date: 2026-08-16
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# NanEcho Surface Adapter - Plan

## Goal Capsule

- **Objective:** Add an EchoSelf-side domain path that treats `echoself/data/nanecho` as the NanEcho corpus/runtime surface and connects it to existing echogenesis generation, a harmonic-resonance ESN node, a tiny local inference adapter, and a thin Vorticog-shaped agent/event map.
- **Authority:** `.cursor/agents/echo-zero.md` domain unions and registries win for training, persona, and memory commands. This plan adds variants to those unions. It does not add parallel booleans.
- **Stop conditions:** Do not train. Do not download models. Do not fabricate checkpoints or losses. Do not synthesize a fallback corpus. Do not clone Vorticog/ReZorg. Do not commit or open a PR unless a later user request asks.
- **Execution profile:** Smallest vertical slice that typechecks and has a Python smoke test.
- **Tail ownership:** Leave a working tree plus this plan file.

---

## Product Contract

### Summary

This checkout already has echogenesis, NanEcho runtime, mech0 typed memory, and an Echo-Zero training algebra. It does not have a Vorticog repo, local `.pt` weights, or a prepared `data/nanecho` bin corpus.

The product is an adapter on the existing `echoself/data/nanecho` package that fails closed: a missing prepared corpus is `PrepFailure`, a harmonic node without oscillator state cannot be constructed, a Vorticog agent without a type cannot be constructed, and training stays `restore_required` while metadata checkpoints exist and no local weights are restored.

### Problem Frame

NanEcho's documented corpus path is `data/nanecho`. This checkout's live path is `echoself/data/nanecho` (cache module only). `data/nanecho` and `NanEcho/data/nanecho` are absent. A silent empty-corpus fallback would look like a prepared dataset and would violate the training safety policy.

Vorticog, harmonic-resonance ESN, and llama-cpp-spec are external shapes. EchoSelf needs typed seams, not a second product.

### Requirements

**Corpus and training safety**

- R1. Resolve the NanEcho data surface by trying `echoself/data/nanecho`, then `NanEcho/data/nanecho`, then `data/nanecho`. The first existing directory is the runtime surface.
- R2. A directory without `train.bin`, `train.txt`, or corpus `metadata.json` is `PrepFailure` with `fallbackCorpus: never`. Do not invent an empty prepared corpus.
- R3. Training readiness is `runnable` only after a restored checkpoint and a prepared corpus. Metadata-only generations without a local `.pt` file are `restore_required`.
- R4. Fresh start requires guardian-verified absence plus an opaque explicit confirmation. This work never supplies that confirmation.

**Generation and reservoir**

- R5. The surface can invoke existing `echogenesis.initialize_generation` / `EchoGenesis.evolve` and return its `GenerationResult`.
- R6. The harmonic-resonance node stores phases and amplitudes. Construction without oscillator state is impossible.
- R7. The reservoir updates in the frequency domain (phase rotation and amplitude leak). It is not a random recurrent ESN.

**Inference**

- R8. The tiny inference adapter follows a llama-cpp-spec shaped client (`init` / `load` / `generate`) and binds NanEcho `runtime.py` when a trusted local `.pt` exists.
- R9. Missing weights yield `kind: "unavailable"`. The adapter does not download GGUF or other models. NanEcho has no verified GGUF export (`NanEcho/PRODUCTION.md`).

**Vorticog mapping and memory**

- R10. A Vorticog-shaped agent requires a closed `type` (`persona` | `need` | `dreamcog` | `erebus`).
- R11. Persona agents require one of the eight Echo-Zero persona dimensions. Need agents require a closed need kind.
- R12. Memory proposals use mech0 types `semantic` | `episodic` | `procedural` | `autognosic`. Cloud Mem0 is not required. This slice proposes facts; it does not write Mem0.

### Actors

- A1. Echo-Zero coordinator (this adapter's owner).
- A2. NanEcho runtime / checkpoint-guardian (subordinates; not invoked to train here).
- A3. mech0 local memory (optional backend if present).
- A4. Implementer / smoke runner.

### Flows

- F1. Surface resolve
  - **Trigger:** Smoke or compose entry.
  - **Steps:** Walk the three candidate paths. Classify runtime surface vs prepared corpus vs `PrepFailure`.
  - **Covered by:** R1, R2
- F2. Compose without training
  - **Trigger:** `compose` command on a resolved surface.
  - **Steps:** Fold lineage; refuse `train` when restore is required; run echogenesis; step the harmonic node; attempt tiny infer; emit Vorticog events and optional mech0 proposals.
  - **Covered by:** R3, R5, R6, R8, R10, R12
- F3. Invalid construction
  - **Trigger:** Caller omits oscillator state, agent type, or tries an empty-corpus fallback.
  - **Outcome:** Raise a typed error. No object is created.
  - **Covered by:** R2, R6, R10

### Acceptance Examples

- AE1. Missing bins
  - **Covers:** R2
  - **Given:** `echoself/data/nanecho` exists and has no `train.bin` / `train.txt` / corpus `metadata.json`
  - **When:** The resolver classifies the path
  - **Then:** Result is `PrepFailure`, not `PreparedCorpus` with empty text
- AE2. No oscillator
  - **Covers:** R6
  - **Given:** Caller constructs the harmonic node with empty phases or amplitudes
  - **When:** Construction runs
  - **Then:** Construction fails
- AE3. Weights absent
  - **Covers:** R8, R9
  - **Given:** No local `.pt` under `.training-progress/checkpoints/` or `out-nanecho/`
  - **When:** Tiny infer `generate` is called
  - **Then:** Result is `unavailable` and no network download starts
- AE4. Divergent lineage
  - **Covers:** R3
  - **Given:** Cached-CI run `504` and backup run `695` disagree on head
  - **When:** Compose asks whether training is legal
  - **Then:** Readiness is `restore_required`; `train` is not dispatched

### Success Criteria

- TypeScript domain module typechecks with the repo `tsc` config.
- Python smoke covers R2, R6, R9, R10, and a successful echogenesis call.
- No new training run, no new corpus files, no model download.

### Scope Boundaries

**In scope**

- EchoSelf adapter and domain unions.
- Thin Vorticog event/agent map.
- Stdlib harmonic ESN (numpy optional).

**Deferred**

- Actual checkpoint restore via `scripts/checkpoint_guardian.py`.
- Full Vorticog/DreamCog/Erebus simulation.
- GGUF/wasm llama.cpp runtime.
- Writing mech0 or cloud Mem0.

**Outside this product**

- A Vorticog clone.
- Accidental fresh-start training.
- Invented val_loss / iteration numbers.

### Dependencies

- `echogenesis/` generation pipeline.
- `NanEcho/runtime.py` and `NanEcho/PRODUCTION.md` for inference rules.
- `mech0/model.py` and `src/types/mech0.ts` for memory types.
- `.cursor/agents/echo-zero.md` for training algebra.
- Live path `echoself/data/nanecho` (cache module). Prepared bins are not on disk.

### Sources

- `.cursor/agents/echo-zero.md`
- `.arena/echo-zero/SYNTHESIS.md`
- `NanEcho/PRODUCTION.md`, `NanEcho/runtime.py`
- `echogenesis/generation.py`, `src/services/echogenesisService.ts`
- `app/services/echoStateNetwork.server.ts` (random recurrent ESN; do not copy)
- `echoself/data/nanecho/training_cache.py`
- Transcript `f68698dd-4cf6-4adc-8a26-db05a8a1139f` (mech0 local backend; no invented metrics)

---

## Planning Contract

### Key Technical Decisions

- KTD1. Local mech0 is the memory backend. Cloud Mem0 is not required. `(session-settled: user-directed — chosen over cloud Mem0: plugin auth is unusable and the user directed local mech0)`
  - **Governs:** R12
- KTD2. Training never accidental-fresh-starts. Data-prep failure carries `fallbackCorpus: never`. `(session-settled: user-approved — chosen over minimal fallback corpus: CLAUDE.md and Echo-Zero type boundary)`
  - **Governs:** R2, R3, R4
- KTD3. Explorer `(2)` copies are not source of truth for application code. Cached-CI `(2)` generation files remain distinct records when present on disk. `(session-settled: user-approved — chosen over treating every (2) file as merge noise: Echo-Zero forbids collapsing runs 504 and 827)`
  - **Governs:** R3
- KTD4. This checkout has no local `.pt` weights. Do not invent losses. Do not start training from scratch. `(session-settled: user-approved — chosen over using documentation default losses or a fresh model: metadata-only generations are not restored weights)`
  - **Governs:** R3, R8, R9
- KTD5. Candidate order is a registry, not a boolean fallback. First existing directory is the runtime surface. Prepared-corpus evidence is a separate classification.
  - **Governs:** R1, R2
- KTD6. Harmonic ESN is a required `OscillatorState` of phases and amplitudes. Do not reuse `app/services/echoStateNetwork.server.ts`.
  - **Governs:** R6, R7
- KTD7. Tiny inference is a llama-cpp-spec shaped client bound to NanEcho runtime. Unavailable is a result kind, not a download trigger.
  - **Governs:** R8, R9
- KTD8. Vorticog support is a closed agent-type union plus need/memory maps. DreamCog maps to Echo-Zero `dream` (`decide_only`). Erebus is an event source kind, not a sidecar process.
  - **Governs:** R10, R11, R12
- KTD9. New capability is added as EchoZeroDomain union variants (`nanecho_surface`, `harmonic_esn`, `tiny_inference`, `vorticog_map`) on `AgentZeroSurface` and compose commands. No `enableX` flags.
  - **Governs:** all requirements
- KTD10. Live lineage on 2026-08-16 is `divergent` (run `504` vs backup `695`). Generation `827` evidence file is not on disk this invocation. Compose reports that verdict and does not train.

### High-Level Technical Design

```text
resolve(CANDIDATE_PATHS)
    -> runtime_surface | prep_blocked
fold(LineageVerdict)
    -> restore_required | runnable
compose
    -> echogenesis.evolve
    -> harmonic.step(OscillatorState)
    -> tinyInfer.generate | unavailable
    -> vorticog.events + mech0 proposals (not applied)
```

Persona weights stay the Echo-Zero registry: cognitive/introspective/adaptive/recursive at 0.15; synergistic/holographic/neural-symbolic/dynamic at 0.10.

### Assumptions

- `echoself/` is an importable package from the repo root.
- `echogenesis.initialize_generation` can run without torch.
- Numpy may be absent; the harmonic node uses stdlib `math`.
- `npm run typecheck` is the TypeScript compile gate. There is no repo `test` script.

### Implementation Constraints

- Windows PowerShell: chain with `;`.
- Repo-relative paths only in this plan.
- Do not write `.pt`, `train.bin`, or downloaded GGUF files.
- Do not mutate `.training-progress/nanecho-cached-ci/` metrics.

### Sequencing

U1 domain types, then U2 Python surface and harmonic node, then U3 TypeScript mirrors, then U4 smoke.

---

## Implementation Units

### U1. EchoZero domain unions for the surface

- **Goal:** Make the new path representable in TypeScript without new booleans.
- **Requirements:** R3, R4, R10, R11, KTD9
- **Files:**
  - `src/domain/echoZero.ts` (create)
- **Approach:** Port the Echo-Zero unions already in `.cursor/agents/echo-zero.md`. Add `AgentZeroSurface` variants `nanecho_surface`, `harmonic_esn`, `tiny_inference`, `vorticog_map`. Add `NanEchoDataRef`, `OscillatorState`, `VorticogAgent`, `TinyInferenceResult`. Keep `PERSONA` weights unchanged.
- **Test scenarios:**
  - TS1. `PERSONA` has eight keys and the documented weights.
  - TS2. A Vorticog agent type is a required field on the type (compile-time).
  - TS3. `PrepFailure.fallbackCorpus` is `never`.
- **Verification:** `npm run typecheck` includes the new file.

### U2. Python corpus surface, harmonic node, compose

- **Goal:** Live adapter on `echoself/data/nanecho` that fails closed and composes the four seams.
- **Requirements:** R1, R2, R3, R5, R6, R7, R8, R9, R10, R11, R12
- **Files:**
  - `echoself/data/nanecho/surface.py` (create)
  - `echoself/data/nanecho/harmonic_resonance_esn.py` (create)
  - `echoself/data/nanecho/__init__.py` (export the new symbols)
- **Approach:** `CANDIDATE_PATHS` registry in resolve order. Classify prepared vs runtime-only vs missing. `HarmonicResonanceESN` requires non-empty phases and amplitudes; `step` rotates phase and leaks amplitude. `compose_surface` calls `echogenesis.initialize_generation().evolve` with a small salience list, steps the reservoir, and returns `unavailable` inference when no `.pt` is found. Vorticog constructors raise on missing type. Do not call `prepare_nanecho.py` or write corpus files.
- **Test scenarios:**
  - TS4. Resolver selects `echoself/data/nanecho` on this checkout.
  - TS5. Classification is `PrepFailure` because bins are absent.
  - TS6. Empty oscillator state raises.
  - TS7. One harmonic step changes phase and keeps amplitude finite.
  - TS8. Agent with no type raises.
  - TS9. Tiny infer returns `unavailable` without downloading.
  - TS10. Echogenesis evolve returns a `GenerationResult`.
  - TS11. Compose reports `restore_required` / `divergent`, not `train`.
- **Verification:** `python -m unittest tests.test_nanecho_surface`

### U3. TypeScript path and inference mirrors

- **Goal:** App-side functions that match the Python fail-closed rules.
- **Requirements:** R1, R2, R8, R9
- **Files:**
  - `src/domain/nanechoSurface.ts` (create)
- **Approach:** Same candidate registry and prepared-corpus evidence names. Tiny client exposes `init`, `load`, `generate` and returns `{ kind: "unavailable" }` when no checkpoint path is supplied. Do not fetch remote models.
- **Test scenarios:**
  - TS12. Resolver returns the first existing candidate or a typed miss.
  - TS13. `generate` without a checkpoint is `unavailable`.
- **Verification:** `npm run typecheck`

### U4. Smoke tests

- **Goal:** Lock the invalid states from the request.
- **Requirements:** AE1–AE4
- **Files:**
  - `tests/test_nanecho_surface.py` (create)
- **Approach:** unittest, repo-root on `sys.path`, no extra pytest fixtures required. Do not assert invented training metrics. Cite run `504` / `695` only as presence of named files.
- **Test scenarios:** TS4–TS11 from U2.
- **Verification:** `python -m unittest tests.test_nanecho_surface`

---

## Verification Contract

| Gate               | Command                                                                            | Applies to | Done signal                                                                         |
| ------------------ | ---------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------- |
| Python smoke       | `python -m unittest tests.test_nanecho_surface`                                    | U2, U4     | All tests pass                                                                      |
| TypeScript compile | `npm run typecheck`                                                                | U1, U3     | New files have no tsc errors. Pre-existing project errors are reported, not hidden. |
| Safety             | No `prepare_nanecho.py`, no guardian `--allow-fresh-start`, no wget/curl of models | all        | Working tree has no new `.pt`, `.bin`, or `.gguf`                                   |

Do not start NanEcho training to verify this plan.

---

## Definition of Done

- Plan file exists with `artifact_readiness: implementation-ready` and `execution: code`.
- U1–U4 files exist and implement the fail-closed constructors.
- Python smoke passes.
- TypeScript domain compiles, or residual tsc errors are pre-existing and named.
- Compose does not train and does not write a fallback corpus.
- Abandoned experiment files are not left in the tree.

### Per-unit done

- U1: `src/domain/echoZero.ts` exports the extended unions and `PERSONA`.
- U2: `compose_surface` returns structured kinds; invalid states raise.
- U3: TS resolver and tiny client match R1/R2/R9.
- U4: unittest file covers AE1–AE4.
