# Candidate 2 rationale

## Name

Selected `echo-zero`: it joins the EchoSelf domain to the explicit Agent Zero coordination model and has no project-agent collision.

Alternatives considered and rejected:

- `nanecho`: rejected because it collides conceptually with the existing NanEcho persona and narrows the role to model training.
- `deep-tree-echo`: rejected because it reuses the existing persona name and obscures the subagent's checkpoint and memory coordination duties.
- `echo-guardian`: rejected because it overemphasizes checkpoint safety and underrepresents persona, evaluation, and memory commands.
- `echo-orchestrator`: rejected because it describes generic control flow rather than the local Agent Zero mapping.

## Structure

The prompt defines one `EchoZeroDomain` algebra:

- registries/unions for training modes, ordered checkpoint sources, persona dimensions, cached-CI generations, and Agent Zero surfaces;
- `TrainingReadiness` and `TrainingOrigin` types that prevent training from bypassing restore, prevent unconfirmed fresh starts, and give preparation failure no fallback-corpus value;
- a `Command`/`Event` protocol containing `restore`, `train`, `evaluate`, `continual_learn`, `dream`, `remember`, and `respond`;
- two distinct cached-CI generation variants grounded in workflow runs `504` and `827`, preserving their separate checkpoint counts, IDs, iterations, and validation losses;
- decision-only Mem0 dream events, with deletion delegated to the sibling owner and writes requiring explicit authorization.

## Rejected designs

- A load/validate/transform/save pipeline: rejected because continual learning and dream are domain operations, not temporal processing stages.
- Boolean state such as `hasCheckpoint`, `forceFreshStart`, and `dataReady`: rejected because contradictory combinations remain representable.
- A phase-handler `if/else` chain: rejected because new operations would scatter policy and duplicate transition rules.
- Separate training, persona, and memory agents: rejected because the arena asks for a small surface and one module of domain knowledge.
- Collapsing both cached-CI summaries into a single “latest” record: rejected because the files describe two distinct generations and do not establish continuity between them.
