"""Seed mech0 from files actually read in this checkout. No invented metrics."""

from __future__ import annotations

from .model import MemoryRecord

REPO = r"C:\hyp\ghx\echo\echoself"

# Persona weights from CLAUDE.md (2026-01-17 guide, still current in this tree).
PERSONA_WEIGHTS = {
    "cognitive": 0.15,
    "introspective": 0.15,
    "adaptive": 0.15,
    "recursive": 0.15,
    "synergistic": 0.10,
    "holographic": 0.10,
    "neural-symbolic": 0.10,
    "dynamic": 0.10,
}

# Arena/Task allowlist from AGENTS.md (read 2026-08-16).
MODEL_ALLOWLIST = [
    "inherit",
    "claude-opus-5-thinking-high",
    "composer-2.5-fast",
    "cursor-grok-4.5-high-fast",
    "cursor-grok-4.6-high-fast",
    "gpt-5.6-sol-medium",
]


def seed_records() -> list[MemoryRecord]:
    return [
        MemoryRecord.create(
            memory_type="semantic",
            content=(
                "Deep Tree Echo persona dimension weights: "
                + ", ".join(f"{name} {weight}" for name, weight in PERSONA_WEIGHTS.items())
                + ". Adaptive attention: threshold = 0.5 + (cognitive_load * 0.3) - (recent_activity * 0.2)."
            ),
            source="CLAUDE.md",
            payload={"concepts": ["persona", "weights", "attention"], "weights": PERSONA_WEIGHTS},
            confidence=1.0,
            pinned=True,
            metadata={"file": "CLAUDE.md"},
        ),
        MemoryRecord.create(
            memory_type="semantic",
            content=(
                "Training modes: CI (4 layers, 200 iterations), Full (12 layers, 50000 iterations), "
                "Relentless (continuous persona reinforcement, scheduled every 4 hours). "
                "nanecho-cached-ci on disk used max_iters 500 / 4 layers / 256 embd — cite the file, do not correct it."
            ),
            source="CLAUDE.md",
            payload={"concepts": ["training-mode", "ci", "full", "relentless"]},
            confidence=1.0,
            pinned=True,
            metadata={"file": "CLAUDE.md"},
        ),
        MemoryRecord.create(
            memory_type="semantic",
            content=(
                "Checkpoint restore priority: "
                "1) .training-progress/checkpoints/latest_checkpoint.pt "
                "2) downloaded artifacts from previous workflow runs "
                "3) GitHub Actions cache "
                "4) any valid checkpoint in backup locations. "
                "force_fresh_start requires an explicit confirmation string."
            ),
            source="CLAUDE.md",
            payload={"concepts": ["checkpoint", "guardian", "priority"]},
            confidence=1.0,
            pinned=True,
            metadata={"file": "CLAUDE.md"},
        ),
        MemoryRecord.create(
            memory_type="semantic",
            content=(
                "On Windows PowerShell, chain shell commands with ; rather than &&, "
                "and pass git commit messages with -m rather than bash HEREDOC. "
                "When committing this repo, exclude untracked *(2)* / *(3)* Explorer copy-duplicates."
            ),
            source="AGENTS.md",
            payload={"concepts": ["powershell", "git", "windows"]},
            confidence=1.0,
            pinned=True,
            metadata={"file": "AGENTS.md"},
        ),
        MemoryRecord.create(
            memory_type="episodic",
            content=(
                "NanEcho cached-CI workflow_run 504 completed 2026-05-30T08:38:47.121973. "
                "parameters: max_iters 500, batch_size 2, learning_rate 0.0002, "
                "model_layers 4, model_embedding 256, force_fresh_start false. "
                "cache_stats: 10 checkpoints, 2531.631278991699 MB, "
                "best_quality_score 1996800.781072235, best_val_loss 0.00035515782814400155. "
                "best_checkpoint ckpt_20260530_083841_13000_22deff1b_9470fbb7 iteration 13000 "
                "created 2026-05-30T08:38:42.071468."
            ),
            source=".training-progress/nanecho-cached-ci/training_summary.json",
            payload={
                "occurred_at": "2026-05-30T08:38:47.121973",
                "event": "nanecho-cached-ci-workflow-504",
            },
            confidence=1.0,
            pinned=True,
            metadata={
                "workflow_run": "504",
                "best_checkpoint": "ckpt_20260530_083841_13000_22deff1b_9470fbb7",
                "file": ".training-progress/nanecho-cached-ci/training_summary.json",
            },
        ),
        MemoryRecord.create(
            memory_type="episodic",
            content=(
                "Best cached-CI checkpoint ckpt_20260530_083841_13000_22deff1b_9470fbb7 "
                "created 2026-05-30T08:38:42.071468 at iteration 13000, "
                "val_loss 0.00035515782814400155, quality_score 1996800.781072235, "
                "tokens_processed 26624000, n_layer 4, n_embd 256, "
                "tags phase_adaptive_mastery high_quality nanecho curriculum introspection. "
                "Notes: Training checkpoint at iteration 13000 (resumed from iteration 12500) | Phase: adaptive_mastery."
            ),
            source=".training-progress/nanecho-cached-ci/cache/metadata.json",
            payload={
                "occurred_at": "2026-05-30T08:38:42.071468",
                "event": "nanecho-best-checkpoint-iter-13000",
            },
            confidence=1.0,
            pinned=True,
            metadata={
                "checkpoint_id": "ckpt_20260530_083841_13000_22deff1b_9470fbb7",
                "file": ".training-progress/nanecho-cached-ci/cache/metadata.json",
            },
        ),
        MemoryRecord.create(
            memory_type="procedural",
            content=(
                "Restore the best cumulative NanEcho checkpoint. Never start from scratch "
                "when a checkpoint exists. force_fresh_start is dangerous and needs an explicit flag."
            ),
            source="scripts/checkpoint_guardian.py",
            payload={
                "instrument": "checkpoint_guardian.restore",
                "signature": "python scripts/checkpoint_guardian.py --output-dir out-nanecho --action restore",
                "steps": [
                    "Run python scripts/checkpoint_guardian.py --output-dir out-nanecho --action restore",
                    "Priority: latest_checkpoint.pt, then artifacts, then GHA cache, then backups",
                    "If restore fails, error out — do not create fallback corpus",
                ],
            },
            confidence=1.0,
            pinned=True,
            metadata={"file": "scripts/checkpoint_guardian.py"},
        ),
        MemoryRecord.create(
            memory_type="procedural",
            content=(
                "Prepare NanEcho persona-weighted training data. CI smoke: "
                "python prepare_nanecho.py --echo_depth=3 --persona_weight=0.7. "
                "Default documented run: python prepare_nanecho.py --echo_depth=5 --persona_weight=0.9."
            ),
            source="NanEcho/prepare_nanecho.py",
            payload={
                "instrument": "prepare_nanecho",
                "signature": "python prepare_nanecho.py --echo_depth=5 --persona_weight=0.9",
                "steps": [
                    "cd NanEcho",
                    "python prepare_nanecho.py --echo_depth=5 --persona_weight=0.9",
                    "Do not invent a minimal fallback corpus if data prep fails",
                ],
            },
            confidence=1.0,
            pinned=True,
            metadata={"file": "NanEcho/prepare_nanecho.py"},
        ),
        MemoryRecord.create(
            memory_type="procedural",
            content=(
                "Dream is Mem0/mech0 consolidation (merge/contradiction/prune). "
                "echo-zero may decide dream but must not apply deletes; a sibling worker applies them. "
                "Remember adds structured facts with infer=false."
            ),
            source=".arena/echo-zero/FRAME.md",
            payload={
                "instrument": "mech0.dream",
                "signature": "POST /memories/search then decide-only dream",
                "steps": [
                    "Load typed memories by type",
                    "Decide merge/contradiction/prune",
                    "Do not apply deletes from this instrument",
                ],
            },
            confidence=0.9,
            pinned=False,
            metadata={"file": ".arena/echo-zero/FRAME.md", "apply": False},
        ),
        MemoryRecord.create(
            memory_type="procedural",
            content=(
                "Continual-learn mines transcript deltas into AGENTS.md. "
                "Hook state lives in .cursor/hooks/state/continual-learning.json."
            ),
            source="AGENTS.md",
            payload={
                "instrument": "continual_learn",
                "signature": "continual_learn → AGENTS.md",
                "steps": [
                    "Read .cursor/hooks/state/continual-learning.json",
                    "Mine high-signal transcript deltas",
                    "Update AGENTS.md learned facts only",
                ],
            },
            confidence=0.9,
            pinned=False,
            metadata={"file": "AGENTS.md"},
        ),
        MemoryRecord.create(
            memory_type="autognosic",
            content=(
                "I am EchoSelf / Deep Tree Echo: a distributed AGI cognitive architecture "
                "with NanEcho (GPT-2 persona model), Echo State Networks, hypergraph pattern "
                "encoding, and EchoLayla. Autognosis is hierarchical self-image building "
                "(self-monitoring, self-modeling, meta-cognitive, self-optimization layers)."
            ),
            source=".github/agents/AUTOGNOSIS.md",
            payload={"about": "identity", "subject": "self"},
            confidence=1.0,
            pinned=True,
            metadata={"file": ".github/agents/AUTOGNOSIS.md"},
        ),
        MemoryRecord.create(
            memory_type="autognosic",
            content=(
                f"Current checkout path is {REPO} (Cursor slug c-hyp-ghx-echo-echoself). "
                "Sibling EchoSelf checkouts exist at C:\\hyp\\ghx\\echoself and C:\\hyp\\ghx\\echoself-1. "
                "Agent transcripts for this workspace live under c-hyp-ghx-echo-echoself/agent-transcripts."
            ),
            source="AGENTS.md",
            payload={"about": "checkout", "subject": "echoself"},
            confidence=1.0,
            pinned=True,
            metadata={"file": "AGENTS.md", "checkout": REPO},
        ),
        MemoryRecord.create(
            memory_type="autognosic",
            content=(
                "Arena/Task runners here only accept: "
                + ", ".join(MODEL_ALLOWLIST)
                + " — not pstack *-max / *-xhigh defaults."
            ),
            source="AGENTS.md",
            payload={"about": "capability", "subject": "system"},
            confidence=1.0,
            pinned=True,
            metadata={"file": "AGENTS.md", "allowlist": MODEL_ALLOWLIST},
        ),
        MemoryRecord.create(
            memory_type="autognosic",
            content=(
                "This process believes cloud Mem0 (plugin-mem0-mem0) is not required and is "
                "currently unusable (HTTP 401 invalid_token). Local mech0 / ech0-mem0 with "
                "SQLite under .mech0/data/ is the memory backend. Memory type is a required "
                "discriminated union: semantic | episodic | procedural | autognosic."
            ),
            source="mech0",
            payload={"about": "belief", "subject": "mech0"},
            confidence=1.0,
            pinned=True,
            metadata={"backend": "local-sqlite", "cloud_mem0_required": False},
        ),
    ]


def main() -> int:
    from .store import MemoryStore

    store = MemoryStore()
    added = 0
    for record in seed_records():
        before = store.counts()["total"]
        store.save(record)
        after = store.counts()["total"]
        if after > before:
            added += 1
    print(f"seeded or confirmed {len(seed_records())} records ({added} new) in {store.db_path}")
    print(store.counts())
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
