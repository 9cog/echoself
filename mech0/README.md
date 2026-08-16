# mech0 (ech0-mem0)

Local self-hosted memory for EchoSelf. Cloud Mem0 is **not** required.

No official `ech0-mem0` / `mech0` project exists upstream, and this machine has no Docker, so this in-repo stack uses Python stdlib (`http.server` + SQLite + FTS5 + hashed embeddings).

## Four memory types

`MemoryType = semantic | episodic | procedural | autognosic`

Type is required. The same fact cannot be stored as two types unless you send an explicit `POST /memories/dual-write`. Autognosic records need `about` (identity|capability|checkout|belief|self_model). Procedural records need `instrument` plus `signature` and/or `steps`.

## Start / stop

From the repo root (`C:\hyp\ghx\echo\echoself`):

```powershell
python -m mech0
```

Stop with Ctrl+C.

Optional:

```powershell
python -m mech0 serve --host 127.0.0.1 --port 8765 --data-dir .mech0/data
python -m mech0 seed
```

First start seeds from files that were actually read (`CLAUDE.md`, `AGENTS.md`, `.training-progress/nanecho-cached-ci/training_summary.json`, `cache/metadata.json`, `AUTOGNOSIS.md`, `checkpoint_guardian.py`, `prepare_nanecho.py`, `.arena/echo-zero/FRAME.md`).

## Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | backend + counts |
| GET | `/types` | the four types |
| POST | `/memories` | typed body; `type` required |
| GET | `/memories?type=` | list by type |
| GET | `/memories/{id}` | fetch one |
| POST | `/memories/search` | `{ query, type?, limit? }` |
| POST | `/memories/dual-write` | explicit two-type write |
| DELETE | `/memories/{id}` | optional `?type=` |
| POST | `/instruments/memory_save` | Agent Zero instrument; `type` required |
| POST | `/instruments/memory_load` | `type` required; `query` or `id` |
| POST | `/instruments/memory_delete` | `type` + `id` required |

Default URL: `http://127.0.0.1:8765`

## Data

Gitignored directory: `.mech0/data/mech0.sqlite`

Env keys (see `.env.example`): `MECH0_URL`, `MECH0_HOST`, `MECH0_PORT`, `MECH0_DATA_DIR`

## Clients

- Python: `from mech0 import Mech0Client`
- TypeScript: `src/services/mech0Client.ts`
