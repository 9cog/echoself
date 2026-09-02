"""SQLite persistence for mech0. Local files only — no cloud Mem0."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .model import (
    DualWriteCommand,
    DualWriteRequired,
    MEMORY_TYPES,
    MemoryRecord,
    MemoryType,
    MemoryTypeRegistry,
    MemoryValidationError,
    REGISTRY,
    parse_memory_type,
)

EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9_./:-]+")


def default_data_dir() -> Path:
    raw = os.environ.get("MECH0_DATA_DIR", "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parent.parent / ".mech0" / "data"


def fingerprint(content: str) -> str:
    norm = " ".join(content.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def hashed_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    vec = [0.0] * dim
    for token in _TOKEN_RE.findall(text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        a = int.from_bytes(digest[:4], "little")
        b = int.from_bytes(digest[4:8], "little")
        vec[a % dim] += 1.0
        vec[b % dim] += 0.5
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _fts_query(raw: str) -> str:
    tokens = _TOKEN_RE.findall(raw.lower())
    if not tokens:
        return '""'
    return " OR ".join(f'"{tok}"' for tok in tokens[:12])


class MemoryStore:
    """One collection API keyed by MemoryType — not four stores."""

    def __init__(
        self,
        data_dir: Path | None = None,
        registry: MemoryTypeRegistry | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "mech0.sqlite"
        self.registry = registry or REGISTRY
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        allowed = ", ".join(f"'{t}'" for t in MEMORY_TYPES)
        self._conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL CHECK (type IN ({allowed})),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                confidence REAL NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL,
                metadata TEXT NOT NULL,
                spec TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                dual_write_id TEXT,
                embedding TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_fp_type
                ON memories(fingerprint, type);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
            CREATE INDEX IF NOT EXISTS idx_memories_fp ON memories(fingerprint);
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED,
                type UNINDEXED,
                content,
                source
            );
            """
        )
        self._conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        spec = self.registry.spec_from_dict(json.loads(row["spec"]))
        return MemoryRecord(
            id=row["id"],
            type=parse_memory_type(row["type"]),
            content=row["content"],
            created_at=row["created_at"],
            confidence=float(row["confidence"]),
            pinned=bool(row["pinned"]),
            source=row["source"],
            metadata=json.loads(row["metadata"]),
            spec=spec,
            dual_write_id=row["dual_write_id"],
        )

    def _existing_types(self, fp: str) -> set[MemoryType]:
        rows = self._conn.execute(
            "SELECT type FROM memories WHERE fingerprint = ?", (fp,)
        ).fetchall()
        return {parse_memory_type(row["type"]) for row in rows}

    def save(self, record: MemoryRecord, *, allow_dual_write: bool = False) -> MemoryRecord:
        fp = fingerprint(record.content)
        existing = self._existing_types(fp)
        if existing and record.type not in existing and not allow_dual_write:
            raise DualWriteRequired(
                f"Fact already stored as {sorted(existing)}; "
                "POST /memories/dual-write to store it under another type"
            )
        same = self._conn.execute(
            "SELECT id FROM memories WHERE fingerprint = ? AND type = ?",
            (fp, record.type),
        ).fetchone()
        if same:
            found = self.get(same["id"])
            if found is None:
                raise MemoryValidationError("duplicate fingerprint row missing")
            return found

        search_text = f"{record.content} {record.source} {json.dumps(record.spec.to_dict())}"
        embedding = hashed_embedding(search_text)
        self._conn.execute(
            """
            INSERT INTO memories (
                id, type, content, created_at, confidence, pinned, source,
                metadata, spec, fingerprint, dual_write_id, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.type,
                record.content,
                record.created_at,
                record.confidence,
                1 if record.pinned else 0,
                record.source,
                json.dumps(record.metadata, ensure_ascii=False),
                json.dumps(record.spec.to_dict(), ensure_ascii=False),
                fp,
                record.dual_write_id,
                json.dumps(embedding),
            ),
        )
        self._conn.execute(
            "INSERT INTO memories_fts (id, type, content, source) VALUES (?, ?, ?, ?)",
            (record.id, record.type, record.content, record.source),
        )
        self._conn.commit()
        return record

    def dual_write(self, command: DualWriteCommand) -> list[MemoryRecord]:
        import uuid

        dual_id = str(uuid.uuid4())
        written: list[MemoryRecord] = []
        for kind in command.types:
            payload = command.specs.get(kind, {})
            record = MemoryRecord.create(
                memory_type=kind,
                content=command.content,
                source=command.source,
                payload=payload,
                confidence=command.confidence,
                pinned=command.pinned,
                metadata={**command.metadata, "dual_write": True},
                dual_write_id=dual_id,
                registry=self.registry,
            )
            written.append(self.save(record, allow_dual_write=True))
        return written

    def get(self, record_id: str) -> MemoryRecord | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (record_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def delete(self, record_id: str, expected_type: MemoryType | None = None) -> bool:
        record = self.get(record_id)
        if record is None:
            return False
        if expected_type is not None and record.type != expected_type:
            raise MemoryValidationError(
                f"memory_delete type {expected_type!r} does not match record {record.type!r}"
            )
        self._conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
        self._conn.execute("DELETE FROM memories_fts WHERE id = ?", (record_id,))
        self._conn.commit()
        return True

    def list(
        self,
        memory_type: MemoryType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        if memory_type is None:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        else:
            kind = parse_memory_type(memory_type)
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE type = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (kind, limit, offset),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise MemoryValidationError("search query is required")
        kind = parse_memory_type(memory_type) if memory_type is not None else None
        qvec = hashed_embedding(query)
        fts = _fts_query(query)
        params: list[Any] = [fts]
        type_sql = ""
        if kind is not None:
            type_sql = "AND memories.type = ?"
            params.append(kind)
        params.append(limit * 4)
        rows = self._conn.execute(
            f"""
            SELECT memories.*
            FROM memories
            JOIN memories_fts ON memories.id = memories_fts.id
            WHERE memories_fts MATCH ?
            {type_sql}
            LIMIT ?
            """,
            params,
        ).fetchall()
        if not rows:
            like = f"%{query.strip()}%"
            if kind is None:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE content LIKE ? OR source LIKE ? LIMIT ?",
                    (like, like, limit * 4),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE type = ? AND (content LIKE ? OR source LIKE ?) LIMIT ?",
                    (kind, like, like, limit * 4),
                ).fetchall()

        scored: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            record = self._row_to_record(row)
            embedding = json.loads(row["embedding"])
            score = _cosine(qvec, embedding)
            if query.lower() in record.content.lower():
                score += 0.15
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {**record.to_dict(), "score": round(score, 6)}
            for score, record in scored[:limit]
        ]

    def counts(self) -> dict[str, int]:
        out = {name: 0 for name in MEMORY_TYPES}
        for row in self._conn.execute(
            "SELECT type, COUNT(*) AS n FROM memories GROUP BY type"
        ):
            out[row["type"]] = int(row["n"])
        out["total"] = sum(out[name] for name in MEMORY_TYPES)
        return out

    def seed_if_empty(self, records: Iterable[MemoryRecord]) -> int:
        existing = self._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        if existing and int(existing["n"]) > 0:
            return 0
        count = 0
        for record in records:
            self.save(record)
            count += 1
        return count
