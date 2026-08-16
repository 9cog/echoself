"""Local HTTP API for ech0-mem0 / mech0. Cloud Mem0 is not required."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .model import (
    AutognosicSubjectError,
    DualWriteCommand,
    DualWriteRequired,
    MEMORY_TYPES,
    MemoryRecord,
    MemoryTypeError,
    MemoryValidationError,
    ProceduralShapeError,
    parse_memory_type,
)
from .store import MemoryStore, default_data_dir

DEFAULT_HOST = os.environ.get("MECH0_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("MECH0_PORT", "8765"))


def _json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise MemoryValidationError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise MemoryValidationError("JSON body must be an object")
    return data


def _record_from_body(body: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord.create(
        memory_type=body.get("type"),
        content=str(body.get("content") or ""),
        source=str(body.get("source") or "api"),
        payload=body.get("spec") if isinstance(body.get("spec"), dict) else body,
        confidence=float(body.get("confidence", 0.8)),
        pinned=bool(body.get("pinned", False)),
        metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
        record_id=body.get("id") if isinstance(body.get("id"), str) else None,
        created_at=body.get("created_at") if isinstance(body.get("created_at"), str) else None,
    )


class Mech0Handler(BaseHTTPRequestHandler):
    store: MemoryStore

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, exc: Exception) -> None:
        self._send(status, {"error": type(exc).__name__, "message": str(exc)})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, {})

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
            if path == "/health":
                self._send(
                    200,
                    {
                        "ok": True,
                        "service": "mech0",
                        "also_known_as": "ech0-mem0",
                        "backend": "local-sqlite",
                        "cloud_mem0_required": False,
                        "data_dir": str(self.store.data_dir),
                        "counts": self.store.counts(),
                    },
                )
                return
            if path == "/types":
                self._send(200, {"types": list(MEMORY_TYPES)})
                return
            if path == "/memories":
                memory_type = parse_memory_type(query["type"]) if query.get("type") else None
                limit = int(query.get("limit") or 50)
                offset = int(query.get("offset") or 0)
                records = [r.to_dict() for r in self.store.list(memory_type, limit, offset)]
                self._send(200, {"memories": records, "count": len(records)})
                return
            if path.startswith("/memories/"):
                record_id = path.split("/", 2)[2]
                record = self.store.get(record_id)
                if record is None:
                    self._send(404, {"error": "NotFound", "message": record_id})
                    return
                self._send(200, record.to_dict())
                return
            self._send(404, {"error": "NotFound", "message": path})
        except (MemoryTypeError, MemoryValidationError) as exc:
            self._error(400, exc)
        except Exception as exc:  # pragma: no cover - last-resort
            self._error(500, exc)

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            body = _json_body(self)
            if path == "/memories":
                record = self.store.save(_record_from_body(body))
                self._send(201, record.to_dict())
                return
            if path == "/memories/search":
                results = self.store.search(
                    str(body.get("query") or ""),
                    parse_memory_type(body["type"]) if body.get("type") else None,
                    int(body.get("limit") or 10),
                )
                self._send(200, {"results": results, "count": len(results)})
                return
            if path == "/memories/dual-write":
                types = body.get("types")
                if not isinstance(types, list) or len(types) != 2:
                    raise MemoryValidationError("dual-write requires types: [typeA, typeB]")
                command = DualWriteCommand(
                    types=(parse_memory_type(types[0]), parse_memory_type(types[1])),
                    content=str(body.get("content") or ""),
                    source=str(body.get("source") or "api"),
                    confidence=float(body.get("confidence", 0.8)),
                    pinned=bool(body.get("pinned", False)),
                    metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
                    specs=body.get("specs") if isinstance(body.get("specs"), dict) else {},
                )
                records = [r.to_dict() for r in self.store.dual_write(command)]
                self._send(201, {"dual_write_id": records[0]["dual_write_id"], "memories": records})
                return
            if path == "/instruments/memory_save":
                if "type" not in body:
                    raise MemoryTypeError("memory_save requires type")
                record = self.store.save(_record_from_body(body))
                self._send(201, {"instrument": "memory_save", "memory": record.to_dict()})
                return
            if path == "/instruments/memory_load":
                if "type" not in body:
                    raise MemoryTypeError("memory_load requires type")
                kind = parse_memory_type(body["type"])
                if body.get("id"):
                    record = self.store.get(str(body["id"]))
                    if record is None:
                        self._send(404, {"error": "NotFound", "message": body["id"]})
                        return
                    if record.type != kind:
                        raise MemoryValidationError("memory_load type does not match record")
                    self._send(200, {"instrument": "memory_load", "memories": [record.to_dict()]})
                    return
                results = self.store.search(str(body.get("query") or ""), kind, int(body.get("limit") or 10))
                self._send(200, {"instrument": "memory_load", "memories": results})
                return
            if path == "/instruments/memory_delete":
                if "type" not in body or "id" not in body:
                    raise MemoryValidationError("memory_delete requires type and id")
                deleted = self.store.delete(str(body["id"]), parse_memory_type(body["type"]))
                self._send(200, {"instrument": "memory_delete", "deleted": deleted, "id": body["id"]})
                return
            self._send(404, {"error": "NotFound", "message": path})
        except DualWriteRequired as exc:
            self._error(409, exc)
        except (
            MemoryTypeError,
            MemoryValidationError,
            AutognosicSubjectError,
            ProceduralShapeError,
        ) as exc:
            self._error(400, exc)
        except Exception as exc:  # pragma: no cover - last-resort
            self._error(500, exc)

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if not path.startswith("/memories/"):
                self._send(404, {"error": "NotFound", "message": path})
                return
            record_id = path.split("/", 2)[2]
            query = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
            expected = parse_memory_type(query["type"]) if query.get("type") else None
            deleted = self.store.delete(record_id, expected)
            if not deleted:
                self._send(404, {"error": "NotFound", "message": record_id})
                return
            self._send(200, {"deleted": True, "id": record_id})
        except (MemoryTypeError, MemoryValidationError) as exc:
            self._error(400, exc)
        except Exception as exc:  # pragma: no cover
            self._error(500, exc)


def make_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    data_dir: Path | None = None,
    seed: bool = True,
) -> ThreadingHTTPServer:
    store = MemoryStore(data_dir=data_dir)
    if seed:
        from .seed import seed_records

        added = store.seed_if_empty(seed_records())
        if added:
            sys.stderr.write(f"mech0 seeded {added} records into {store.db_path}\n")
    Mech0Handler.store = store
    server = ThreadingHTTPServer((host, port), Mech0Handler)
    server.mech0_store = store  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    data_dir = default_data_dir()
    seed = True
    i = 0
    while i < len(args):
        if args[i] in {"--host"} and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] in {"--port"} and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] in {"--data-dir"} and i + 1 < len(args):
            data_dir = Path(args[i + 1])
            i += 2
        elif args[i] == "--no-seed":
            seed = False
            i += 1
        else:
            i += 1
    server = make_server(host, port, data_dir, seed=seed)
    print(f"mech0 (ech0-mem0) listening on http://{host}:{port}")
    print(f"data: {server.mech0_store.db_path}")  # type: ignore[attr-defined]
    print("cloud Mem0 is not required")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nmech0 stopped")
    finally:
        server.mech0_store.close()  # type: ignore[attr-defined]
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
