"""Python client for the local mech0 / ech0-mem0 HTTP API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .model import MEMORY_TYPES, MemoryType, parse_memory_type

DEFAULT_URL = os.environ.get("MECH0_URL", "http://127.0.0.1:8765")


class Mech0Error(RuntimeError):
    def __init__(self, status: int, payload: Any) -> None:
        self.status = status
        self.payload = payload
        super().__init__(f"mech0 HTTP {status}: {payload}")


class Mech0Client:
    """Thin EchoSelf client: add/search/delete by required MemoryType."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> Any:
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8")) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {"message": str(exc)}
            except json.JSONDecodeError:
                payload = {"message": raw.decode("utf-8", errors="replace")}
            raise Mech0Error(exc.code, payload) from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def types(self) -> list[str]:
        return list(self._request("GET", "/types").get("types") or MEMORY_TYPES)

    def add(
        self,
        *,
        memory_type: MemoryType | str,
        content: str,
        source: str = "client",
        spec: dict[str, Any] | None = None,
        confidence: float = 0.8,
        pinned: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kind = parse_memory_type(memory_type)
        payload = {
            "type": kind,
            "content": content,
            "source": source,
            "confidence": confidence,
            "pinned": pinned,
            "metadata": metadata or {},
            **(spec or {}),
        }
        if spec:
            payload["spec"] = spec
        return self._request("POST", "/memories", payload)

    def get(self, record_id: str) -> dict[str, Any]:
        return self._request("GET", f"/memories/{record_id}")

    def list(self, memory_type: MemoryType | str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query = {"limit": limit}
        if memory_type is not None:
            query["type"] = parse_memory_type(memory_type)
        return self._request("GET", "/memories", query=query).get("memories") or []

    def search(
        self,
        query: str,
        memory_type: MemoryType | str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"query": query, "limit": limit}
        if memory_type is not None:
            body["type"] = parse_memory_type(memory_type)
        return self._request("POST", "/memories/search", body).get("results") or []

    def delete(self, record_id: str, memory_type: MemoryType | str | None = None) -> dict[str, Any]:
        query = {"type": parse_memory_type(memory_type)} if memory_type else None
        return self._request("DELETE", f"/memories/{record_id}", query=query)

    def memory_save(self, *, memory_type: MemoryType | str, content: str, **kwargs: Any) -> dict[str, Any]:
        kind = parse_memory_type(memory_type)
        return self._request(
            "POST",
            "/instruments/memory_save",
            {"type": kind, "content": content, **kwargs},
        )

    def memory_load(self, *, memory_type: MemoryType | str, query: str | None = None, record_id: str | None = None, limit: int = 10) -> dict[str, Any]:
        body: dict[str, Any] = {"type": parse_memory_type(memory_type), "limit": limit}
        if query:
            body["query"] = query
        if record_id:
            body["id"] = record_id
        return self._request("POST", "/instruments/memory_load", body)

    def memory_delete(self, *, memory_type: MemoryType | str, record_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/instruments/memory_delete",
            {"type": parse_memory_type(memory_type), "id": record_id},
        )
