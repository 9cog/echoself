/**
 * Thin EchoSelf client for local mech0 / ech0-mem0.
 * Cloud Mem0 is not required. Memory type is always required.
 */

import type { MemoryRecord, MemorySaveInput, MemoryType } from "../types/mech0";

const DEFAULT_URL = "http://127.0.0.1:8765";

export class Mech0ClientError extends Error {
  constructor(
    public status: number,
    public payload: unknown
  ) {
    super(`mech0 HTTP ${status}: ${JSON.stringify(payload)}`);
    this.name = "Mech0ClientError";
  }
}

export class Mech0Client {
  constructor(private readonly baseUrl: string = DEFAULT_URL) {}

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    query?: Record<string, string | number | undefined>
  ): Promise<T> {
    const params = new URLSearchParams();
    if (query) {
      for (const [key, value] of Object.entries(query)) {
        if (value !== undefined) params.set(key, String(value));
      }
    }
    const qs = params.toString();
    const url = `${this.baseUrl}${path}${qs ? `?${qs}` : ""}`;
    const response = await fetch(url, {
      method,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const payload = response.status === 204 ? {} : await response.json();
    if (!response.ok) {
      throw new Mech0ClientError(response.status, payload);
    }
    return payload as T;
  }

  health() {
    return this.request<{
      ok: boolean;
      service: string;
      cloud_mem0_required: boolean;
      data_dir: string;
      counts: Record<string, number>;
    }>("GET", "/health");
  }

  add(input: MemorySaveInput) {
    return this.request<MemoryRecord>("POST", "/memories", input);
  }

  list(type?: MemoryType, limit = 50) {
    return this.request<{ memories: MemoryRecord[]; count: number }>(
      "GET",
      "/memories",
      undefined,
      { type, limit }
    );
  }

  search(query: string, type?: MemoryType, limit = 10) {
    return this.request<{ results: MemoryRecord[]; count: number }>(
      "POST",
      "/memories/search",
      { query, type, limit }
    );
  }

  delete(id: string, type?: MemoryType) {
    return this.request<{ deleted: boolean; id: string }>(
      "DELETE",
      `/memories/${id}`,
      undefined,
      { type }
    );
  }

  memorySave(input: MemorySaveInput) {
    return this.request<{ instrument: "memory_save"; memory: MemoryRecord }>(
      "POST",
      "/instruments/memory_save",
      input
    );
  }

  memoryLoad(type: MemoryType, query?: string, id?: string, limit = 10) {
    return this.request<{
      instrument: "memory_load";
      memories: MemoryRecord[];
    }>("POST", "/instruments/memory_load", { type, query, id, limit });
  }

  memoryDelete(type: MemoryType, id: string) {
    return this.request<{
      instrument: "memory_delete";
      deleted: boolean;
      id: string;
    }>("POST", "/instruments/memory_delete", { type, id });
  }
}

export const getMech0Client = (baseUrl?: string) =>
  new Mech0Client(baseUrl ?? DEFAULT_URL);
