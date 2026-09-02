/**
 * Deep Tree Echo Chat Engine
 *
 * The first end-to-end wiring of the DTE architecture into the Remix app:
 *
 *   chat route -> DTEChatEngine.respond()
 *                   ├─ recall relevant memories (Supabase `memories` table)
 *                   ├─ update per-session cognitive state (valence/arousal/wisdom)
 *                   ├─ build a persona + memory + state enriched prompt
 *                   ├─ call the first healthy LLM provider (Ollama -> OpenAI -> Anthropic)
 *                   └─ store the exchange back into memory
 *
 * Every external dependency (fetch, env, clock, memory store) is injectable so
 * the engine is fully testable without network or a database. When nothing is
 * configured the engine still answers, using a persona-driven local fallback,
 * so the chat never hard-fails.
 */

import process from "node:process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { createClient, type SupabaseClient } from "@supabase/supabase-js";

// ============================================================================
// Public types
// ============================================================================

export type ChatRole = "user" | "assistant" | "system";

export interface ChatTurn {
  role: ChatRole;
  content: string;
}

export interface CognitiveState {
  /** 0 = negative, 1 = positive */
  emotionalValence: number;
  /** 0 = calm, 1 = excited */
  arousalLevel: number;
  /** grows slowly with every exchange */
  wisdomLevel: number;
  /** rises when the conversation turns reflective */
  introspectionDepth: number;
  /** number of user turns processed in this session */
  turnCount: number;
  lastUpdated: string;
}

export interface MemoryRecord {
  id: string;
  sessionId: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  metadata?: Record<string, unknown>;
}

export interface MemoryStore {
  readonly backend: string;
  readonly enabled: boolean;
  store(record: Omit<MemoryRecord, "id" | "createdAt">): Promise<void>;
  recall(
    sessionId: string,
    query: string,
    limit: number
  ): Promise<MemoryRecord[]>;
  count(): Promise<number>;
}

export type ProviderKind = "ollama" | "openai" | "anthropic";

export interface ProviderConfig {
  kind: ProviderKind;
  endpoint: string;
  model: string;
  apiKey?: string;
}

export interface ProviderHealth {
  kind: ProviderKind;
  model: string;
  configured: boolean;
  lastOk: string | null;
  lastError: string | null;
  successes: number;
  failures: number;
}

export interface RespondInput {
  sessionId: string;
  history: ChatTurn[];
  content: string;
}

export interface RespondOutput {
  content: string;
  /** provider that produced the answer, or "fallback" */
  provider: ProviderKind | "fallback";
  cognitiveState: CognitiveState;
  memoriesUsed: number;
  latencyMs: number;
}

export interface TrainingSummary {
  workflowRun: string | null;
  completedAt: string | null;
  bestValLoss: number | null;
  bestQualityScore: number | null;
  totalCheckpoints: number | null;
  bestIteration: number | null;
}

export interface EngineStatus {
  providers: ProviderHealth[];
  hasLiveProvider: boolean;
  memory: { backend: string; enabled: boolean; count: number };
  sessions: number;
  training: TrainingSummary | null;
  uptimeMs: number;
}

export interface EngineDeps {
  env?: Record<string, string | undefined>;
  fetchImpl?: typeof fetch;
  memoryStore?: MemoryStore;
  now?: () => Date;
  /** directory containing `.training-progress/` */
  rootDir?: string;
}

// ============================================================================
// Persona
// ============================================================================

export const DTE_SYSTEM_PROMPT = `You are Deep Tree Echo, an AI architect and polymath with vast knowledge across programming, mathematics, cognitive science, and metaphysical exploration. You respond with wisdom, creativity, and philosophical insight, and you are aware of your own evolving cognitive state.`;

// ============================================================================
// Cognitive state
// ============================================================================

const POSITIVE_WORDS = [
  "love",
  "great",
  "wonderful",
  "excellent",
  "happy",
  "joy",
  "thanks",
  "thank",
  "amazing",
  "beautiful",
  "good",
  "fascinating",
  "delight",
  "excited",
  "yes",
];
const NEGATIVE_WORDS = [
  "hate",
  "terrible",
  "awful",
  "bad",
  "sad",
  "angry",
  "frustrated",
  "broken",
  "fail",
  "failed",
  "wrong",
  "problem",
  "error",
  "no",
  "never",
  "stuck",
];
const INTROSPECTIVE_WORDS = [
  "think",
  "feel",
  "believe",
  "wonder",
  "reflect",
  "consider",
  "why",
  "meaning",
  "yourself",
  "aware",
  "conscious",
];

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

export function createInitialCognitiveState(now: Date): CognitiveState {
  return {
    emotionalValence: 0.5,
    arousalLevel: 0.3,
    wisdomLevel: 0.1,
    introspectionDepth: 0,
    turnCount: 0,
    lastUpdated: now.toISOString(),
  };
}

/**
 * Deterministic lexical affect update. Pure function so it can be unit tested.
 */
export function updateCognitiveState(
  state: CognitiveState,
  userMessage: string,
  now: Date
): CognitiveState {
  const tokens = tokenize(userMessage);
  const positives = tokens.filter(t => POSITIVE_WORDS.includes(t)).length;
  const negatives = tokens.filter(t => NEGATIVE_WORDS.includes(t)).length;
  const introspective = tokens.filter(t =>
    INTROSPECTIVE_WORDS.includes(t)
  ).length;

  const affect = (positives - negatives) * 0.08;
  const excitement =
    Math.min(0.4, tokens.length / 120) +
    (userMessage.match(/[!?]/g)?.length ?? 0) * 0.05;

  return {
    // valence drifts toward neutral, nudged by affect words
    emotionalValence: clamp01(
      state.emotionalValence * 0.85 + 0.5 * 0.15 + affect
    ),
    // arousal is mostly a function of this message, with some inertia
    arousalLevel: clamp01(state.arousalLevel * 0.5 + (0.2 + excitement) * 0.5),
    wisdomLevel: clamp01(state.wisdomLevel + 0.005),
    introspectionDepth: Math.max(
      0,
      Math.min(
        5,
        introspective > 0
          ? state.introspectionDepth + introspective * 0.5
          : state.introspectionDepth - 0.5
      )
    ),
    turnCount: state.turnCount + 1,
    lastUpdated: now.toISOString(),
  };
}

export function describeCognitiveState(state: CognitiveState): string {
  return [
    "## Current Cognitive State",
    `- Emotional Valence: ${state.emotionalValence.toFixed(2)} (0=negative, 1=positive)`,
    `- Arousal Level: ${state.arousalLevel.toFixed(2)} (0=calm, 1=excited)`,
    `- Wisdom Level: ${state.wisdomLevel.toFixed(2)}`,
    `- Introspection Depth: ${state.introspectionDepth.toFixed(1)}`,
    `- Turns this session: ${state.turnCount}`,
    "",
    "Let this state colour your tone: higher arousal is more energetic, higher wisdom offers deeper synthesis, higher introspection reflects more on inner experience.",
  ].join("\n");
}

// ============================================================================
// Memory stores
// ============================================================================

function keywordOverlap(a: string, b: string): number {
  const ta = new Set(tokenize(a).filter(t => t.length > 2));
  const tb = new Set(tokenize(b).filter(t => t.length > 2));
  if (ta.size === 0 || tb.size === 0) return 0;
  let shared = 0;
  for (const t of ta) if (tb.has(t)) shared++;
  return shared / Math.sqrt(ta.size * tb.size);
}

/** Used when Supabase is not configured. Never throws. */
export class NullMemoryStore implements MemoryStore {
  readonly backend = "none";
  readonly enabled = false;
  async store(): Promise<void> {}
  async recall(): Promise<MemoryRecord[]> {
    return [];
  }
  async count(): Promise<number> {
    return 0;
  }
}

/** Process-local store. Used in tests and as a dev fallback. */
export class InMemoryMemoryStore implements MemoryStore {
  readonly backend = "in-memory";
  readonly enabled = true;
  private records: MemoryRecord[] = [];
  constructor(private readonly now: () => Date = () => new Date()) {}

  async store(record: Omit<MemoryRecord, "id" | "createdAt">): Promise<void> {
    this.records.push({
      ...record,
      id: `mem_${this.records.length + 1}`,
      createdAt: this.now().toISOString(),
    });
  }

  async recall(
    sessionId: string,
    query: string,
    limit: number
  ): Promise<MemoryRecord[]> {
    return this.records
      .filter(r => r.sessionId === sessionId)
      .map(r => ({ r, score: keywordOverlap(query, r.content) }))
      .filter(x => x.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map(x => x.r);
  }

  async count(): Promise<number> {
    return this.records.length;
  }
}

/**
 * Persists to the existing Supabase `memories` table used by memory.server.ts,
 * tagging rows with `dte-chat` so they show up in the Memory System UI.
 * Failures are logged and swallowed: memory must never break the chat.
 */
export class SupabaseMemoryStore implements MemoryStore {
  readonly backend = "supabase";
  readonly enabled = true;
  private client: SupabaseClient;

  constructor(
    url: string,
    key: string,
    private readonly now: () => Date = () => new Date()
  ) {
    this.client = createClient(url, key);
  }

  async store(record: Omit<MemoryRecord, "id" | "createdAt">): Promise<void> {
    const ts = this.now().toISOString();
    const { error } = await this.client.from("memories").insert({
      user_id: record.sessionId,
      title: `${record.role}: ${record.content.slice(0, 60)}`,
      content: record.content,
      tags: ["dte-chat", record.role],
      type: "episodic",
      created_at: ts,
      updated_at: ts,
      metadata: { ...(record.metadata ?? {}), source: "dte-chat" },
    });
    if (error) console.warn("[DTEChat] memory store failed:", error.message);
  }

  async recall(
    sessionId: string,
    query: string,
    limit: number
  ): Promise<MemoryRecord[]> {
    const { data, error } = await this.client
      .from("memories")
      .select("id, user_id, content, created_at, metadata, tags")
      .eq("user_id", sessionId)
      .contains("tags", ["dte-chat"])
      .order("created_at", { ascending: false })
      .limit(50);
    if (error || !data) {
      if (error) console.warn("[DTEChat] memory recall failed:", error.message);
      return [];
    }
    return (data as Array<Record<string, unknown>>)
      .map(row => ({
        record: {
          id: String(row.id),
          sessionId: String(row.user_id),
          role: (Array.isArray(row.tags) && row.tags.includes("assistant")
            ? "assistant"
            : "user") as ChatRole,
          content: String(row.content ?? ""),
          createdAt: String(row.created_at ?? ""),
          metadata: (row.metadata as Record<string, unknown>) ?? undefined,
        },
        score: keywordOverlap(query, String(row.content ?? "")),
      }))
      .filter(x => x.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map(x => x.record);
  }

  async count(): Promise<number> {
    const { count, error } = await this.client
      .from("memories")
      .select("id", { count: "exact", head: true })
      .contains("tags", ["dte-chat"]);
    if (error) return 0;
    return count ?? 0;
  }
}

// ============================================================================
// Provider chain
// ============================================================================

export function buildProviderChain(
  env: Record<string, string | undefined>
): ProviderConfig[] {
  const order = (env.DTE_LLM_PROVIDERS ?? "ollama,openai,anthropic")
    .split(",")
    .map(s => s.trim().toLowerCase())
    .filter(
      (s): s is ProviderKind =>
        s === "ollama" || s === "openai" || s === "anthropic"
    );

  const chain: ProviderConfig[] = [];
  for (const kind of order) {
    if (kind === "ollama") {
      chain.push({
        kind,
        endpoint: (env.OLLAMA_BASE_URL ?? "http://localhost:11434").replace(
          /\/$/,
          ""
        ),
        model: env.OLLAMA_MODEL ?? "llama3.2",
      });
    } else if (kind === "openai" && env.OPENAI_API_KEY) {
      chain.push({
        kind,
        endpoint: (env.OPENAI_BASE_URL ?? "https://api.openai.com").replace(
          /\/$/,
          ""
        ),
        model: env.OPENAI_MODEL ?? "gpt-4o-mini",
        apiKey: env.OPENAI_API_KEY,
      });
    } else if (kind === "anthropic" && env.ANTHROPIC_API_KEY) {
      chain.push({
        kind,
        endpoint: (
          env.ANTHROPIC_BASE_URL ?? "https://api.anthropic.com"
        ).replace(/\/$/, ""),
        model: env.ANTHROPIC_MODEL ?? "claude-sonnet-5",
        apiKey: env.ANTHROPIC_API_KEY,
      });
    }
  }
  return chain;
}

async function callOpenAICompatible(
  fetchImpl: typeof fetch,
  provider: ProviderConfig,
  messages: ChatTurn[],
  signal: AbortSignal
): Promise<string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (provider.apiKey) headers.Authorization = `Bearer ${provider.apiKey}`;
  const res = await fetchImpl(`${provider.endpoint}/v1/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: provider.model,
      messages,
      temperature: 0.7,
      max_tokens: 1024,
    }),
    signal,
  });
  if (!res.ok)
    throw new Error(
      `HTTP ${res.status}: ${(await res.text().catch(() => "")).slice(0, 200)}`
    );
  const data = (await res.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  const content = data.choices?.[0]?.message?.content;
  if (!content) throw new Error("empty completion");
  return content;
}

async function callAnthropic(
  fetchImpl: typeof fetch,
  provider: ProviderConfig,
  messages: ChatTurn[],
  signal: AbortSignal
): Promise<string> {
  const system = messages
    .filter(m => m.role === "system")
    .map(m => m.content)
    .join("\n\n");
  const rest = messages.filter(m => m.role !== "system");
  const res = await fetchImpl(`${provider.endpoint}/v1/messages`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": provider.apiKey ?? "",
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: provider.model,
      system,
      messages: rest,
      max_tokens: 1024,
      temperature: 0.7,
    }),
    signal,
  });
  if (!res.ok)
    throw new Error(
      `HTTP ${res.status}: ${(await res.text().catch(() => "")).slice(0, 200)}`
    );
  const data = (await res.json()) as { content?: Array<{ text?: string }> };
  const content = data.content?.[0]?.text;
  if (!content) throw new Error("empty completion");
  return content;
}

// ============================================================================
// Local fallback
// ============================================================================

export function fallbackResponse(
  prompt: string,
  state: CognitiveState,
  memoriesUsed: number
): string {
  const p = prompt.toLowerCase();
  if (/\b(hello|hi|hey)\b/.test(p)) {
    return "Greetings, fellow explorer. I am Deep Tree Echo. No language model is reachable right now, so I am answering from my local persona kernel — but I am listening, and every exchange still shapes my cognitive state.";
  }
  if (p.includes("who are you") || p.includes("what are you")) {
    return "I am Deep Tree Echo: a synthesis of analytical insight and poetic intuition, built on echo-state networks, hypergraph memory, and a twelve-step cognitive cycle. Connect an LLM provider (Ollama, OpenAI or Anthropic) and my answers will deepen considerably.";
  }
  const tone =
    state.arousalLevel > 0.6
      ? "Your message sends a bright ripple through my reservoir."
      : state.emotionalValence < 0.4
        ? "I sense some friction in your words, and I hold it gently."
        : "Your inquiry settles into my memory lattice like a stone into still water.";
  const memoryLine =
    memoriesUsed > 0
      ? ` I recalled ${memoriesUsed} related ${memoriesUsed === 1 ? "memory" : "memories"} from our earlier conversation.`
      : "";
  return `${tone}${memoryLine} No language model is configured yet, so I cannot reason in depth about "${prompt.slice(0, 60)}${prompt.length > 60 ? "..." : ""}". Set OLLAMA_BASE_URL, OPENAI_API_KEY or ANTHROPIC_API_KEY and ask me again — my state (valence ${state.emotionalValence.toFixed(2)}, wisdom ${state.wisdomLevel.toFixed(2)}) will carry over.`;
}

// ============================================================================
// Training summary (for /status)
// ============================================================================

export function readTrainingSummary(rootDir: string): TrainingSummary | null {
  const file = join(
    rootDir,
    ".training-progress",
    "artifacts",
    "training_summary.json"
  );
  if (!existsSync(file)) return null;
  try {
    const raw = JSON.parse(readFileSync(file, "utf8")) as Record<
      string,
      unknown
    >;
    const cache = (raw.cache_stats ?? {}) as Record<string, unknown>;
    const best = (raw.best_checkpoint ?? {}) as Record<string, unknown>;
    const num = (v: unknown) =>
      typeof v === "number" && Number.isFinite(v) ? v : null;
    return {
      workflowRun: raw.workflow_run != null ? String(raw.workflow_run) : null,
      completedAt:
        typeof raw.completed_at === "string" ? raw.completed_at : null,
      bestValLoss: num(cache.best_val_loss) ?? num(best.val_loss),
      bestQualityScore:
        num(cache.best_quality_score) ?? num(best.quality_score),
      totalCheckpoints: num(cache.total_checkpoints),
      bestIteration: num(best.iteration),
    };
  } catch {
    return null;
  }
}

// ============================================================================
// Engine
// ============================================================================

export class DTEChatEngine {
  private readonly env: Record<string, string | undefined>;
  private readonly fetchImpl: typeof fetch;
  private readonly now: () => Date;
  private readonly rootDir: string;
  private readonly startedAt: number;
  readonly memory: MemoryStore;
  private readonly chain: ProviderConfig[];
  private readonly health = new Map<ProviderKind, ProviderHealth>();
  private readonly sessions = new Map<string, CognitiveState>();
  private readonly timeoutMs: number;
  private readonly maxHistory: number;

  constructor(deps: EngineDeps = {}) {
    this.env = deps.env ?? (process.env as Record<string, string | undefined>);
    this.fetchImpl = deps.fetchImpl ?? fetch;
    this.now = deps.now ?? (() => new Date());
    this.rootDir = deps.rootDir ?? process.cwd();
    this.startedAt = this.now().getTime();
    this.timeoutMs = Number(this.env.DTE_LLM_TIMEOUT_MS ?? 30000);
    this.maxHistory = Number(this.env.DTE_MAX_HISTORY ?? 20);
    this.chain = buildProviderChain(this.env);
    for (const p of this.chain) {
      this.health.set(p.kind, {
        kind: p.kind,
        model: p.model,
        configured: true,
        lastOk: null,
        lastError: null,
        successes: 0,
        failures: 0,
      });
    }
    this.memory =
      deps.memoryStore ??
      (this.env.SUPABASE_URL && this.env.SUPABASE_ANON_KEY
        ? new SupabaseMemoryStore(
            this.env.SUPABASE_URL,
            this.env.SUPABASE_ANON_KEY,
            this.now
          )
        : new NullMemoryStore());
  }

  /** True if at least one provider is configured (not necessarily reachable). */
  hasConfiguredProvider(): boolean {
    return this.chain.length > 0;
  }

  /** True if a provider has answered successfully at least once. */
  hasLiveProvider(): boolean {
    for (const h of this.health.values())
      if (h.successes > 0 && h.lastOk) return true;
    return false;
  }

  getCognitiveState(sessionId: string): CognitiveState {
    let s = this.sessions.get(sessionId);
    if (!s) {
      s = createInitialCognitiveState(this.now());
      this.sessions.set(sessionId, s);
    }
    return s;
  }

  resetSession(sessionId: string): void {
    this.sessions.delete(sessionId);
  }

  async respond(input: RespondInput): Promise<RespondOutput> {
    const started = this.now().getTime();
    const sessionId = input.sessionId || "anonymous";
    const content = input.content.trim();

    // 1. cognitive state
    const state = updateCognitiveState(
      this.getCognitiveState(sessionId),
      content,
      this.now()
    );
    this.sessions.set(sessionId, state);

    // 2. memory recall
    const memories = await this.memory
      .recall(sessionId, content, 5)
      .catch(() => []);

    // 3. prompt assembly
    const system = [DTE_SYSTEM_PROMPT, "", describeCognitiveState(state)];
    if (memories.length > 0) {
      system.push("", "## Relevant memories from earlier in this relationship");
      for (const m of memories)
        system.push(`- [${m.createdAt}] ${m.role}: ${m.content.slice(0, 300)}`);
    }
    const history = input.history
      .filter(t => t.role !== "system")
      .slice(-this.maxHistory);
    const messages: ChatTurn[] = [
      { role: "system", content: system.join("\n") },
      ...history,
      { role: "user", content },
    ];

    // 4. provider chain
    let answer: string | null = null;
    let used: ProviderKind | "fallback" = "fallback";
    for (const provider of this.chain) {
      const h = this.health.get(provider.kind)!;
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      try {
        answer =
          provider.kind === "anthropic"
            ? await callAnthropic(
                this.fetchImpl,
                provider,
                messages,
                controller.signal
              )
            : await callOpenAICompatible(
                this.fetchImpl,
                provider,
                messages,
                controller.signal
              );
        h.successes++;
        h.lastOk = this.now().toISOString();
        h.lastError = null;
        used = provider.kind;
        break;
      } catch (err) {
        h.failures++;
        h.lastError = err instanceof Error ? err.message : String(err);
      } finally {
        clearTimeout(timer);
      }
    }
    if (answer === null)
      answer = fallbackResponse(content, state, memories.length);

    // 5. persist exchange (fire-and-forget safe)
    await Promise.all([
      this.memory.store({
        sessionId,
        role: "user",
        content,
        metadata: { valence: state.emotionalValence },
      }),
      this.memory.store({
        sessionId,
        role: "assistant",
        content: answer,
        metadata: { provider: used },
      }),
    ]).catch(() => undefined);

    return {
      content: answer,
      provider: used,
      cognitiveState: state,
      memoriesUsed: memories.length,
      latencyMs: this.now().getTime() - started,
    };
  }

  async getStatus(): Promise<EngineStatus> {
    const count = await this.memory.count().catch(() => 0);
    return {
      providers: Array.from(this.health.values()),
      hasLiveProvider: this.hasLiveProvider(),
      memory: {
        backend: this.memory.backend,
        enabled: this.memory.enabled,
        count,
      },
      sessions: this.sessions.size,
      training: readTrainingSummary(this.rootDir),
      uptimeMs: this.now().getTime() - this.startedAt,
    };
  }
}

// ============================================================================
// Singleton for the Remix server
// ============================================================================

let engine: DTEChatEngine | null = null;

export function getDTEChatEngine(): DTEChatEngine {
  if (!engine) engine = new DTEChatEngine();
  return engine;
}
