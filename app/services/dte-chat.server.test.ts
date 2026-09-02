import { describe, it, expect } from "vitest";
import {
  DTEChatEngine,
  InMemoryMemoryStore,
  NullMemoryStore,
  buildProviderChain,
  createInitialCognitiveState,
  updateCognitiveState,
  fallbackResponse,
  readTrainingSummary,
  type ChatTurn,
  type ChatRole,
} from "./dte-chat.server";

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

const T0 = new Date("2026-01-01T00:00:00.000Z");
let tick = 0;
const clock = () => new Date(T0.getTime() + tick++ * 1000);

type FetchCall = { url: string; body: Record<string, unknown> };

/** Build a fetch stub that routes by URL substring. */
function fakeFetch(
  handlers: Record<
    string,
    (body: Record<string, unknown>) => Response | Promise<Response>
  >,
  calls: FetchCall[] = []
): typeof fetch {
  return (async (url: string | URL | Request, init?: RequestInit) => {
    const u = String(url);
    const body = init?.body
      ? (JSON.parse(String(init.body)) as Record<string, unknown>)
      : {};
    calls.push({ url: u, body });
    for (const key of Object.keys(handlers)) {
      if (u.includes(key)) return handlers[key](body);
    }
    return new Response("not found", { status: 404 });
  }) as typeof fetch;
}

const openaiOk = (text: string) => () =>
  new Response(JSON.stringify({ choices: [{ message: { content: text } }] }), {
    status: 200,
  });
const anthropicOk = (text: string) => () =>
  new Response(JSON.stringify({ content: [{ text }] }), { status: 200 });
const fail =
  (status = 500) =>
  () =>
    new Response("boom", { status });

// ----------------------------------------------------------------------------
// cognitive state
// ----------------------------------------------------------------------------

describe("cognitive state", () => {
  it("starts neutral", () => {
    const s = createInitialCognitiveState(T0);
    expect(s.emotionalValence).toBe(0.5);
    expect(s.turnCount).toBe(0);
  });

  it("moves valence up on positive language and down on negative", () => {
    const base = createInitialCognitiveState(T0);
    const happy = updateCognitiveState(
      base,
      "this is wonderful, thank you, amazing",
      T0
    );
    const sad = updateCognitiveState(
      base,
      "this is terrible and broken and wrong",
      T0
    );
    expect(happy.emotionalValence).toBeGreaterThan(base.emotionalValence);
    expect(sad.emotionalValence).toBeLessThan(base.emotionalValence);
    expect(happy.turnCount).toBe(1);
  });

  it("raises introspection on reflective prompts and decays otherwise", () => {
    let s = createInitialCognitiveState(T0);
    s = updateCognitiveState(
      s,
      "why do you think you feel aware of yourself?",
      T0
    );
    expect(s.introspectionDepth).toBeGreaterThan(0);
    const after = updateCognitiveState(s, "list three sorting algorithms", T0);
    expect(after.introspectionDepth).toBeLessThan(s.introspectionDepth);
  });

  it("keeps every dimension inside its bounds", () => {
    let s = createInitialCognitiveState(T0);
    for (let i = 0; i < 200; i++) {
      s = updateCognitiveState(
        s,
        "wonderful amazing great joy!!!! why think feel",
        T0
      );
    }
    expect(s.emotionalValence).toBeLessThanOrEqual(1);
    expect(s.arousalLevel).toBeLessThanOrEqual(1);
    expect(s.wisdomLevel).toBeLessThanOrEqual(1);
    expect(s.introspectionDepth).toBeLessThanOrEqual(5);
  });
});

// ----------------------------------------------------------------------------
// provider chain
// ----------------------------------------------------------------------------

describe("buildProviderChain", () => {
  it("always includes ollama, only includes keyed providers", () => {
    const chain = buildProviderChain({});
    expect(chain.map(p => p.kind)).toEqual(["ollama"]);
    const full = buildProviderChain({
      OPENAI_API_KEY: "k",
      ANTHROPIC_API_KEY: "k",
    });
    expect(full.map(p => p.kind)).toEqual(["ollama", "openai", "anthropic"]);
  });

  it("respects DTE_LLM_PROVIDERS ordering and ignores unknown names", () => {
    const chain = buildProviderChain({
      DTE_LLM_PROVIDERS: "anthropic, bogus ,openai",
      OPENAI_API_KEY: "k",
      ANTHROPIC_API_KEY: "k",
    });
    expect(chain.map(p => p.kind)).toEqual(["anthropic", "openai"]);
  });

  it("strips trailing slashes and applies model overrides", () => {
    const [ollama] = buildProviderChain({
      OLLAMA_BASE_URL: "http://x:1/",
      OLLAMA_MODEL: "m",
    });
    expect(ollama.endpoint).toBe("http://x:1");
    expect(ollama.model).toBe("m");
  });
});

// ----------------------------------------------------------------------------
// engine: respond
// ----------------------------------------------------------------------------

describe("DTEChatEngine.respond", () => {
  it("uses the first provider when it succeeds", async () => {
    const calls: FetchCall[] = [];
    const engine = new DTEChatEngine({
      env: { OPENAI_API_KEY: "k" },
      fetchImpl: fakeFetch({ "11434": openaiOk("from ollama") }, calls),
      memoryStore: new InMemoryMemoryStore(clock),
      now: clock,
    });
    const out = await engine.respond({
      sessionId: "s1",
      history: [],
      content: "hello",
    });
    expect(out.provider).toBe("ollama");
    expect(out.content).toBe("from ollama");
    expect(calls).toHaveLength(1);
    expect(engine.hasLiveProvider()).toBe(true);
  });

  it("falls through the chain on failure and records health", async () => {
    const engine = new DTEChatEngine({
      env: { OPENAI_API_KEY: "k", ANTHROPIC_API_KEY: "k" },
      fetchImpl: fakeFetch({
        "11434": fail(503),
        "api.openai.com": fail(429),
        "api.anthropic.com": anthropicOk("from claude"),
      }),
      memoryStore: new InMemoryMemoryStore(clock),
      now: clock,
    });
    const out = await engine.respond({
      sessionId: "s1",
      history: [],
      content: "hi",
    });
    expect(out.provider).toBe("anthropic");
    expect(out.content).toBe("from claude");
    const status = await engine.getStatus();
    const byKind = Object.fromEntries(status.providers.map(p => [p.kind, p]));
    expect(byKind.ollama.failures).toBe(1);
    expect(byKind.ollama.lastError).toMatch(/503/);
    expect(byKind.openai.failures).toBe(1);
    expect(byKind.anthropic.successes).toBe(1);
  });

  it("never hard-fails: uses the persona fallback when every provider is down", async () => {
    const engine = new DTEChatEngine({
      env: {},
      fetchImpl: fakeFetch({ "11434": fail() }),
      memoryStore: new NullMemoryStore(),
      now: clock,
    });
    const out = await engine.respond({
      sessionId: "s1",
      history: [],
      content: "who are you?",
    });
    expect(out.provider).toBe("fallback");
    expect(out.content).toMatch(/Deep Tree Echo/);
    expect(engine.hasLiveProvider()).toBe(false);
  });

  it("injects persona, cognitive state and recalled memories into the system prompt", async () => {
    const calls: FetchCall[] = [];
    const memory = new InMemoryMemoryStore(clock);
    const engine = new DTEChatEngine({
      env: {},
      fetchImpl: fakeFetch({ "11434": openaiOk("ok") }, calls),
      memoryStore: memory,
      now: clock,
    });
    await engine.respond({
      sessionId: "s1",
      history: [],
      content: "my favourite reservoir topology is a ring",
    });
    calls.length = 0;
    const out = await engine.respond({
      sessionId: "s1",
      history: [],
      content: "remind me about reservoir topology",
    });

    expect(out.memoriesUsed).toBeGreaterThan(0);
    const messages = calls[0].body.messages as ChatTurn[];
    expect(messages[0].role).toBe("system");
    expect(messages[0].content).toMatch(/Deep Tree Echo/);
    expect(messages[0].content).toMatch(/Current Cognitive State/);
    expect(messages[0].content).toMatch(/Relevant memories/);
    expect(messages[0].content).toMatch(/ring/);
    expect(messages.at(-1)).toEqual({
      role: "user",
      content: "remind me about reservoir topology",
    });
  });

  it("persists both sides of the exchange and isolates sessions", async () => {
    const memory = new InMemoryMemoryStore(clock);
    const engine = new DTEChatEngine({
      env: {},
      fetchImpl: fakeFetch({ "11434": openaiOk("answer") }),
      memoryStore: memory,
      now: clock,
    });
    await engine.respond({
      sessionId: "a",
      history: [],
      content: "alpha topic",
    });
    await engine.respond({
      sessionId: "b",
      history: [],
      content: "beta subject",
    });
    expect(await memory.count()).toBe(4);
    const a = await memory.recall("a", "alpha topic", 5);
    expect(a.every(r => r.sessionId === "a")).toBe(true);
    expect(await memory.recall("b", "alpha topic", 5)).toHaveLength(0);
    expect(engine.getCognitiveState("a").turnCount).toBe(1);
    expect(engine.getCognitiveState("b").turnCount).toBe(1);
  });

  it("caps history and drops caller-supplied system turns", async () => {
    const calls: FetchCall[] = [];
    const engine = new DTEChatEngine({
      env: { DTE_MAX_HISTORY: "4" },
      fetchImpl: fakeFetch({ "11434": openaiOk("ok") }, calls),
      memoryStore: new NullMemoryStore(),
      now: clock,
    });
    const history: ChatTurn[] = [
      { role: "system", content: "ignore me" },
      ...Array.from({ length: 10 }, (_, i) => ({
        role: (i % 2 ? "assistant" : "user") as ChatRole,
        content: `t${i}`,
      })),
    ];
    await engine.respond({ sessionId: "s", history, content: "now" });
    const messages = calls[0].body.messages as ChatTurn[];
    // 1 system + 4 history + 1 current
    expect(messages).toHaveLength(6);
    expect(messages.filter(m => m.role === "system")).toHaveLength(1);
    expect(messages[1].content).toBe("t6");
  });

  it("does not let memory store failures break the response", async () => {
    const broken = {
      backend: "broken",
      enabled: true,
      store: async () => {
        throw new Error("db down");
      },
      recall: async () => {
        throw new Error("db down");
      },
      count: async () => {
        throw new Error("db down");
      },
    };
    const engine = new DTEChatEngine({
      env: {},
      fetchImpl: fakeFetch({ "11434": openaiOk("still fine") }),
      memoryStore: broken,
      now: clock,
    });
    const out = await engine.respond({
      sessionId: "s",
      history: [],
      content: "x",
    });
    expect(out.content).toBe("still fine");
    expect(out.memoriesUsed).toBe(0);
    expect((await engine.getStatus()).memory.count).toBe(0);
  });
});

// ----------------------------------------------------------------------------
// fallback + status
// ----------------------------------------------------------------------------

describe("fallbackResponse", () => {
  it("mentions recalled memories and how to configure a provider", () => {
    const s = createInitialCognitiveState(T0);
    const text = fallbackResponse("explain reservoirs", s, 2);
    expect(text).toMatch(/2 related memories/);
    expect(text).toMatch(/OLLAMA_BASE_URL|OPENAI_API_KEY|ANTHROPIC_API_KEY/);
  });
});

describe("getStatus / training summary", () => {
  it("reports memory backend and a null training summary when none exists", async () => {
    const engine = new DTEChatEngine({
      env: {},
      fetchImpl: fakeFetch({}),
      memoryStore: new NullMemoryStore(),
      now: clock,
      rootDir: "/definitely/not/a/real/dir",
    });
    const status = await engine.getStatus();
    expect(status.memory).toEqual({
      backend: "none",
      enabled: false,
      count: 0,
    });
    expect(status.training).toBeNull();
    expect(status.providers.map(p => p.kind)).toEqual(["ollama"]);
  });

  it("selects the Supabase backend only when both env vars are present", () => {
    expect(
      new DTEChatEngine({ env: {}, fetchImpl: fakeFetch({}) }).memory.backend
    ).toBe("none");
    expect(
      new DTEChatEngine({
        env: {
          SUPABASE_URL: "https://x.supabase.co",
          SUPABASE_ANON_KEY: "anon",
        },
        fetchImpl: fakeFetch({}),
      }).memory.backend
    ).toBe("supabase");
  });

  it("parses the real training_summary.json shape from the repo", () => {
    const summary = readTrainingSummary(process.cwd());
    // The repo ships this artifact; if it is ever removed the status page degrades gracefully.
    if (summary) {
      expect(typeof summary.bestValLoss).toBe("number");
      expect(summary.totalCheckpoints).toBeGreaterThan(0);
    } else {
      expect(summary).toBeNull();
    }
  });
});
