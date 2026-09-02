// Tests for the Deep Tree Echo chat engine.
//
// Runs on Node's built-in test runner so no extra dependency is needed:
//   npm test  ->  node --experimental-transform-types --test "app/**/*.spec.ts"
// (Node >= 22.7 for type stripping; CI runs lint/typecheck, not this script.)
// Named *.spec.ts on purpose: the Deno workflow treats *.test.ts / *_test.ts as
// Deno tests and would try to run this Node test under `deno test`.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import process from "node:process";
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
} from "./dte-chat.server.ts";

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
    assert.equal(s.emotionalValence, 0.5);
    assert.equal(s.turnCount, 0);
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
    assert.ok(happy.emotionalValence > base.emotionalValence);
    assert.ok(sad.emotionalValence < base.emotionalValence);
    assert.equal(happy.turnCount, 1);
  });

  it("raises introspection on reflective prompts and decays otherwise", () => {
    let s = createInitialCognitiveState(T0);
    s = updateCognitiveState(
      s,
      "why do you think you feel aware of yourself?",
      T0
    );
    assert.ok(s.introspectionDepth > 0);
    const after = updateCognitiveState(s, "list three sorting algorithms", T0);
    assert.ok(after.introspectionDepth < s.introspectionDepth);
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
    assert.ok(s.emotionalValence <= 1);
    assert.ok(s.arousalLevel <= 1);
    assert.ok(s.wisdomLevel <= 1);
    assert.ok(s.introspectionDepth <= 5);
  });
});

// ----------------------------------------------------------------------------
// provider chain
// ----------------------------------------------------------------------------

describe("buildProviderChain", () => {
  it("always includes ollama, only includes keyed providers", () => {
    const chain = buildProviderChain({});
    assert.deepEqual(
      chain.map(p => p.kind),
      ["ollama"]
    );
    const full = buildProviderChain({
      OPENAI_API_KEY: "k",
      ANTHROPIC_API_KEY: "k",
    });
    assert.deepEqual(
      full.map(p => p.kind),
      ["ollama", "openai", "anthropic"]
    );
  });

  it("respects DTE_LLM_PROVIDERS ordering and ignores unknown names", () => {
    const chain = buildProviderChain({
      DTE_LLM_PROVIDERS: "anthropic, bogus ,openai",
      OPENAI_API_KEY: "k",
      ANTHROPIC_API_KEY: "k",
    });
    assert.deepEqual(
      chain.map(p => p.kind),
      ["anthropic", "openai"]
    );
  });

  it("strips trailing slashes and applies model overrides", () => {
    const [ollama] = buildProviderChain({
      OLLAMA_BASE_URL: "http://x:1/",
      OLLAMA_MODEL: "m",
    });
    assert.equal(ollama.endpoint, "http://x:1");
    assert.equal(ollama.model, "m");
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
    assert.equal(out.provider, "ollama");
    assert.equal(out.content, "from ollama");
    assert.equal(calls.length, 1);
    assert.equal(engine.hasLiveProvider(), true);
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
    assert.equal(out.provider, "anthropic");
    assert.equal(out.content, "from claude");
    const status = await engine.getStatus();
    const byKind = Object.fromEntries(status.providers.map(p => [p.kind, p]));
    assert.equal(byKind.ollama.failures, 1);
    assert.match(byKind.ollama.lastError ?? "", /503/);
    assert.equal(byKind.openai.failures, 1);
    assert.equal(byKind.anthropic.successes, 1);
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
    assert.equal(out.provider, "fallback");
    assert.match(out.content, /Deep Tree Echo/);
    assert.equal(engine.hasLiveProvider(), false);
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

    assert.ok(out.memoriesUsed > 0);
    const messages = calls[0].body.messages as ChatTurn[];
    assert.equal(messages[0].role, "system");
    assert.match(messages[0].content, /Deep Tree Echo/);
    assert.match(messages[0].content, /Current Cognitive State/);
    assert.match(messages[0].content, /Relevant memories/);
    assert.match(messages[0].content, /ring/);
    assert.deepEqual(messages.at(-1), {
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
    assert.equal(await memory.count(), 4);
    const a = await memory.recall("a", "alpha topic", 5);
    assert.ok(a.every(r => r.sessionId === "a"));
    assert.equal((await memory.recall("b", "alpha topic", 5)).length, 0);
    assert.equal(engine.getCognitiveState("a").turnCount, 1);
    assert.equal(engine.getCognitiveState("b").turnCount, 1);
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
    assert.equal(messages.length, 6);
    assert.equal(messages.filter(m => m.role === "system").length, 1);
    assert.equal(messages[1].content, "t6");
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
    assert.equal(out.content, "still fine");
    assert.equal(out.memoriesUsed, 0);
    assert.equal((await engine.getStatus()).memory.count, 0);
  });
});

// ----------------------------------------------------------------------------
// fallback + status
// ----------------------------------------------------------------------------

describe("fallbackResponse", () => {
  it("mentions recalled memories and how to configure a provider", () => {
    const s = createInitialCognitiveState(T0);
    const text = fallbackResponse("explain reservoirs", s, 2);
    assert.match(text, /2 related memories/);
    assert.match(text, /OLLAMA_BASE_URL|OPENAI_API_KEY|ANTHROPIC_API_KEY/);
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
    assert.deepEqual(status.memory, {
      backend: "none",
      enabled: false,
      count: 0,
    });
    assert.equal(status.training, null);
    assert.deepEqual(
      status.providers.map(p => p.kind),
      ["ollama"]
    );
  });

  it("selects the Supabase backend only when both env vars are present", () => {
    assert.equal(
      new DTEChatEngine({ env: {}, fetchImpl: fakeFetch({}) }).memory.backend,
      "none"
    );
    assert.equal(
      new DTEChatEngine({
        env: {
          SUPABASE_URL: "https://x.supabase.co",
          SUPABASE_ANON_KEY: "anon",
        },
        fetchImpl: fakeFetch({}),
      }).memory.backend,
      "supabase"
    );
  });

  it("parses the real training_summary.json shape from the repo", () => {
    const summary = readTrainingSummary(process.cwd());
    // The repo ships this artifact; if it is ever removed the status page degrades gracefully.
    if (summary) {
      assert.equal(typeof summary.bestValLoss, "number");
      assert.ok((summary.totalCheckpoints ?? 0) > 0);
    } else {
      assert.equal(summary, null);
    }
  });
});
