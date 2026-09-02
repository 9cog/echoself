import {
  json,
  type ActionFunctionArgs,
  type LoaderFunctionArgs,
} from "@remix-run/node";
import {
  useActionData,
  useLoaderData,
  useNavigation,
  Form,
  Link,
} from "@remix-run/react";
import { useEffect, useState } from "react";
import ChatInterface, { type Message } from "~/components/ChatInterface";
import {
  getDTEChatEngine,
  type CognitiveState,
  type ChatTurn,
} from "~/services/dte-chat.server";

const SESSION_COOKIE = "dte_session";

/** Anonymous, cookie-backed session id so memory and cognitive state persist across requests. */
function getSessionId(request: Request): { id: string; setCookie?: string } {
  const cookie = request.headers.get("Cookie") ?? "";
  const match = cookie.match(
    new RegExp(`(?:^|;\\s*)${SESSION_COOKIE}=([^;]+)`)
  );
  if (match) return { id: decodeURIComponent(match[1]) };
  const id = `anon_${crypto.randomUUID()}`;
  return {
    id,
    setCookie: `${SESSION_COOKIE}=${encodeURIComponent(id)}; Path=/; Max-Age=${60 * 60 * 24 * 30}; SameSite=Lax; HttpOnly`,
  };
}

export async function loader({ request }: LoaderFunctionArgs) {
  const engine = getDTEChatEngine();
  const session = getSessionId(request);
  const status = await engine.getStatus();

  return json(
    {
      providerConfigured: engine.hasConfiguredProvider(),
      providerLive: status.hasLiveProvider,
      memoryBackend: status.memory.backend,
      cognitiveState: engine.getCognitiveState(session.id),
      initialMessage: {
        id: "welcome",
        role: "assistant" as const,
        content: "Welcome to Deep Tree Echo. How can I assist you today?",
        timestamp: new Date().toISOString(),
      },
    },
    session.setCookie
      ? { headers: { "Set-Cookie": session.setCookie } }
      : undefined
  );
}

export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();
  const content = (formData.get("content") as string | null)?.trim();
  const history = JSON.parse(
    (formData.get("history") as string) || "[]"
  ) as Message[];

  if (!content) {
    return json({ error: "Message content is required" });
  }

  const engine = getDTEChatEngine();
  const session = getSessionId(request);

  const userMessage: Message = {
    id: `msg_${Date.now()}`,
    role: "user",
    content,
    timestamp: new Date().toISOString(),
  };

  const turns: ChatTurn[] = history.map(m => ({
    role: m.role,
    content: m.content,
  }));

  try {
    const result = await engine.respond({
      sessionId: session.id,
      history: turns,
      content,
    });

    const assistantMessage: Message = {
      id: `msg_${Date.now() + 1}`,
      role: "assistant",
      content: result.content,
      timestamp: new Date().toISOString(),
    };

    return json(
      {
        userMessage,
        assistantMessage,
        provider: result.provider,
        memoriesUsed: result.memoriesUsed,
        latencyMs: result.latencyMs,
        cognitiveState: result.cognitiveState,
        success: true,
      },
      session.setCookie
        ? { headers: { "Set-Cookie": session.setCookie } }
        : undefined
    );
  } catch (error) {
    console.error("[chat] DTE engine error:", error);
    return json({
      userMessage,
      error: "Failed to generate response. Please try again.",
    });
  }
}

function CognitiveStrip({
  state,
  provider,
  memoriesUsed,
  latencyMs,
  memoryBackend,
}: {
  state: CognitiveState;
  provider?: string;
  memoriesUsed?: number;
  latencyMs?: number;
  memoryBackend: string;
}) {
  const pct = (n: number) => `${Math.round(n * 100)}%`;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-1.5 text-xs border-b border-border bg-card/50 text-card-foreground/80">
      <span title="Emotional valence">
        valence {pct(state.emotionalValence)}
      </span>
      <span title="Arousal">arousal {pct(state.arousalLevel)}</span>
      <span title="Wisdom">wisdom {pct(state.wisdomLevel)}</span>
      <span title="Introspection depth">
        introspection {state.introspectionDepth.toFixed(1)}
      </span>
      <span title="Turns this session">turns {state.turnCount}</span>
      <span className="opacity-60">·</span>
      <span title="Memory backend">memory: {memoryBackend}</span>
      {provider && (
        <span
          title="Provider that answered the last message"
          className={
            provider === "fallback" ? "text-yellow-400" : "text-primary"
          }
        >
          via {provider}
          {typeof memoriesUsed === "number" && memoriesUsed > 0
            ? ` · ${memoriesUsed} memories`
            : ""}
          {typeof latencyMs === "number" ? ` · ${latencyMs}ms` : ""}
        </span>
      )}
      <Link to="/status" className="ml-auto text-primary hover:underline">
        status
      </Link>
    </div>
  );
}

export default function ChatPage() {
  const loaderData = useLoaderData<typeof loader>();
  const actionData = useActionData<typeof action>();
  const navigation = useNavigation();
  const [messages, setMessages] = useState<Message[]>([
    loaderData.initialMessage,
  ]);
  const [showInfo, setShowInfo] = useState(false);

  useEffect(() => {
    if (actionData && "userMessage" in actionData) {
      setMessages(prev => [...prev, actionData.userMessage]);
      if ("assistantMessage" in actionData) {
        setMessages(prev => [...prev, actionData.assistantMessage]);
      }
    }
  }, [actionData]);

  const isProcessing = navigation.state === "submitting";
  const latest =
    actionData && "cognitiveState" in actionData ? actionData : null;
  const state = latest?.cognitiveState ?? loaderData.cognitiveState;

  return (
    <div className="h-screen flex flex-col">
      <CognitiveStrip
        state={state}
        provider={latest?.provider}
        memoriesUsed={latest?.memoriesUsed}
        latencyMs={latest?.latencyMs}
        memoryBackend={loaderData.memoryBackend}
      />
      <Form method="post" className="flex-1 min-h-0">
        <input type="hidden" name="history" value={JSON.stringify(messages)} />
        <input type="hidden" name="content" id="message-content" />

        <ChatInterface
          messages={messages}
          onSendMessage={content => {
            const input = document.getElementById(
              "message-content"
            ) as HTMLInputElement | null;
            if (input) {
              input.value = content;
              input.form?.requestSubmit();
            }
          }}
          isProcessing={isProcessing}
          apiKeyConfigured={loaderData.providerConfigured}
          onConfigureApiKey={() => setShowInfo(true)}
        />
      </Form>

      {showInfo && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-card rounded-lg shadow-xl max-w-md w-full p-6">
            <h2 className="text-xl font-semibold mb-4">
              Connect a language model
            </h2>
            <p className="mb-4 text-sm opacity-80">
              Deep Tree Echo tries providers in order and falls back to its
              local persona kernel when none respond. Configure any of these on
              the server:
            </p>
            <ul className="text-sm space-y-1 mb-4 font-mono">
              <li>OLLAMA_BASE_URL / OLLAMA_MODEL</li>
              <li>OPENAI_API_KEY / OPENAI_MODEL</li>
              <li>ANTHROPIC_API_KEY / ANTHROPIC_MODEL</li>
              <li>DTE_LLM_PROVIDERS (ordering)</li>
            </ul>
            <p className="text-sm opacity-80 mb-4">
              Memory backend:{" "}
              <span className="font-mono">{loaderData.memoryBackend}</span>. Set
              SUPABASE_URL and SUPABASE_ANON_KEY to persist conversations.
            </p>
            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => setShowInfo(false)}
                className="px-4 py-2 bg-primary text-white rounded-md"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
