import { getDTEChatEngine, type ChatTurn } from "./dte-chat.server";
import { getMemoryService } from "./memory.server";

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ChatOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  systemPrompt?: string;
  useMemoryContext?: boolean;
  useESN?: boolean;
  sessionId?: string;
}

/**
 * Chat service with optional vector-memory context, delegating generation to
 * the Deep Tree Echo engine's provider chain. No simulated responses: if no
 * model answers, the engine's `NoModelAvailableError` propagates.
 */
export class AIChatService {
  private static instance: AIChatService;
  private readonly engine = getDTEChatEngine();
  private memoryService = getMemoryService();

  private constructor() {}

  public static getInstance(): AIChatService {
    if (!AIChatService.instance) AIChatService.instance = new AIChatService();
    return AIChatService.instance;
  }

  public isInitialized(): boolean {
    return this.engine.hasConfiguredProvider();
  }

  public async generateResponse(
    messages: ChatMessage[],
    options: ChatOptions = {}
  ): Promise<string> {
    const last = messages[messages.length - 1];
    if (!last || last.role !== "user") {
      throw new Error("generateResponse: last message must be from the user");
    }

    const extraSystem: string[] = [];
    if (options.systemPrompt) extraSystem.push("", options.systemPrompt);

    if (options.useMemoryContext) {
      const relevant = await this.memoryService
        .searchMemories(last.content, { limit: 3 })
        .catch(() => []);
      if (relevant.length > 0) {
        extraSystem.push(
          "",
          "## Relevant context from long-term memory",
          ...relevant.map(m => `- ${m.content}`)
        );
      }
    }
    if (options.useESN) {
      extraSystem.push(
        "",
        "Consider temporal patterns and resonances across the conversation."
      );
    }

    const history: ChatTurn[] = messages
      .slice(0, -1)
      .filter(m => m.role !== "system")
      .map(m => ({ role: m.role, content: m.content }));

    if (extraSystem.length === 0) {
      const { content } = await this.engine.respond({
        sessionId: options.sessionId ?? "ai-chat-service",
        history,
        content: last.content,
      });
      return content;
    }
    // With extra system context, go through generate() so the persona,
    // cognitive state and the extra context are all in one system block.
    const transcript = history.map(t => `${t.role}: ${t.content}`).join("\n");
    const { text } = await this.engine.generate(
      options.sessionId ?? "ai-chat-service",
      transcript ? `${transcript}\nuser: ${last.content}` : last.content,
      extraSystem
    );
    return text;
  }
}

let aiChatService: AIChatService | null = null;

export const getAIChatService = () => {
  if (!aiChatService) aiChatService = AIChatService.getInstance();
  return aiChatService;
};
