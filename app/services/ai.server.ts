import { getDTEChatEngine, type ChatTurn } from "./dte-chat.server";

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ChatOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  systemPrompt?: string;
  /** Session used for cognitive state and memory; defaults to a shared one. */
  sessionId?: string;
}

/**
 * Thin façade over the Deep Tree Echo chat engine.
 *
 * Everything the character says comes from a model via the engine's provider
 * chain (Ollama → OpenAI → Anthropic). When no provider answers the engine
 * throws `NoModelAvailableError`; this service lets that propagate so callers
 * show it as an operator-facing status rather than inventing a reply.
 */
export class AIService {
  private readonly engine = getDTEChatEngine();

  /** True when at least one provider is configured. */
  public isInitialized(): boolean {
    return this.engine.hasConfiguredProvider();
  }

  public async generateResponse(
    prompt: string,
    options: ChatOptions = {}
  ): Promise<string> {
    const extra = options.systemPrompt ? ["", options.systemPrompt] : [];
    const { text } = await this.engine.generate(
      options.sessionId ?? "ai-service",
      prompt,
      extra
    );
    return text;
  }

  public async generateResponseWithHistory(
    messages: ChatMessage[],
    options: ChatOptions = {}
  ): Promise<string> {
    const last = messages[messages.length - 1];
    if (!last || last.role !== "user") {
      throw new Error(
        "generateResponseWithHistory: last message must be from the user"
      );
    }
    const history: ChatTurn[] = messages
      .slice(0, -1)
      .filter(m => m.role !== "system")
      .map(m => ({ role: m.role, content: m.content }));
    const { content } = await this.engine.respond({
      sessionId: options.sessionId ?? "ai-service",
      history,
      content: last.content,
    });
    return content;
  }
}

let aiService: AIService | null = null;

export const getAIService = () => {
  if (!aiService) aiService = new AIService();
  return aiService;
};
