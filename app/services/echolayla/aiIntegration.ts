/**
 * EchoLayla AI Integration
 *
 * Integration with various AI services for actual inference.
 * This module provides adapters for different AI providers.
 */

import type { InferenceConfig } from "./types";
import process from "node:process";

/**
 * AI Provider types
 */
export type AIProvider = "openai" | "huggingface" | "local";

/**
 * AI Response structure
 */
export interface AIResponse {
  content: string;
  finishReason: "stop" | "length" | "error";
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

/**
 * Streaming callback
 */
export type StreamCallback = (chunk: string) => void;

/**
 * Base AI Adapter interface
 */
export interface AIAdapter {
  generate(
    messages: Array<{ role: string; content: string }>,
    config: InferenceConfig
  ): Promise<AIResponse>;

  generateStream(
    messages: Array<{ role: string; content: string }>,
    config: InferenceConfig,
    onChunk: StreamCallback
  ): Promise<AIResponse>;
}

/**
 * OpenAI Adapter
 */
export class OpenAIAdapter implements AIAdapter {
  private apiKey: string;
  private baseURL: string;

  constructor(apiKey?: string, baseURL?: string) {
    this.apiKey = apiKey || process.env.OPENAI_API_KEY || "";
    this.baseURL = baseURL || "https://api.openai.com/v1";
  }

  async generate(
    messages: Array<{ role: string; content: string }>,
    config: InferenceConfig
  ): Promise<AIResponse> {
    if (!this.apiKey) {
      throw new Error("OpenAI API key not configured");
    }

    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: config.model || "gpt-3.5-turbo",
        messages,
        temperature: config.temperature,
        max_tokens: config.maxTokens,
        top_p: config.topP,
        stream: false,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(
        `OpenAI API error: ${error.error?.message || response.statusText}`
      );
    }

    const data = await response.json();
    const choice = data.choices[0];

    return {
      content: choice.message.content,
      finishReason: choice.finish_reason,
      usage: data.usage,
    };
  }

  async generateStream(
    messages: Array<{ role: string; content: string }>,
    config: InferenceConfig,
    onChunk: StreamCallback
  ): Promise<AIResponse> {
    if (!this.apiKey) {
      throw new Error("OpenAI API key not configured");
    }

    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: config.model || "gpt-3.5-turbo",
        messages,
        temperature: config.temperature,
        max_tokens: config.maxTokens,
        top_p: config.topP,
        stream: true,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(
        `OpenAI API error: ${error.error?.message || response.statusText}`
      );
    }

    let fullContent = "";
    let finishReason: "stop" | "length" | "error" = "stop";

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error("Failed to get response reader");
    }

    try {
      let done = false;
      while (!done) {
        const result = await reader.read();
        done = result.done;
        if (done) break;

        const chunk = decoder.decode(result.value);
        const lines = chunk.split("\n").filter(line => line.trim() !== "");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") continue;

            try {
              const parsed = JSON.parse(data);
              const content = parsed.choices[0]?.delta?.content || "";

              if (content) {
                fullContent += content;
                onChunk(content);
              }

              if (parsed.choices[0]?.finish_reason) {
                finishReason = parsed.choices[0].finish_reason;
              }
            } catch (e) {
              console.error("Error parsing SSE data:", e);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    return {
      content: fullContent,
      finishReason,
    };
  }
}

/**
 * HuggingFace Adapter
 */
export class HuggingFaceAdapter implements AIAdapter {
  private apiKey: string;
  private model: string;

  constructor(apiKey?: string, model?: string) {
    this.apiKey = apiKey || process.env.HUGGINGFACE_API_KEY || "";
    this.model = model || "mistralai/Mistral-7B-Instruct-v0.2";
  }

  async generate(
    messages: Array<{ role: string; content: string }>,
    config: InferenceConfig
  ): Promise<AIResponse> {
    if (!this.apiKey) {
      throw new Error("HuggingFace API key not configured");
    }

    // Convert messages to HuggingFace format
    const prompt = this.formatMessages(messages);

    const response = await fetch(
      `https://api-inference.huggingface.co/models/${this.model}`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          inputs: prompt,
          parameters: {
            temperature: config.temperature,
            max_new_tokens: config.maxTokens,
            top_p: config.topP,
            top_k: config.topK,
            return_full_text: false,
          },
        }),
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(
        `HuggingFace API error: ${error.error || response.statusText}`
      );
    }

    const data = await response.json();
    const generated = data[0]?.generated_text || "";

    return {
      content: generated,
      finishReason: "stop",
    };
  }

  async generateStream(
    messages: Array<{ role: string; content: string }>,
    config: InferenceConfig,
    onChunk: StreamCallback
  ): Promise<AIResponse> {
    // HuggingFace Inference API doesn't support streaming by default
    // Fall back to non-streaming
    const result = await this.generate(messages, config);
    onChunk(result.content);
    return result;
  }

  private formatMessages(
    messages: Array<{ role: string; content: string }>
  ): string {
    // Simple formatting for instruction-tuned models
    return messages
      .map(msg => {
        if (msg.role === "system") {
          return `<<SYS>>\n${msg.content}\n<</SYS>>`;
        }
        if (msg.role === "user") {
          return `[INST] ${msg.content} [/INST]`;
        }
        return msg.content;
      })
      .join("\n\n");
  }
}

/**
 * Mock Adapter for development/testing
 */
export class MockAdapter implements AIAdapter {
  async generate(
    messages: Array<{ role: string; content: string }>,
    config: InferenceConfig
  ): Promise<AIResponse> {
    const lastMessage = messages[messages.length - 1];
    const characterName =
      config.systemPrompt?.match(/You are (\w+),/)?.[1] || "AI";

    return {
      content: `[${characterName}] This is a mock response to: "${lastMessage.content}"`,
      finishReason: "stop",
      usage: {
        promptTokens: 50,
        completionTokens: 20,
        totalTokens: 70,
      },
    };
  }

  async generateStream(
    messages: Array<{ role: string; content: string }>,
    config: InferenceConfig,
    onChunk: StreamCallback
  ): Promise<AIResponse> {
    const response = await this.generate(messages, config);

    // Simulate streaming by chunking
    const words = response.content.split(" ");
    for (const word of words) {
      await new Promise(resolve => setTimeout(resolve, 50));
      onChunk(word + " ");
    }

    return response;
  }
}

/**
 * AI Service Factory
 */
export class AIServiceFactory {
  static createAdapter(
    provider: AIProvider,
    options?: Record<string, string>
  ): AIAdapter {
    switch (provider) {
      case "openai":
        return new OpenAIAdapter(options?.apiKey, options?.baseURL);
      case "huggingface":
        return new HuggingFaceAdapter(options?.apiKey, options?.model);
      case "local":
        // TODO: Implement local inference adapter (LLaMA.cpp, etc.)
        return new MockAdapter();
      default:
        return new MockAdapter();
    }
  }
}

/**
 * Get default AI adapter based on environment
 */
export function getDefaultAdapter(): AIAdapter {
  // Check for API keys in environment
  if (process.env.OPENAI_API_KEY) {
    return AIServiceFactory.createAdapter("openai");
  }

  if (process.env.HUGGINGFACE_API_KEY) {
    return AIServiceFactory.createAdapter("huggingface");
  }

  // Fall back to mock for development
  return AIServiceFactory.createAdapter("local");
}
