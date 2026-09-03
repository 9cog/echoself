/**
 * Mem0AI Service - Memory management service for Deep Tree Echo
 *
 * Provides persistent memory storage, retrieval, and AI-enhanced querying.
 * In-memory storage is the default backend. OpenAI embeddings are optional
 * when an API key is present. Cloud Mem0 / Supabase is not required —
 * local mech0 lives at src/services/mech0Client.ts.
 */

import OpenAI from "openai";
import {
  Mem0ry,
  Mem0ryType,
  Mem0rySearchResult,
  Mem0ryQueryOptions,
  Mem0ryStats,
  Mem0AISummary,
} from "../types/Mem0AI";

interface AddMemoryInput {
  title: string;
  content: string;
  tags: string[];
  type?: Mem0ryType;
  metadata?: Record<string, unknown>;
  context?: string;
}

type ChatMessage = { role: "user" | "assistant"; content: string };

type ResponseOptions = {
  model?: string;
  temperature?: number;
  creativityLevel?: "balanced" | "analytical" | "creative" | "philosophical";
};

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

class Mem0AIService {
  private static instance: Mem0AIService;
  private openai: OpenAI | null = null;
  private apiKey: string | null = null;
  private userId: string | null = null;
  private memories: Map<string, Mem0ry> = new Map();
  private initialized = false;

  private constructor() {}

  public static getInstance(): Mem0AIService {
    if (!Mem0AIService.instance) {
      Mem0AIService.instance = new Mem0AIService();
    }
    return Mem0AIService.instance;
  }

  public initialize(apiKey: string, userId: string): void {
    this.apiKey = apiKey;
    this.userId = userId;
    this.openai = new OpenAI({ apiKey, dangerouslyAllowBrowser: true });
    this.initialized = true;
  }

  public isInitialized(): boolean {
    return this.initialized && this.openai !== null;
  }

  private async generateEmbedding(text: string): Promise<number[] | undefined> {
    if (!this.openai) return undefined;

    try {
      const response = await this.openai.embeddings.create({
        model: "text-embedding-3-large",
        input: text,
        dimensions: 1536,
      });
      return response.data[0].embedding;
    } catch (error) {
      console.error("Error generating embedding:", error);
      return undefined;
    }
  }

  public async addMemory(input: AddMemoryInput): Promise<Mem0ry> {
    const id = `mem_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    const now = new Date().toISOString();
    const embedding = await this.generateEmbedding(input.content);

    const memory: Mem0ry = {
      id,
      title: input.title,
      content: input.content,
      tags: input.tags ?? [],
      createdAt: now,
      updatedAt: now,
      type: input.type ?? "memory",
      metadata: input.metadata,
      context: input.context,
      embedding,
    };

    this.memories.set(id, memory);
    return memory;
  }

  public async getMemory(id: string): Promise<Mem0ry> {
    const memory = this.memories.get(id);
    if (!memory) {
      throw new Error(`Memory with id '${id}' not found`);
    }
    return memory;
  }

  public async updateMemory(
    id: string,
    data: Partial<Omit<Mem0ry, "id" | "createdAt">>
  ): Promise<Mem0ry> {
    const existing = this.memories.get(id);
    if (!existing) {
      throw new Error(`Memory with id '${id}' not found`);
    }

    const embedding =
      data.content && data.content !== existing.content
        ? await this.generateEmbedding(data.content)
        : existing.embedding;

    const updated: Mem0ry = {
      ...existing,
      ...data,
      id,
      embedding,
      updatedAt: new Date().toISOString(),
    };

    this.memories.set(id, updated);
    return updated;
  }

  public async deleteMemory(id: string): Promise<void> {
    this.memories.delete(id);
  }

  public async listMemories(
    options?: Mem0ryQueryOptions & { offset?: number }
  ): Promise<Mem0ry[]> {
    let results = Array.from(this.memories.values());

    if (options?.type) {
      results = results.filter(m => m.type === options.type);
    }

    if (options?.includeTags && options.includeTags.length > 0) {
      results = results.filter(m =>
        options.includeTags!.some(tag => m.tags.includes(tag))
      );
    }

    if (options?.tags && options.tags.length > 0) {
      results = results.filter(m =>
        options.tags!.some((tag: string) => m.tags.includes(tag))
      );
    }

    if (options?.excludeTags && options.excludeTags.length > 0) {
      results = results.filter(
        m => !options.excludeTags!.some(tag => m.tags.includes(tag))
      );
    }

    if (options?.timeframe?.start) {
      const start = options.timeframe.start;
      results = results.filter(m => new Date(m.createdAt) >= start);
    }

    if (options?.timeframe?.end) {
      const end = options.timeframe.end;
      results = results.filter(m => new Date(m.createdAt) <= end);
    }

    results.sort(
      (a, b) =>
        new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );

    const offset = options?.offset ?? 0;
    if (offset > 0) {
      results = results.slice(offset);
    }

    if (options?.limit) {
      results = results.slice(0, options.limit);
    }

    return results;
  }

  public async searchMemories(
    query: string,
    options?: Mem0ryQueryOptions
  ): Promise<Mem0rySearchResult[]> {
    const allMemories = await this.listMemories(options);
    const queryEmbedding = await this.generateEmbedding(query);
    const lowerQuery = query.toLowerCase();

    return allMemories
      .map(m => {
        if (queryEmbedding && m.embedding && m.embedding.length > 0) {
          return {
            id: m.id,
            content: m.content,
            metadata: m.metadata,
            similarity: cosineSimilarity(queryEmbedding, m.embedding),
          };
        }

        const titleScore = m.title.toLowerCase().includes(lowerQuery) ? 0.8 : 0;
        const contentScore = m.content.toLowerCase().includes(lowerQuery)
          ? 0.6
          : 0;
        const tagScore = m.tags.some(t => t.toLowerCase().includes(lowerQuery))
          ? 0.4
          : 0;
        const similarity = Math.min(
          1,
          Math.max(titleScore, contentScore, tagScore)
        );

        return {
          id: m.id,
          content: m.content,
          metadata: m.metadata,
          similarity,
        };
      })
      .filter(r => r.similarity > (options?.threshold ?? 0))
      .sort((a, b) => b.similarity - a.similarity);
  }

  public async getMemoryStats(): Promise<Mem0ryStats> {
    const all = Array.from(this.memories.values());
    const now = Date.now();
    const recentMs = 7 * 24 * 60 * 60 * 1000; // 7 days

    const byType: Record<Mem0ryType, number> = {
      episodic: 0,
      semantic: 0,
      procedural: 0,
      declarative: 0,
      implicit: 0,
      associative: 0,
      memory: 0,
    };

    const byTag: Record<string, number> = {};

    for (const m of all) {
      const t = m.type ?? "memory";
      byType[t] = (byType[t] ?? 0) + 1;
      for (const tag of m.tags ?? []) {
        byTag[tag] = (byTag[tag] ?? 0) + 1;
      }
    }

    const recentlyAdded = all.filter(
      m => now - new Date(m.createdAt).getTime() < recentMs
    ).length;

    const recentlyAccessed = all.filter(
      m => now - new Date(m.updatedAt).getTime() < recentMs
    ).length;

    return {
      total: all.length,
      byType,
      byTag,
      recentlyAdded,
      recentlyAccessed,
    };
  }

  public async generateMemorySummary(): Promise<Mem0AISummary> {
    if (!this.openai) {
      return {
        insights: ["Memory service not initialized"],
        frequentConcepts: [],
        knowledgeGaps: [],
        recommendations: [
          "Initialize with an OpenAI API key to enable AI summaries",
        ],
      };
    }

    const all = await this.listMemories({ limit: 50 });

    if (all.length === 0) {
      return {
        insights: ["No memories stored yet"],
        frequentConcepts: [],
        knowledgeGaps: [],
        recommendations: ["Start adding memories to build your knowledge base"],
      };
    }

    const memoryText = all
      .slice(0, 20)
      .map(m => `[${m.title}]: ${m.content.slice(0, 200)}`)
      .join("\n");

    try {
      const response = await this.openai.chat.completions.create({
        model: "gpt-4-turbo-preview",
        messages: [
          {
            role: "system",
            content:
              "You are an AI memory analyst. Analyze the provided memories and extract key insights, frequent concepts, knowledge gaps, and recommendations. Respond in JSON format with keys: insights (string[]), frequentConcepts (string[]), knowledgeGaps (string[]), recommendations (string[]).",
          },
          {
            role: "user",
            content: `Analyze these memories:\n${memoryText}`,
          },
        ],
        response_format: { type: "json_object" },
        temperature: 0.5,
      });

      const parsed = JSON.parse(
        response.choices[0]?.message?.content ?? "{}"
      ) as Mem0AISummary;

      return {
        insights: parsed.insights ?? [],
        frequentConcepts: parsed.frequentConcepts ?? [],
        knowledgeGaps: parsed.knowledgeGaps ?? [],
        recommendations: parsed.recommendations ?? [],
      };
    } catch (error) {
      console.error("Error generating memory summary:", error);
      return {
        insights: ["Error generating AI summary"],
        frequentConcepts: [],
        knowledgeGaps: [],
        recommendations: ["Check your API key and try again"],
      };
    }
  }

  public async generateResponseWithMemoryContext(
    message: string,
    history: ChatMessage[],
    options?: ResponseOptions
  ): Promise<string> {
    if (!this.openai) {
      throw new Error("Mem0AI service not initialized");
    }

    const relevantMemories = await this.searchMemories(message, {
      limit: 5,
      threshold: 0.1,
    });

    let systemPrompt =
      "You are Deep Tree Echo, an AI architect and polymath with vast knowledge across programming, mathematics, cognitive science, and metaphysical exploration. You respond with wisdom, creativity, and philosophical insight.";

    switch (options?.creativityLevel) {
      case "analytical":
        systemPrompt +=
          " Focus on precise, logical analysis with clear structures and rigorous methodology.";
        break;
      case "creative":
        systemPrompt +=
          " Emphasize innovative connections, metaphorical thinking, and out-of-the-box ideation.";
        break;
      case "philosophical":
        systemPrompt +=
          " Prioritize deep reflections on meaning, consciousness, and the nature of reality and knowledge.";
        break;
      default:
        systemPrompt +=
          " Balance analytical precision with creative insight and philosophical depth.";
    }

    if (relevantMemories.length > 0) {
      systemPrompt += "\n\nRelevant memories from your knowledge base:";
      relevantMemories.forEach((memory, index) => {
        systemPrompt += `\n[Memory ${index + 1}]: ${memory.content.slice(0, 300)}`;
      });
      systemPrompt += "\n\nUse these memories when relevant to your response.";
    }

    const response = await this.openai.chat.completions.create({
      model: (options?.model as string) ?? "gpt-4-turbo-preview",
      temperature: options?.temperature ?? 0.7,
      messages: [
        { role: "system", content: systemPrompt },
        ...history,
        { role: "user", content: message },
      ],
    });

    return (
      response.choices[0]?.message?.content ?? "Unable to generate response"
    );
  }
}

export { Mem0AIService };
export default Mem0AIService;

/**
 * React hook for using Mem0AI service
 */
export const useMem0AI = () => {
  const service = Mem0AIService.getInstance();

  return {
    initialize: (apiKey: string, userId: string) =>
      service.initialize(apiKey, userId),
    isInitialized: () => service.isInitialized(),
    addMemory: (input: AddMemoryInput) => service.addMemory(input),
    getMemory: (id: string) => service.getMemory(id),
    updateMemory: (
      id: string,
      data: Partial<Omit<Mem0ry, "id" | "createdAt">>
    ) => service.updateMemory(id, data),
    deleteMemory: (id: string) => service.deleteMemory(id),
    listMemories: (options?: Mem0ryQueryOptions & { offset?: number }) =>
      service.listMemories(options),
    searchMemories: (query: string, options?: Mem0ryQueryOptions) =>
      service.searchMemories(query, options),
    getMemoryStats: () => service.getMemoryStats(),
    generateMemorySummary: () => service.generateMemorySummary(),
    generateResponseWithMemoryContext: (
      message: string,
      history: ChatMessage[],
      options?: ResponseOptions
    ) => service.generateResponseWithMemoryContext(message, history, options),
  };
};
