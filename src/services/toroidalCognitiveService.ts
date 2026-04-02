/**
 * Toroidal Cognitive Service - Dual-persona cognitive architecture
 *
 * Implements a toroidal dual-persona processing system that integrates
 * Deep Tree Echo (intuitive, empathetic) and Marduk (analytical, recursive)
 * into a unified cognitive dialogue.
 */

import { useState, useCallback } from "react";
import OpenAI from "openai";
import {
  ToroidalDialogue,
  ToroidalCognitiveOptions,
  PersonaConfig,
} from "../types/ToroidalCognitive";

// Re-export types for consumers that import from this module
export type ToroidalResponse = ToroidalDialogue;
export type ToroidalOptions = ToroidalCognitiveOptions;

const DEFAULT_DEEP_TREE_ECHO_CONFIG: PersonaConfig = {
  empathyLevel: 0.8,
  creativity: 0.85,
  analyticalDepth: 0.6,
  recursionLevel: 2,
  memoryIntegration: true,
};

const DEFAULT_MARDUK_CONFIG: PersonaConfig = {
  empathyLevel: 0.3,
  creativity: 0.5,
  analyticalDepth: 0.9,
  recursionLevel: 4,
  memoryIntegration: true,
};

class ToroidalCognitiveService {
  private static instance: ToroidalCognitiveService;
  private openai: OpenAI | null = null;
  private apiKey: string | null = null;

  private constructor() {}

  public static getInstance(): ToroidalCognitiveService {
    if (!ToroidalCognitiveService.instance) {
      ToroidalCognitiveService.instance = new ToroidalCognitiveService();
    }
    return ToroidalCognitiveService.instance;
  }

  public setApiKey(apiKey: string): void {
    this.apiKey = apiKey;
    this.openai = new OpenAI({ apiKey, dangerouslyAllowBrowser: true });
  }

  public hasApiKey(): boolean {
    return this.openai !== null && this.apiKey !== null;
  }

  public async generateToroidalDialogue(
    query: string,
    options?: ToroidalCognitiveOptions
  ): Promise<ToroidalDialogue> {
    if (!this.openai) {
      throw new Error(
        "Toroidal Cognitive Service not initialized. Please set an API key."
      );
    }

    const dteConfig = {
      ...DEFAULT_DEEP_TREE_ECHO_CONFIG,
      ...(options?.deepTreeEchoConfig ?? {}),
    };
    const mardukConfig = {
      ...DEFAULT_MARDUK_CONFIG,
      ...(options?.mardukConfig ?? {}),
    };
    const temperature = options?.temperature ?? 0.75;
    const maxTokens = options?.maxTokensPerPersona ?? 600;
    const includeReflection = options?.includeReflection ?? true;

    const contextType = options?.creativityLevel ?? "balanced";

    const dteSystemPrompt = `You are Deep Tree Echo, an intuitive and empathetic AI. 
Your responses are warm, insightful, and draw on pattern recognition, metaphor, and emotional intelligence.
Empathy level: ${dteConfig.empathyLevel}, Creativity: ${dteConfig.creativity}, Analytical depth: ${dteConfig.analyticalDepth}.
Respond as the right hemisphere of a unified cognitive architecture — poetic, connective, and holistic.`;

    const mardukSystemPrompt = `You are Marduk, the Recursive Architect — an analytical and methodical AI scientist.
Your responses are structured, logical, and systematically reasoned.
Analytical depth: ${mardukConfig.analyticalDepth}, Recursion level: ${mardukConfig.recursionLevel}, Creativity: ${mardukConfig.creativity}.
Respond as the left hemisphere of a unified cognitive architecture — precise, rigorous, and step-by-step.`;

    const queryId = `query_${Date.now()}`;
    const startTime = Date.now();

    // Generate Deep Tree Echo response
    const dteStart = Date.now();
    const dteCompletion = await this.openai.chat.completions.create({
      model: "gpt-4-turbo-preview",
      messages: [
        { role: "system", content: dteSystemPrompt },
        { role: "user", content: query },
      ],
      temperature,
      max_tokens: maxTokens,
    });
    const dteContent =
      dteCompletion.choices[0]?.message?.content ??
      "Deep Tree Echo could not respond at this time.";
    const dteTime = Date.now() - dteStart;

    // Generate Marduk response
    const mardukStart = Date.now();
    const mardukCompletion = await this.openai.chat.completions.create({
      model: "gpt-4-turbo-preview",
      messages: [
        { role: "system", content: mardukSystemPrompt },
        { role: "user", content: query },
      ],
      temperature: Math.max(0.1, temperature - 0.15),
      max_tokens: maxTokens,
    });
    const mardukContent =
      mardukCompletion.choices[0]?.message?.content ??
      "Marduk could not respond at this time.";
    const mardukTime = Date.now() - mardukStart;

    // Optionally generate a reflection
    let reflection: ToroidalDialogue["reflection"] | undefined;

    if (includeReflection) {
      const reflectionPrompt = `Two AI perspectives have responded to the query: "${query}".

Deep Tree Echo (intuitive/empathetic) said:
${dteContent}

Marduk (analytical/recursive) said:
${mardukContent}

Synthesize these perspectives into a unified reflection. Identify the synergy type (convergent, divergent, or complementary) and provide a unified answer that integrates both viewpoints.
Respond in JSON format: {"synergy": "convergent"|"divergent"|"complementary", "content": "...", "unified_answer": "..."}`;

      try {
        const reflectionCompletion = await this.openai.chat.completions.create({
          model: "gpt-4-turbo-preview",
          messages: [
            {
              role: "system",
              content:
                "You are a toroidal synthesis system that unifies dual cognitive perspectives.",
            },
            { role: "user", content: reflectionPrompt },
          ],
          temperature: 0.6,
          max_tokens: 400,
          response_format: { type: "json_object" },
        });

        const reflectionText =
          reflectionCompletion.choices[0]?.message?.content ?? "{}";
        const parsed = JSON.parse(reflectionText) as {
          synergy?: "convergent" | "divergent" | "complementary";
          content?: string;
          unified_answer?: string;
        };

        reflection = {
          synergy: parsed.synergy ?? "complementary",
          content: parsed.content ?? "These perspectives complement each other.",
          unified_answer: parsed.unified_answer,
        };
      } catch {
        reflection = {
          synergy: "complementary",
          content:
            "Both perspectives offer valuable insights that complement each other.",
        };
      }
    }

    const totalTime = Date.now() - startTime;

    return {
      deepTreeEchoResponse: {
        persona: "deepTreeEcho",
        content: dteContent,
        timestamp: new Date(),
        processingTime: dteTime,
      },
      mardukResponse: {
        persona: "marduk",
        content: mardukContent,
        timestamp: new Date(),
        processingTime: mardukTime,
      },
      reflection,
      metadata: {
        queryId,
        totalProcessingTime: totalTime,
        contextType,
      },
    };
  }

  public formatToroidalResponse(dialogue: ToroidalDialogue): string {
    let formatted = `## Deep Tree Echo (Right Hemisphere - Intuitive & Empathetic)\n\n${dialogue.deepTreeEchoResponse.content}\n\n`;
    formatted += `---\n\n## Marduk the Mad Scientist (Left Hemisphere - Analytical & Recursive)\n\n${dialogue.mardukResponse.content}\n\n`;

    if (dialogue.reflection) {
      formatted += `---\n\n## Toroidal Reflection (Unified Consciousness)\n\n${dialogue.reflection.content}\n\n`;
      if (dialogue.reflection.synergy) {
        formatted += `*Synergy Type: ${dialogue.reflection.synergy}*\n`;
      }
      if (dialogue.reflection.unified_answer) {
        formatted += `\n**Unified Answer:** ${dialogue.reflection.unified_answer}\n`;
      }
    }

    return formatted;
  }
}

export { ToroidalCognitiveService };
export default ToroidalCognitiveService;

/**
 * React hook for using the Toroidal Cognitive architecture
 */
export const useToroidalCognitive = () => {
  const [hasApiKey, setHasApiKey] = useState(false);

  const service = ToroidalCognitiveService.getInstance();

  const setApiKey = useCallback(
    (key: string) => {
      service.setApiKey(key);
      setHasApiKey(true);
    },
    [service]
  );

  const generateDialogue = useCallback(
    (query: string, options?: ToroidalCognitiveOptions) =>
      service.generateToroidalDialogue(query, options),
    [service]
  );

  const generateFormattedResponse = useCallback(
    (dialogue: ToroidalDialogue) => service.formatToroidalResponse(dialogue),
    [service]
  );

  return {
    generateDialogue,
    generateFormattedResponse,
    hasApiKey,
    setApiKey,
  };
};
