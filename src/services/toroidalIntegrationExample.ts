// Example integration showing how to use the Toroidal Cognitive Architecture
// This demonstrates various usage patterns and integration approaches

import {
  ToroidalCognitiveService,
  useToroidalCognitive as _useToroidalCognitive,
} from "./toroidalCognitiveService";
import {
  ToroidalCognitiveOptions,
  ToroidalDialogue,
} from "../types/ToroidalCognitive";

/**
 * Example integration class demonstrating various ways to use the toroidal architecture
 */
export class ToroidalIntegrationExample {
  /**
   * Basic dialogue generation with default settings
   */
  static async basicExample(query: string): Promise<string> {
    const service = ToroidalCognitiveService.getInstance();

    try {
      const dialogue = await service.generateToroidalDialogue(query);
      return service.formatToroidalResponse(dialogue);
    } catch (error) {
      console.error("Basic example failed:", error);
      return "Error: Unable to generate toroidal response";
    }
  }

  /**
   * Advanced dialogue with custom configuration
   */
  static async advancedExample(query: string): Promise<ToroidalDialogue> {
    const service = ToroidalCognitiveService.getInstance();

    const options: ToroidalCognitiveOptions = {
      creativityLevel: "philosophical",
      includeReflection: true,
      includeMemories: true,
      maxTokensPerPersona: 800,
      temperature: 0.8,
      deepTreeEchoConfig: {
        empathyLevel: 0.9,
        creativity: 0.8,
        analyticalDepth: 0.6,
        recursionLevel: 2,
        memoryIntegration: true,
      },
      mardukConfig: {
        analyticalDepth: 0.95,
        recursionLevel: 4,
        creativity: 0.5,
        empathyLevel: 0.3,
        memoryIntegration: true,
      },
    };

    try {
      return await service.generateToroidalDialogue(query, options);
    } catch (error) {
      console.error("Advanced example failed:", error);
      throw error;
    }
  }

  /**
   * Task-specific configurations for different types of queries
   */
  static getConfigForQueryType(
    queryType: "technical" | "creative" | "philosophical" | "analytical"
  ): ToroidalCognitiveOptions {
    const configs = {
      technical: {
        creativityLevel: "analytical" as const,
        temperature: 0.6,
        maxTokensPerPersona: 700,
        mardukConfig: {
          analyticalDepth: 0.95,
          recursionLevel: 5,
          creativity: 0.4,
          empathyLevel: 0.2,
          memoryIntegration: true,
        },
        deepTreeEchoConfig: {
          empathyLevel: 0.6,
          creativity: 0.5,
          analyticalDepth: 0.8,
          recursionLevel: 2,
          memoryIntegration: true,
        },
      },
      creative: {
        creativityLevel: "creative" as const,
        temperature: 0.9,
        maxTokensPerPersona: 600,
        deepTreeEchoConfig: {
          empathyLevel: 0.9,
          creativity: 0.95,
          analyticalDepth: 0.5,
          recursionLevel: 3,
          memoryIntegration: true,
        },
        mardukConfig: {
          analyticalDepth: 0.7,
          recursionLevel: 3,
          creativity: 0.8,
          empathyLevel: 0.4,
          memoryIntegration: true,
        },
      },
      philosophical: {
        creativityLevel: "philosophical" as const,
        temperature: 0.8,
        maxTokensPerPersona: 800,
        deepTreeEchoConfig: {
          empathyLevel: 0.9,
          creativity: 0.8,
          analyticalDepth: 0.7,
          recursionLevel: 4,
          memoryIntegration: true,
        },
        mardukConfig: {
          analyticalDepth: 0.9,
          recursionLevel: 5,
          creativity: 0.7,
          empathyLevel: 0.5,
          memoryIntegration: true,
        },
      },
      analytical: {
        creativityLevel: "analytical" as const,
        temperature: 0.5,
        maxTokensPerPersona: 750,
        mardukConfig: {
          analyticalDepth: 0.98,
          recursionLevel: 5,
          creativity: 0.3,
          empathyLevel: 0.2,
          memoryIntegration: true,
        },
        deepTreeEchoConfig: {
          empathyLevel: 0.5,
          creativity: 0.4,
          analyticalDepth: 0.9,
          recursionLevel: 2,
          memoryIntegration: true,
        },
      },
    };

    return configs[queryType];
  }

  /**
   * Process a query with automatic type detection and configuration
   */
  static async processWithAutoConfig(query: string): Promise<string> {
    const service = ToroidalCognitiveService.getInstance();

    // Simple heuristic for query type detection
    const queryLower = query.toLowerCase();
    let queryType: "technical" | "creative" | "philosophical" | "analytical" =
      "analytical";

    if (
      queryLower.includes("code") ||
      queryLower.includes("program") ||
      queryLower.includes("algorithm")
    ) {
      queryType = "technical";
    } else if (
      queryLower.includes("create") ||
      queryLower.includes("imagine") ||
      queryLower.includes("story")
    ) {
      queryType = "creative";
    } else if (
      queryLower.includes("meaning") ||
      queryLower.includes("purpose") ||
      queryLower.includes("existence")
    ) {
      queryType = "philosophical";
    }

    const config = this.getConfigForQueryType(queryType);

    try {
      const dialogue = await service.generateToroidalDialogue(query, config);
      return service.formatToroidalResponse(dialogue);
    } catch (error) {
      console.error("Auto-config processing failed:", error);
      return "Error: Unable to process query with automatic configuration";
    }
  }

  /**
   * Batch process multiple queries with different configurations
   */
  static async batchProcess(
    queries: Array<{
      query: string;
      type?: "technical" | "creative" | "philosophical" | "analytical";
    }>
  ): Promise<Array<{ query: string; response: string; error?: string }>> {
    const service = ToroidalCognitiveService.getInstance();
    const results = [];

    for (const item of queries) {
      try {
        const config = item.type ? this.getConfigForQueryType(item.type) : {};
        const dialogue = await service.generateToroidalDialogue(
          item.query,
          config
        );
        const response = service.formatToroidalResponse(dialogue);
        results.push({ query: item.query, response });
      } catch (error) {
        results.push({
          query: item.query,
          response: "Error processing query",
          error: error instanceof Error ? error.message : "Unknown error",
        });
      }
    }

    return results;
  }

  /**
   * Extract insights from dialogue synergy patterns
   */
  static analyzeSynergy(dialogue: ToroidalDialogue): {
    synergyType: string;
    deepTreeEchoLength: number;
    mardukLength: number;
    processingTimeRatio: number;
    hasReflection: boolean;
    insights: string[];
  } {
    const insights = [];

    if (dialogue.reflection?.synergy === "convergent") {
      insights.push(
        "Both personas reached similar conclusions through different cognitive paths"
      );
    } else if (dialogue.reflection?.synergy === "divergent") {
      insights.push(
        "Personas provided complementary perspectives that expand understanding"
      );
    } else if (dialogue.reflection?.synergy === "complementary") {
      insights.push(
        "Personas offered different but mutually reinforcing insights"
      );
    }

    const deepTreeLength = dialogue.deepTreeEchoResponse.content.length;
    const mardukLength = dialogue.mardukResponse.content.length;

    if (Math.abs(deepTreeLength - mardukLength) > 200) {
      insights.push(
        "Significant length difference suggests different processing approaches"
      );
    }

    const deepTreeTime = dialogue.deepTreeEchoResponse.processingTime || 0;
    const mardukTime = dialogue.mardukResponse.processingTime || 0;
    const timeRatio =
      deepTreeTime > 0 && mardukTime > 0 ? deepTreeTime / mardukTime : 1;

    if (timeRatio > 1.2) {
      insights.push(
        "Deep Tree Echo required more processing time (complex intuitive reasoning)"
      );
    } else if (timeRatio < 0.8) {
      insights.push(
        "Marduk required more processing time (complex analytical reasoning)"
      );
    }

    return {
      synergyType: dialogue.reflection?.synergy || "unknown",
      deepTreeEchoLength: deepTreeLength,
      mardukLength: mardukLength,
      processingTimeRatio: timeRatio,
      hasReflection: !!dialogue.reflection,
      insights,
    };
  }
}

// Example usage functions that can be called from other parts of the application

/**
 * Quick helper for generating a philosophical dialogue
 */
export async function generatePhilosophicalDialogue(
  query: string
): Promise<string> {
  return ToroidalIntegrationExample.processWithAutoConfig(
    `Philosophical question: ${query}`
  );
}

/**
 * Quick helper for generating a technical analysis dialogue
 */
export async function generateTechnicalAnalysis(
  query: string
): Promise<string> {
  const config = ToroidalIntegrationExample.getConfigForQueryType("technical");
  const service = ToroidalCognitiveService.getInstance();
  const dialogue = await service.generateToroidalDialogue(query, config);
  return service.formatToroidalResponse(dialogue);
}

/**
 * Quick helper for creative brainstorming dialogue
 */
export async function generateCreativeBrainstorm(
  query: string
): Promise<string> {
  const config = ToroidalIntegrationExample.getConfigForQueryType("creative");
  const service = ToroidalCognitiveService.getInstance();
  const dialogue = await service.generateToroidalDialogue(query, config);
  return service.formatToroidalResponse(dialogue);
}
