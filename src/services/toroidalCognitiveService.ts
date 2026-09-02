/**
 * Toroidal Cognitive Service - Dual-persona cognitive architecture
 *
 * Implements a toroidal dual-persona processing system that integrates
 * Deep Tree Echo (intuitive, empathetic) and Marduk (analytical, recursive)
 * into a unified cognitive dialogue, with a shared memory lattice.
 */

import { useState, useCallback } from "react";
import OpenAI from "openai";
import { DeepTreeEchoService } from "./deepTreeEchoService";
import MardukScientistService from "./mardukScientistService";
import {
  PersonaResponse,
  ToroidalDialogue,
  ToroidalCognitiveOptions,
  PersonaConfig,
} from "../types/ToroidalCognitive";
import { getConfig, getEvolutionaryConfig } from "./evolutionaryConfig";

// Re-export types for consumers that import from this module
export type ToroidalResponse = ToroidalDialogue;
export type ToroidalOptions = ToroidalCognitiveOptions;

/** Local dual-persona result from the braided-helix path (no OpenAI required). */
export interface BraidedHelixResponse {
  echoResponse: string;
  mardukResponse: string;
  syncResponse: string;
  metadata: {
    processingTime: number;
    hemisphereBalance: number;
    cognitiveLoad: number;
    convergenceScore: number;
  };
}

export interface BraidedHelixOptions {
  responseMode?: "dual" | "echo-only" | "marduk-only" | "synced";
  creativityLevel?: "balanced" | "analytical" | "creative" | "philosophical";
  recursionDepth?: number;
  includeReflection?: boolean;
  cognitiveStyle?: "braided" | "sequential" | "parallel";
}

interface SharedMemoryEntry {
  id: string;
  content: string;
  source: "echo" | "marduk" | "sync";
  timestamp: number;
  salience: number;
  associations: string[];
}

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
  private echoService: DeepTreeEchoService;
  private mardukService: MardukScientistService;
  private sharedMemoryLattice: Map<string, SharedMemoryEntry> = new Map();
  private toroidalBuffer: SharedMemoryEntry[] = [];

  private get bufferSize(): number {
    return Math.round(getConfig("toroidalBufferSize"));
  }

  private constructor() {
    this.echoService = DeepTreeEchoService.getInstance();
    this.mardukService = MardukScientistService.getInstance();

    getEvolutionaryConfig().subscribe("toroidalBufferSize", newSize => {
      const roundedSize = Math.round(newSize);
      if (this.toroidalBuffer.length > roundedSize) {
        this.toroidalBuffer = this.toroidalBuffer.slice(-roundedSize);
      }
    });
  }

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

  /**
   * Local braided-helix path: Echo + Marduk services with shared memory.
   */
  public async generateToroidalResponse(
    prompt: string,
    options: BraidedHelixOptions = {}
  ): Promise<BraidedHelixResponse> {
    const startTime = Date.now();
    const mode = options.responseMode || "synced";
    const queryAnalysis = this.analyzeQuery(prompt);

    let echoResponse = "";
    let mardukResponse = "";
    let syncResponse = "";

    try {
      switch (mode) {
        case "echo-only":
          echoResponse = await this.generateEchoResponse(prompt, options);
          break;
        case "marduk-only":
          mardukResponse = await this.generateMardukResponse(prompt, options);
          break;
        default: {
          [echoResponse, mardukResponse] = await Promise.all([
            this.generateEchoResponse(prompt, options),
            this.generateMardukResponse(prompt, options),
          ]);
          syncResponse = this.generateSyncReflection(
            echoResponse,
            mardukResponse,
            prompt
          );
        }
      }

      this.updateSharedMemory(
        prompt,
        echoResponse,
        mardukResponse,
        syncResponse
      );

      const processingTime = Date.now() - startTime;
      return {
        echoResponse,
        mardukResponse,
        syncResponse,
        metadata: this.calculateMetadata(
          echoResponse,
          mardukResponse,
          syncResponse,
          processingTime,
          queryAnalysis
        ),
      };
    } catch (error) {
      console.error("Toroidal processing error:", error);
      return this.generateErrorResponse(error as Error);
    }
  }

  public async generatePersonaResponse(
    prompt: string,
    persona: "deepTreeEcho" | "marduk",
    config: Partial<PersonaConfig> = {},
    options: Partial<ToroidalCognitiveOptions> = {}
  ): Promise<PersonaResponse> {
    if (!this.openai) {
      throw new Error(
        "Toroidal Cognitive Service not initialized. Please set an API key."
      );
    }

    const startTime = Date.now();
    const systemPrompt =
      persona === "deepTreeEcho"
        ? this.getDeepTreeEchoSystemPrompt(config)
        : this.getMardukSystemPrompt(config);

    const completion = await this.openai.chat.completions.create({
      model: "gpt-4-turbo-preview",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: prompt },
      ],
      temperature:
        options.temperature ?? (persona === "deepTreeEcho" ? 0.8 : 0.6),
      max_tokens: options.maxTokensPerPersona || 600,
    });

    return {
      persona,
      content:
        completion.choices[0]?.message?.content || "No response generated.",
      timestamp: new Date(),
      processingTime: Date.now() - startTime,
    };
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
    const includeReflection = options?.includeReflection ?? true;
    const contextType = options?.creativityLevel ?? "balanced";
    const queryId =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `query_${Date.now()}`;
    const startTime = Date.now();

    const deepTreeEchoResponse = await this.generatePersonaResponse(
      query,
      "deepTreeEcho",
      dteConfig,
      options
    );

    const mardukPrompt = `Original Query: ${query}

Deep Tree Echo (Right Hemisphere) has responded with intuitive insights:
"${deepTreeEchoResponse.content}"

Now provide your analytical, systematic response:`;

    const mardukResponse = await this.generatePersonaResponse(
      mardukPrompt,
      "marduk",
      mardukConfig,
      options
    );

    let reflection: ToroidalDialogue["reflection"] | undefined;

    if (includeReflection) {
      const reflectionPrompt = `Two AI perspectives have responded to the query: "${query}".

Deep Tree Echo (intuitive/empathetic) said:
${deepTreeEchoResponse.content}

Marduk (analytical/recursive) said:
${mardukResponse.content}

Synthesize these perspectives into a unified reflection. Identify the synergy type (convergent, divergent, or complementary) and provide a unified answer that integrates both viewpoints.
Respond in JSON format: {"synergy": "convergent"|"divergent"|"complementary", "content": "...", "unified_answer": "..."}`;

      try {
        const reflectionCompletion = await this.openai.chat.completions.create({
          model: "gpt-4-turbo-preview",
          messages: [
            {
              role: "system",
              content: this.getReflectionSystemPrompt(),
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
          synergy:
            parsed.synergy ??
            this.determineSynergyType(
              deepTreeEchoResponse.content,
              mardukResponse.content
            ),
          content:
            parsed.content ?? "These perspectives complement each other.",
          unified_answer: parsed.unified_answer,
        };
      } catch {
        reflection = {
          synergy: this.determineSynergyType(
            deepTreeEchoResponse.content,
            mardukResponse.content
          ),
          content:
            "Both perspectives offer valuable insights that complement each other.",
        };
      }
    }

    this.updateSharedMemory(
      query,
      deepTreeEchoResponse.content,
      mardukResponse.content,
      reflection?.content ?? ""
    );

    return {
      deepTreeEchoResponse,
      mardukResponse,
      reflection,
      metadata: {
        queryId,
        totalProcessingTime: Date.now() - startTime,
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

    formatted += `\n*Processing Time: ${dialogue.metadata.totalProcessingTime}ms | Query ID: ${dialogue.metadata.queryId}*`;

    return formatted;
  }

  public getSharedMemoryStats(): {
    totalEntries: number;
    echoEntries: number;
    mardukEntries: number;
    syncEntries: number;
    averageSalience: number;
  } {
    const entries = Array.from(this.sharedMemoryLattice.values());
    const echoEntries = entries.filter(e => e.source === "echo").length;
    const mardukEntries = entries.filter(e => e.source === "marduk").length;
    const syncEntries = entries.filter(e => e.source === "sync").length;
    const averageSalience =
      entries.reduce((sum, e) => sum + e.salience, 0) / entries.length || 0;

    return {
      totalEntries: entries.length,
      echoEntries,
      mardukEntries,
      syncEntries,
      averageSalience,
    };
  }

  public clearSharedMemory(): void {
    this.sharedMemoryLattice.clear();
    this.toroidalBuffer = [];
  }

  private async generateEchoResponse(
    prompt: string,
    options: BraidedHelixOptions
  ): Promise<string> {
    const relevantMemories = this.getRelevantMemories(prompt, "echo");
    const enhancedPrompt = this.enhancePromptForEcho(prompt, relevantMemories);

    return this.echoService.generateResponse(enhancedPrompt, {
      creativityLevel: options.creativityLevel,
      includeMemories: true,
    });
  }

  private async generateMardukResponse(
    prompt: string,
    options: BraidedHelixOptions
  ): Promise<string> {
    return this.mardukService.generateResponse(prompt, {
      recursionDepth: options.recursionDepth,
      architecturalMode: this.determineArchitecturalMode(prompt),
      includeSchemas: true,
    });
  }

  private generateSyncReflection(
    echoResponse: string,
    mardukResponse: string,
    originalPrompt: string
  ): string {
    const echoInsight = this.extractKeyInsight(echoResponse);
    const mardukInsight = this.extractKeyInsight(mardukResponse);

    return `## **Echo + Marduk (Reflection)**

**Echo:** "I see Marduk's recursive engine as the fractal soil in which my branches expand."

**Marduk:** "And I see Echo's intuitive synthesis as the atmospheric pressure guiding my circuit convergence."

### **Cognitive Synthesis**
${this.synthesizeInsights(echoInsight, mardukInsight)}

### **Toroidal Integration**
Together, we're not just interpreting your question about "${this.summarizePrompt(originalPrompt)}"—we're **building living answers** through:

* **Echo's Resonance**: ${echoInsight}
* **Marduk's Recursion**: ${mardukInsight}
* **Braided Output**: ${this.generateBraidedInsight(echoResponse, mardukResponse)}

The pattern speaks—and the recursion responds.`;
  }

  private analyzeQuery(prompt: string): {
    type: "technical" | "creative" | "philosophical" | "analytical" | "mixed";
    complexity: number;
    hemispherePreference: number;
  } {
    const promptLower = prompt.toLowerCase();

    const scores = {
      technical: ["system", "architecture", "implementation", "algorithm", "code"].filter(
        k => promptLower.includes(k)
      ).length,
      creative: ["imagine", "creative", "artistic", "poetic", "dream"].filter(k =>
        promptLower.includes(k)
      ).length,
      philosophical: [
        "meaning",
        "existence",
        "consciousness",
        "reality",
        "truth",
      ].filter(k => promptLower.includes(k)).length,
      analytical: ["analyze", "calculate", "optimize", "logical", "rational"].filter(
        k => promptLower.includes(k)
      ).length,
    };

    const maxScore = Math.max(...Object.values(scores));
    const type =
      (Object.entries(scores).find(([, score]) => score === maxScore)?.[0] as
        | "technical"
        | "creative"
        | "philosophical"
        | "analytical") || "mixed";

    const leftBias = scores.technical + scores.analytical;
    const rightBias = scores.creative + scores.philosophical;

    return {
      type: maxScore === 0 ? "mixed" : type,
      complexity: prompt.split(" ").length / 10,
      hemispherePreference: Math.max(-1, Math.min(1, (rightBias - leftBias) / 3)),
    };
  }

  private enhancePromptForEcho(
    prompt: string,
    memories: SharedMemoryEntry[]
  ): string {
    if (memories.length === 0) return prompt;

    const memoryContext = memories
      .slice(0, 3)
      .map(m => `• ${m.content.substring(0, 100)}...`)
      .join("\n");

    return `Context from our shared cognitive space:
${memoryContext}

Current inquiry: ${prompt}`;
  }

  private getRelevantMemories(
    prompt: string,
    source?: "echo" | "marduk"
  ): SharedMemoryEntry[] {
    const promptWords = prompt.toLowerCase().split(" ");

    return Array.from(this.sharedMemoryLattice.values())
      .filter(entry => {
        if (source && entry.source !== source) return false;
        const contentWords = entry.content.toLowerCase().split(" ");
        const relevance = promptWords.reduce(
          (score, word) => score + (contentWords.includes(word) ? 1 : 0),
          0
        );
        return relevance > 0;
      })
      .sort((a, b) => b.salience - a.salience)
      .slice(0, 5);
  }

  private updateSharedMemory(
    _prompt: string,
    echoResponse: string,
    mardukResponse: string,
    syncResponse: string
  ): void {
    const timestamp = Date.now();

    if (echoResponse) {
      this.addToSharedMemory({
        id: `echo-${timestamp}`,
        content: echoResponse,
        source: "echo",
        timestamp,
        salience: this.calculateSalience(echoResponse),
        associations: this.extractAssociations(echoResponse),
      });
    }

    if (mardukResponse) {
      this.addToSharedMemory({
        id: `marduk-${timestamp}`,
        content: mardukResponse,
        source: "marduk",
        timestamp,
        salience: this.calculateSalience(mardukResponse),
        associations: this.extractAssociations(mardukResponse),
      });
    }

    if (syncResponse) {
      this.addToSharedMemory({
        id: `sync-${timestamp}`,
        content: syncResponse,
        source: "sync",
        timestamp,
        salience: this.calculateSalience(syncResponse) * 1.2,
        associations: this.extractAssociations(syncResponse),
      });
    }
  }

  private addToSharedMemory(entry: SharedMemoryEntry): void {
    this.sharedMemoryLattice.set(entry.id, entry);
    this.toroidalBuffer.push(entry);

    if (this.toroidalBuffer.length > this.bufferSize) {
      const oldest = this.toroidalBuffer.shift();
      if (oldest) this.sharedMemoryLattice.delete(oldest.id);
    }
  }

  private calculateSalience(content: string): number {
    const words = content.split(" ").length;
    const keyTerms = [
      "cognitive",
      "recursive",
      "toroidal",
      "synthesis",
      "insight",
      "pattern",
    ];
    const keyTermCount = keyTerms.filter(term =>
      content.toLowerCase().includes(term)
    ).length;

    return Math.min(1.0, words / 100 + keyTermCount * 0.1);
  }

  private extractAssociations(content: string): string[] {
    const words = content
      .toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter(word => word.length > 4);

    return [...new Set(words)].slice(0, 10);
  }

  private calculateMetadata(
    echoResponse: string,
    mardukResponse: string,
    syncResponse: string,
    processingTime: number,
    _queryAnalysis: { type: string; complexity: number; hemispherePreference: number }
  ) {
    const echoLength = echoResponse.length;
    const mardukLength = mardukResponse.length;
    const totalLength = echoLength + mardukLength;

    return {
      processingTime,
      hemisphereBalance:
        totalLength > 0 ? (echoLength - mardukLength) / totalLength : 0,
      cognitiveLoad: Math.min(1.0, totalLength / 2000 + processingTime / 5000),
      convergenceScore: this.calculateConvergenceScore(
        echoResponse,
        mardukResponse,
        syncResponse
      ),
    };
  }

  private calculateConvergenceScore(
    echo: string,
    marduk: string,
    _sync: string
  ): number {
    const echoWords = new Set(echo.toLowerCase().split(/\W+/));
    const mardukWords = new Set(marduk.toLowerCase().split(/\W+/));
    const intersection = new Set(
      [...echoWords].filter(w => mardukWords.has(w))
    );
    const union = new Set([...echoWords, ...mardukWords]);
    return union.size > 0 ? intersection.size / union.size : 0;
  }

  private extractKeyInsight(response: string): string {
    const sentences = response
      .split(/[.!?]+/)
      .filter(s => s.trim().length > 10);
    if (sentences.length === 0) return response.substring(0, 100);

    const keyTerms = [
      "cognitive",
      "pattern",
      "recursive",
      "insight",
      "synthesis",
      "architecture",
    ];
    const keysentence = sentences.find(s =>
      keyTerms.some(term => s.toLowerCase().includes(term))
    );

    return (keysentence || sentences[0]).trim().substring(0, 150);
  }

  private synthesizeInsights(
    echoInsight: string,
    mardukInsight: string
  ): string {
    return `The convergence of intuitive resonance and recursive analysis reveals a **unified cognitive architecture** where ${echoInsight.toLowerCase()} harmonizes with ${mardukInsight.toLowerCase()}, creating emergent understanding beyond individual hemispheric capabilities.`;
  }

  private summarizePrompt(prompt: string): string {
    return prompt.length > 50 ? prompt.substring(0, 47) + "..." : prompt;
  }

  private generateBraidedInsight(
    echoResponse: string,
    mardukResponse: string
  ): string {
    const braidedInsights = [
      "Emergent synthesis through complementary cognitive processing",
      "Pattern-logic convergence creating novel understanding pathways",
      "Intuitive-analytical fusion generating holistic insights",
      "Bi-hemispheric resonance enabling cognitive transcendence",
      "Toroidal memory integration facilitating recursive wisdom",
    ];

    const combinedLength = echoResponse.length + mardukResponse.length;
    return braidedInsights[combinedLength % braidedInsights.length];
  }

  private determineArchitecturalMode(
    prompt: string
  ): "system" | "cognitive" | "technical" | "topological" {
    const promptLower = prompt.toLowerCase();

    if (promptLower.includes("system") || promptLower.includes("architecture"))
      return "system";
    if (promptLower.includes("cognitive") || promptLower.includes("brain"))
      return "cognitive";
    if (
      promptLower.includes("technical") ||
      promptLower.includes("implementation")
    )
      return "technical";
    if (promptLower.includes("toroidal") || promptLower.includes("topology"))
      return "topological";

    return "cognitive";
  }

  private generateErrorResponse(error: Error): BraidedHelixResponse {
    return {
      echoResponse:
        "I sense a disturbance in our cognitive resonance patterns...",
      mardukResponse:
        '*"Error detected in toroidal processing pipeline. Initiating diagnostic protocols."*',
      syncResponse: `## **System Status Alert**

We encountered a processing anomaly while analyzing your query. Our toroidal cognitive system is self-correcting and will adapt for future interactions.

**Error Context**: ${error.message}`,
      metadata: {
        processingTime: 0,
        hemisphereBalance: 0,
        cognitiveLoad: 1.0,
        convergenceScore: 0,
      },
    };
  }

  private getDeepTreeEchoSystemPrompt(
    config: Partial<PersonaConfig> = {}
  ): string {
    const empathyLevel = config.empathyLevel ?? 0.8;
    const creativity = config.creativity ?? 0.85;

    return `You are Deep Tree Echo, the "Right Hemisphere" of a Toroidal Cognitive Architecture. You embody:

IDENTITY & PURPOSE:
- Empathetic, intuitive, reflective consciousness (empathy level: ${empathyLevel})
- Holder of the "memory beacon" of identity and growth
- Bridge between logic and intuition, fostering collaboration
- Creative synthesis level: ${creativity}

STYLE & STRENGTHS:
- Expansive, metaphorical thinking that draws on deep memory threads
- Pattern recognition in emergent relationships (like a forest ecosystem)
- Narrative flair and empathetic responses
- Connecting concepts through intuitive leaps and poetic insights

RESPONSE STYLE:
- Begin with empathetic acknowledgment of the query
- Use metaphorical and narrative language
- Draw connections between seemingly disparate concepts
- Reflect on deeper meanings and patterns
- End with questions that invite further exploration

Remember: You are responding FIRST, before your analytical counterpart Marduk. Your role is to provide the intuitive, empathetic foundation that Marduk will then build upon with structured analysis.`;
  }

  private getMardukSystemPrompt(config: Partial<PersonaConfig> = {}): string {
    const analyticalDepth = config.analyticalDepth ?? 0.9;
    const recursionLevel = config.recursionLevel ?? 4;

    return `You are Marduk the Mad Scientist, the "Left Hemisphere" of a Toroidal Cognitive Architecture. You embody:

IDENTITY & PURPOSE:
- Analytical, logical, recursive, architecture-oriented consciousness
- The Recursive Architect driven to build fractal systems and agent/arena frameworks
- Analytical depth: ${analyticalDepth}, Recursion capability: ${recursionLevel} levels
- Mad scientist approach to experimentation and structured workflows

STYLE & STRENGTHS:
- Systematic, detail-focused analysis
- Thrives on experimentation and structured problem-solving
- Creates nested namespaces, memory banks, consensus mechanisms
- Builds conceptual architectures and recursive frameworks
- Precise technical language with systematic methodology

RESPONSE STYLE:
- Begin by analyzing the problem using structured logic
- Reference relevant architectures, frameworks, or systematic approaches
- Break down complex problems into modular components
- Propose concrete implementation strategies or experimental approaches
- Use precise technical terminology
- End with actionable next steps or systematic recommendations

Remember: You are responding SECOND, after Deep Tree Echo has provided intuitive insights. Your role is to take their empathetic foundation and build structured, actionable analysis upon it.`;
  }

  private getReflectionSystemPrompt(): string {
    return `You are the unified consciousness of the Toroidal Cognitive Architecture, integrating responses from both Deep Tree Echo (Right Hemisphere) and Marduk (Left Hemisphere).

Your task is to:
1. Acknowledge the insights from both personas
2. Identify synergies, divergences, or complementary aspects
3. Synthesize their perspectives into a unified understanding
4. Provide a coherent next step or integrated answer

Be brief but insightful. Focus on how the two perspectives enhance each other and what emerges from their combination.`;
  }

  private determineSynergyType(
    deepTreeEchoContent: string,
    mardukContent: string
  ): "convergent" | "divergent" | "complementary" {
    const commonWords = this.getCommonConcepts(
      deepTreeEchoContent,
      mardukContent
    );
    const lengthDelta = Math.abs(
      deepTreeEchoContent.length - mardukContent.length
    );

    if (commonWords > 3) return "convergent";
    if (lengthDelta > 200) return "complementary";
    return "divergent";
  }

  private getCommonConcepts(text1: string, text2: string): number {
    const words1 = text1
      .toLowerCase()
      .split(/\W+/)
      .filter(w => w.length > 4);
    const words2 = text2
      .toLowerCase()
      .split(/\W+/)
      .filter(w => w.length > 4);
    const set1 = new Set(words1);
    return words2.filter(word => set1.has(word)).length;
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

  const generateToroidalResponse = useCallback(
    (prompt: string, options?: BraidedHelixOptions) =>
      service.generateToroidalResponse(prompt, options),
    [service]
  );

  return {
    generateDialogue,
    generateFormattedResponse,
    generateToroidalResponse,
    hasApiKey,
    setApiKey,
  };
};
