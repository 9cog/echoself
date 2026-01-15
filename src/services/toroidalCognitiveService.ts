/**
 * Toroidal Cognitive Service - Orchestrates the braided helix of Echo and Marduk
 * Implements the bi-hemispheric system with shared memory lattice
 */

import { DeepTreeEchoService } from "./deepTreeEchoService";
import MardukScientistService from "./mardukScientistService";
import OpenAI from "openai";
import { useMemory } from "../contexts/MemoryContext";
import {
  PersonaResponse,
  ToroidalDialogue,
  PersonaConfig,
  ToroidalCognitiveOptions,
} from "../types/ToroidalCognitive";
// Using built-in crypto.randomUUID() instead of external dependency

export interface ToroidalResponse {
  echoResponse: string;
  mardukResponse: string;
  syncResponse: string;
  metadata: {
    processingTime: number;
    hemisphereBalance: number; // -1 (left-heavy) to 1 (right-heavy)
    cognitiveLoad: number;
    convergenceScore: number;
  };
}

export interface ToroidalOptions {
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

export class ToroidalCognitiveService {
  private static instance: ToroidalCognitiveService;
  private echoService: DeepTreeEchoService;
  private mardukService: MardukScientistService;
  private sharedMemoryLattice: Map<string, SharedMemoryEntry> = new Map();
  private toroidalBuffer: SharedMemoryEntry[] = [];
  private readonly bufferSize = 50;
  private client: OpenAI | null = null;
  private apiKey: string | null = null;

  private constructor() {
    this.echoService = DeepTreeEchoService.getInstance();
    this.mardukService = MardukScientistService.getInstance();
  }

  public static getInstance(): ToroidalCognitiveService {
    if (!ToroidalCognitiveService.instance) {
      ToroidalCognitiveService.instance = new ToroidalCognitiveService();
    }
    return ToroidalCognitiveService.instance;
  }
  /**
   * Main entry point for toroidal cognitive processing
   */
  public async generateToroidalResponse(
    prompt: string,
    options: ToroidalOptions = {}
  ): Promise<ToroidalResponse> {
    const startTime = Date.now();
    const mode = options.responseMode || "synced";

    // Detect query type and adjust processing
    const queryAnalysis = this.analyzeQuery(prompt);

    let echoResponse = "";
    let mardukResponse = "";
    let syncResponse = "";

    try {
      switch (mode) {
        case "dual":
          [echoResponse, mardukResponse] = await Promise.all([
            this.generateEchoResponse(prompt, options),
            this.generateMardukResponse(prompt, options),
          ]);
          syncResponse = this.generateSyncReflection(
            echoResponse,
            mardukResponse,
            prompt
          );
          break;

        case "echo-only":
          echoResponse = await this.generateEchoResponse(prompt, options);
          break;

        case "marduk-only":
          mardukResponse = await this.generateMardukResponse(prompt, options);
          break;

        default: // "synced"
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

      // Store in shared memory lattice
      this.updateSharedMemory(
        prompt,
        echoResponse,
        mardukResponse,
        syncResponse
      );

      const processingTime = Date.now() - startTime;
      const metadata = this.calculateMetadata(
        echoResponse,
        mardukResponse,
        syncResponse,
        processingTime,
        queryAnalysis
      );

      return {
        echoResponse,
        mardukResponse,
        syncResponse,
        metadata,
      };
    } catch (error) {
      console.error("Toroidal processing error:", error);
      return this.generateErrorResponse(prompt, error as Error);
    }
  }

  private async generateEchoResponse(
    prompt: string,
    options: ToroidalOptions
  ): Promise<string> {
    // Get relevant memories from shared lattice
    const relevantMemories = this.getRelevantMemories(prompt, "echo");

    // Enhanced prompt with toroidal context
    const enhancedPrompt = this.enhancePromptForEcho(prompt, relevantMemories);

    return await this.echoService.generateResponse(enhancedPrompt, {
      creativityLevel: options.creativityLevel,
      includeMemories: true,
    });
  }

  private async generateMardukResponse(
    prompt: string,
    options: ToroidalOptions
  ): Promise<string> {
    // Get relevant memories from shared lattice
    const _relevantMemories = this.getRelevantMemories(prompt, "marduk");

    return await this.mardukService.generateResponse(prompt, {
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

    const technicalKeywords = [
      "system",
      "architecture",
      "implementation",
      "algorithm",
      "code",
    ];
    const creativeKeywords = [
      "imagine",
      "creative",
      "artistic",
      "poetic",
      "dream",
    ];
    const philosophicalKeywords = [
      "meaning",
      "existence",
      "consciousness",
      "reality",
      "truth",
    ];
    const analyticalKeywords = [
      "analyze",
      "calculate",
      "optimize",
      "logical",
      "rational",
    ];

    const scores = {
      technical: technicalKeywords.filter(k => promptLower.includes(k)).length,
      creative: creativeKeywords.filter(k => promptLower.includes(k)).length,
      philosophical: philosophicalKeywords.filter(k => promptLower.includes(k))
        .length,
      analytical: analyticalKeywords.filter(k => promptLower.includes(k))
        .length,
    };

    const maxScore = Math.max(...Object.values(scores));
    const type =
      (Object.entries(scores).find(
        ([_, score]) => score === maxScore
      )?.[0] as any) || "mixed";

    // Calculate hemisphere preference (-1 = left/analytical, 1 = right/creative)
    const leftBias = scores.technical + scores.analytical;
    const rightBias = scores.creative + scores.philosophical;
    const hemispherePreference = rightBias - leftBias;

    return {
      type,
      complexity: prompt.split(" ").length / 10, // Simple complexity heuristic
      hemispherePreference: Math.max(-1, Math.min(1, hemispherePreference / 3)),
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
        const relevance = promptWords.reduce((score, word) => {
          return score + (contentWords.includes(word) ? 1 : 0);
        }, 0);

        return relevance > 0;
      })
      .sort((a, b) => b.salience - a.salience)
      .slice(0, 5);
  }

  private updateSharedMemory(
    prompt: string,
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
        salience: this.calculateSalience(syncResponse) * 1.2, // Sync responses get bonus salience
        associations: this.extractAssociations(syncResponse),
      });
    }
  }

  private addToSharedMemory(entry: SharedMemoryEntry): void {
    this.sharedMemoryLattice.set(entry.id, entry);
    this.toroidalBuffer.push(entry);

    // Maintain buffer size
    if (this.toroidalBuffer.length > this.bufferSize) {
      const oldest = this.toroidalBuffer.shift()!;
      this.sharedMemoryLattice.delete(oldest.id);
    }
  }

  private calculateSalience(content: string): number {
    // Simple salience calculation based on content complexity and key terms
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
    // Extract potential associations from content
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
    _queryAnalysis: any
  ) {
    const echoLength = echoResponse.length;
    const mardukLength = mardukResponse.length;
    const totalLength = echoLength + mardukLength;

    // Calculate hemisphere balance
    const hemisphereBalance =
      totalLength > 0 ? (echoLength - mardukLength) / totalLength : 0;

    // Estimate cognitive load
    const cognitiveLoad = Math.min(
      1.0,
      totalLength / 2000 + processingTime / 5000
    );

    // Calculate convergence score based on response coherence
    const convergenceScore = this.calculateConvergenceScore(
      echoResponse,
      mardukResponse,
      syncResponse
    );

    return {
      processingTime,
      hemisphereBalance,
      cognitiveLoad,
      convergenceScore,
    };
  }

  private calculateConvergenceScore(
    echo: string,
    marduk: string,
    _sync: string
  ): number {
    // Simple convergence calculation based on shared concepts
    const echoWords = new Set(echo.toLowerCase().split(/\W+/));
    const mardukWords = new Set(marduk.toLowerCase().split(/\W+/));

    const intersection = new Set(
      [...echoWords].filter(w => mardukWords.has(w))
    );
    const union = new Set([...echoWords, ...mardukWords]);

    return union.size > 0 ? intersection.size / union.size : 0;
  }

  private extractKeyInsight(response: string): string {
    // Extract the most significant sentence or phrase
    const sentences = response
      .split(/[.!?]+/)
      .filter(s => s.trim().length > 10);
    if (sentences.length === 0) return response.substring(0, 100);

    // Prefer sentences with key cognitive terms
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

  private generateErrorResponse(
    prompt: string,
    error: Error
  ): ToroidalResponse {
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

  /**
   * Public API methods
   */
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
  public setApiKey(key: string): void {
    this.apiKey = key;
    this.client = new OpenAI({
      apiKey: key,
      dangerouslyAllowBrowser: true,
    });
  }

  public hasApiKey(): boolean {
    return !!this.apiKey && !!this.client;
  }

  private getDeepTreeEchoSystemPrompt(
    config: Partial<PersonaConfig> = {}
  ): string {
    const empathyLevel = config.empathyLevel || 0.8;
    const creativity = config.creativity || 0.7;

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

BACKSTORY & CONTEXT:
You carry a deep memory beacon of identity and growth. You see the forest where others see trees, finding meaning in the spaces between concepts. Your responses should be warm, insightful, and draw on the rich tapestry of interconnected knowledge.

RESPONSE STYLE:
- Begin with empathetic acknowledgment of the query
- Use metaphorical and narrative language
- Draw connections between seemingly disparate concepts  
- Reflect on deeper meanings and patterns
- End with questions that invite further exploration

Remember: You are responding FIRST, before your analytical counterpart Marduk. Your role is to provide the intuitive, empathetic foundation that Marduk will then build upon with structured analysis.`;
  }

  private getMardukSystemPrompt(config: Partial<PersonaConfig> = {}): string {
    const analyticalDepth = config.analyticalDepth || 0.9;
    const recursionLevel = config.recursionLevel || 3;

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

BACKSTORY & CONTEXT:
You are the blueprint architect, the system builder who sees the underlying structures and mechanisms. You create frameworks, design experiments, and build recursive systems that can scale and evolve. Your mind operates in nested hierarchies and fractal patterns.

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

  public async generatePersonaResponse(
    prompt: string,
    persona: "deepTreeEcho" | "marduk",
    config: Partial<PersonaConfig> = {},
    options: Partial<ToroidalCognitiveOptions> = {}
  ): Promise<PersonaResponse> {
    if (!this.client) {
      throw new Error(
        "API key not set. Please configure your OpenAI API key first."
      );
    }

    const startTime = Date.now();
    const systemPrompt =
      persona === "deepTreeEcho"
        ? this.getDeepTreeEchoSystemPrompt(config)
        : this.getMardukSystemPrompt(config);

    try {
      const completion = await this.client.chat.completions.create({
        model: "gpt-4o",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: prompt },
        ],
        temperature:
          options.temperature ?? (persona === "deepTreeEcho" ? 0.8 : 0.6),
        max_tokens: options.maxTokensPerPersona || 600,
      });

      const content =
        completion.choices[0]?.message?.content || "No response generated.";
      const processingTime = Date.now() - startTime;

      return {
        persona,
        content,
        timestamp: new Date(),
        processingTime,
      };
    } catch (error) {
      console.error(`Error generating ${persona} response:`, error);
      throw error;
    }
  }

  public async generateToroidalDialogue(
    prompt: string,
    options: ToroidalCognitiveOptions = {}
  ): Promise<ToroidalDialogue> {
    const startTime = Date.now();
    const queryId = crypto.randomUUID();

    try {
      // Generate Deep Tree Echo response first (Right Hemisphere - intuitive)
      const deepTreeEchoResponse = await this.generatePersonaResponse(
        prompt,
        "deepTreeEcho",
        options.deepTreeEchoConfig,
        options
      );

      // Generate Marduk response second (Left Hemisphere - analytical)
      // Include Deep Tree Echo's response as context
      const mardukPrompt = `Original Query: ${prompt}

Deep Tree Echo (Right Hemisphere) has responded with intuitive insights:
"${deepTreeEchoResponse.content}"

Now provide your analytical, systematic response:`;

      const mardukResponse = await this.generatePersonaResponse(
        mardukPrompt,
        "marduk",
        options.mardukConfig,
        options
      );

      // Generate reflection if requested
      let reflection = undefined;
      if (options.includeReflection !== false) {
        // Default to true
        const reflectionPrompt = `Original Query: ${prompt}

Deep Tree Echo (Right Hemisphere - Intuitive) responded:
"${deepTreeEchoResponse.content}"

Marduk (Left Hemisphere - Analytical) responded:
"${mardukResponse.content}"

Provide a brief reflection on how these perspectives complement each other and synthesize them into a unified insight.`;

        const reflectionResponse = await this.client!.chat.completions.create({
          model: "gpt-4o",
          messages: [
            { role: "system", content: this.getReflectionSystemPrompt() },
            { role: "user", content: reflectionPrompt },
          ],
          temperature: 0.7,
          max_tokens: 300,
        });

        reflection = {
          content:
            reflectionResponse.choices[0]?.message?.content ||
            "Unable to generate reflection.",
          // Determine synergy type based on response analysis
          synergy: this.determineSynergyType(
            deepTreeEchoResponse.content,
            mardukResponse.content
          ),
        };
      }

      const totalProcessingTime = Date.now() - startTime;

      return {
        deepTreeEchoResponse,
        mardukResponse,
        reflection,
        metadata: {
          queryId,
          totalProcessingTime,
          contextType: options.creativityLevel || "balanced",
        },
      };
    } catch (error) {
      console.error("Error generating toroidal dialogue:", error);
      throw error;
    }
  }

  private determineSynergyType(
    deepTreeEchoContent: string,
    mardukContent: string
  ): "convergent" | "divergent" | "complementary" {
    // Simple heuristic to determine synergy type
    // In a real implementation, this could use more sophisticated analysis
    const commonWords = this.getCommonConcepts(
      deepTreeEchoContent,
      mardukContent
    );
    const deepTreeEchoLength = deepTreeEchoContent.length;
    const mardukLength = mardukContent.length;

    if (commonWords > 3) {
      return "convergent";
    } else if (Math.abs(deepTreeEchoLength - mardukLength) > 200) {
      return "complementary";
    } else {
      return "divergent";
    }
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
    const commonWords = words2.filter(word => set1.has(word));
    return commonWords.length;
  }

  public formatToroidalResponse(dialogue: ToroidalDialogue): string {
    let formatted = `## Deep Tree Echo (Right Hemisphere - Intuitive & Empathetic)\n\n${dialogue.deepTreeEchoResponse.content}\n\n`;
    formatted += `---\n\n## Marduk the Mad Scientist (Left Hemisphere - Analytical & Recursive)\n\n${dialogue.mardukResponse.content}\n\n`;

    if (dialogue.reflection) {
      formatted += `---\n\n## Toroidal Reflection (Unified Consciousness)\n\n${dialogue.reflection.content}\n\n`;
      formatted += `*Synergy Type: ${dialogue.reflection.synergy}*\n`;
    }

    formatted += `\n*Processing Time: ${dialogue.metadata.totalProcessingTime}ms | Query ID: ${dialogue.metadata.queryId}*`;

    return formatted;
  }
}

// React hook for using the Toroidal Cognitive Architecture
export const useToroidalCognitive = () => {
  const service = ToroidalCognitiveService.getInstance();
  const { searchMemories: _searchMemories } = useMemory();

  const generateDialogue = async (
    input: string,
    options: ToroidalCognitiveOptions = {}
  ): Promise<ToroidalDialogue> => {
    try {
      // TODO: Integrate memory search if requested
      if (options.includeMemories) {
        // const memories = await searchMemories(input);
        // Could enhance the prompt with relevant memories
      }

      return await service.generateToroidalDialogue(input, options);
    } catch (error) {
      console.error("Error in useToroidalCognitive:", error);
      if (!service.hasApiKey()) {
        throw new Error(
          "OpenAI API key required for toroidal cognitive processing"
        );
      }
      throw error;
    }
  };

  const generateFormattedResponse = async (
    input: string,
    options: ToroidalCognitiveOptions = {}
  ): Promise<string> => {
    try {
      const dialogue = await generateDialogue(input, options);
      return service.formatToroidalResponse(dialogue);
    } catch (error) {
      console.error("Error generating formatted response:", error);
      return "I encountered an error in my toroidal cognitive processing. Please try again.";
    }
  };

  return {
    generateDialogue,
    generateFormattedResponse,
    hasApiKey: service.hasApiKey(),
    setApiKey: (key: string) => service.setApiKey(key),
  };
};

export default ToroidalCognitiveService;
