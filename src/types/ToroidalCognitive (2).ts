// Types for the Toroidal Cognitive Architecture
export interface PersonaResponse {
  persona: "deepTreeEcho" | "marduk";
  content: string;
  timestamp: Date;
  processingTime?: number;
}

export interface ToroidalDialogue {
  deepTreeEchoResponse: PersonaResponse;
  mardukResponse: PersonaResponse;
  reflection?: {
    content: string;
    synergy?: "convergent" | "divergent" | "complementary";
    unified_answer?: string;
  };
  metadata: {
    queryId: string;
    totalProcessingTime: number;
    contextType: "analytical" | "creative" | "philosophical" | "balanced";
  };
}

export interface PersonaConfig {
  creativity: number; // 0-1 scale
  analyticalDepth: number; // 0-1 scale
  recursionLevel: number; // 1-5 scale for Marduk
  empathyLevel: number; // 0-1 scale for Deep Tree Echo
  memoryIntegration: boolean;
}

export interface ToroidalCognitiveOptions {
  creativityLevel?: "analytical" | "creative" | "philosophical" | "balanced";
  includeReflection?: boolean;
  includeMemories?: boolean;
  maxTokensPerPersona?: number;
  temperature?: number;
  deepTreeEchoConfig?: Partial<PersonaConfig>;
  mardukConfig?: Partial<PersonaConfig>;
}
