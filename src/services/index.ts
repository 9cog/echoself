/**
 * Centralized exports for all services in the Deep Tree Echo system
 */

// Core services
export { DeepTreeEchoService } from "./deepTreeEchoService";
export { DeepTreeEchoOpenAIService } from "./openaiService";

// Evolutionary Configuration System
export {
  EvolutionaryConfigManager,
  getEvolutionaryConfig,
  getConfig,
  setConfig,
  reportFitness,
  createDefaultConfig,
} from "./evolutionaryConfig";
export type {
  EvolutionaryParameter,
  EvolutionaryBounds,
  CognitiveConfig,
  EvolutionaryMetrics,
} from "./evolutionaryConfig";

// Toroidal Cognitive System services
export { default as ToroidalCognitiveService } from "./toroidalCognitiveService";
export { default as MardukScientistService } from "./mardukScientistService";

// Relevance Realization Integration (Plan Implementation Phase 1)
export {
  RelevanceRealizationClient,
  getRelevanceClient,
  resetRelevanceClient,
  createAdaptiveFeedbackIntegration,
} from "./relevanceRealizationClient";

// Wisdom Cultivation Integration (Plan Implementation Phase 3)
export {
  WisdomCultivationClient,
  getWisdomClient,
  resetWisdomClient,
} from "./wisdomCultivationClient";

// Perspectival Service
export {
  PerspectivalService,
  getPerspectivalService,
  resetPerspectivalService,
} from "./perspectivalService";

// Other services
export * from "./stackblitzService";
export { Mech0Client, getMech0Client } from "./mech0Client";

// Types from Relevance Realization
export type {
  Possibility,
  RelevanceResult,
  RelevanceContext,
  OpponentStates,
  RelevanceCriteria,
  EngineState,
  FeedbackOutcome,
} from "./relevanceRealizationClient";

// Types from Wisdom Cultivation
export type {
  Belief,
  Insight,
  SelfDeception,
  VirtueType,
  WisdomScore,
  WisdomState,
  CultivationResult,
  RegulationAssessment,
} from "./wisdomCultivationClient";

// Types from Perspectival Service
export type {
  Frame,
  FrameType,
  SalienceLandscape,
  Aspect,
  GestaltShift,
  PerceptionResult,
} from "./perspectivalService";

// Types
export type {
  ToroidalResponse,
  ToroidalOptions,
} from "./toroidalCognitiveService";
