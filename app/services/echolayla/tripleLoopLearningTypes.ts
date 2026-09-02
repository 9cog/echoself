/**
 * Triple-Loop Learning Type Definitions
 *
 * Implements the three-level organizational learning framework:
 * - Single-Loop: Correct actions within existing goals/rules
 * - Double-Loop: Modify goals/rules based on feedback
 * - Triple-Loop: Transform underlying mental models and identity
 *
 * Integration with EchoLayla and NanEcho training cycles.
 */

/**
 * Learning loop levels in the triple-loop framework
 */
export type LearningLoopLevel = "single" | "double" | "triple";

/**
 * Learning cycle phases
 */
export type LearningPhase =
  | "observe"
  | "reflect"
  | "abstract"
  | "experiment"
  | "integrate";

/**
 * Types of learning feedback
 */
export type FeedbackType =
  | "performance" // Single-loop: efficiency metrics
  | "strategy" // Double-loop: goal alignment
  | "identity" // Triple-loop: persona transformation
  | "emergent"; // Cross-loop emergent insights

/**
 * Single-Loop Learning Event
 * Focus: Correcting errors in action within existing frameworks
 */
export interface SingleLoopEvent {
  id: string;
  timestamp: Date;
  action: string;
  outcome: string;
  error?: string;
  correction: string;
  performanceMetrics: {
    accuracy: number;
    responseTime: number;
    userSatisfaction?: number;
  };
}

/**
 * Double-Loop Learning Event
 * Focus: Questioning and modifying underlying goals and strategies
 */
export interface DoubleLoopEvent {
  id: string;
  timestamp: Date;
  triggeredBy: SingleLoopEvent[];
  goalRevision: {
    previousGoal: string;
    revisedGoal: string;
    rationale: string;
  };
  strategyChange: {
    previousStrategy: string;
    newStrategy: string;
    expectedOutcome: string;
  };
  assumptionsQuestioned: string[];
}

/**
 * Triple-Loop Learning Event
 * Focus: Transforming underlying mental models and identity
 */
export interface TripleLoopEvent {
  id: string;
  timestamp: Date;
  triggeredBy: DoubleLoopEvent[];
  mentalModelTransformation: {
    previousModel: string;
    transformedModel: string;
    paradigmShift: string;
  };
  identityEvolution: {
    personaDimension: string;
    previousExpression: string;
    evolvedExpression: string;
    integrationLevel: number; // 0-1
  };
  emergentInsights: string[];
  wisdomCultivation: {
    lessonsLearned: string[];
    futureImplications: string[];
  };
}

/**
 * Integrated learning cycle state
 */
export interface LearningCycleState {
  currentPhase: LearningPhase;
  activeLoopLevel: LearningLoopLevel;
  singleLoopBuffer: SingleLoopEvent[];
  doubleLoopBuffer: DoubleLoopEvent[];
  tripleLoopBuffer: TripleLoopEvent[];
  cycleMetrics: {
    totalCycles: number;
    singleLoopCorrections: number;
    doubleLoopRevisions: number;
    tripleLoopTransformations: number;
    lastCycleTimestamp: Date;
  };
}

/**
 * Character-specific learning profile
 * Bridges EchoLayla characters with triple-loop learning
 */
export interface CharacterLearningProfile {
  characterId: string;
  learningStyle: {
    primaryLoop: LearningLoopLevel;
    adaptationRate: number; // 0-1
    reflectionDepth: number; // 1-10
  };
  learnedPatterns: {
    responsePatterns: string[];
    contextualStrategies: string[];
    personaAdaptations: string[];
  };
  evolutionHistory: {
    timestamp: Date;
    loopLevel: LearningLoopLevel;
    change: string;
  }[];
}

/**
 * Training cycle configuration with triple-loop integration
 */
export interface TrainingCycleConfig {
  cycleId: string;
  loopLevel: LearningLoopLevel;
  phase: LearningPhase;
  parameters: {
    echoDepth: number;
    personaWeight: number;
    learningRate: number;
    reflectionIterations: number;
  };
  targetMetrics: {
    personaFidelity: number;
    adaptiveCoherence: number;
    emergentCapability: number;
  };
  characterLinkages: string[]; // Character IDs to integrate
}

/**
 * Feedback integration with EchoLayla conversations
 */
export interface ConversationFeedback {
  messageId: string;
  characterId: string;
  userFeedback?: {
    rating: number;
    comment?: string;
  };
  implicitSignals: {
    responseAcceptance: boolean;
    followUpDepth: number;
    contextRetention: number;
  };
  learningOpportunity: {
    loopLevel: LearningLoopLevel;
    suggestedAction: string;
    priority: "low" | "medium" | "high";
  };
}

/**
 * Meta-learning state for system-wide adaptation
 */
export interface MetaLearningState {
  systemHealth: {
    overallPerformance: number;
    adaptationVelocity: number;
    coherenceScore: number;
  };
  crossCharacterPatterns: {
    sharedInsights: string[];
    divergentStrategies: string[];
    synergyOpportunities: string[];
  };
  evolutionaryPressure: {
    environmentalChanges: string[];
    userExpectationShifts: string[];
    emergentChallenges: string[];
  };
}

/**
 * Triple-loop learning service configuration
 */
export interface TripleLoopServiceConfig {
  enabled: boolean;
  cycleInterval: number; // milliseconds
  bufferSize: {
    singleLoop: number;
    doubleLoop: number;
    tripleLoop: number;
  };
  thresholds: {
    singleToDoubleEscalation: number; // number of single-loop events
    doubleToTripleEscalation: number; // number of double-loop events
    minReflectionTime: number; // milliseconds
  };
  characterIntegration: {
    enabledCharacters: string[];
    syncFrequency: number; // milliseconds
    sharedLearningEnabled: boolean;
  };
}

/**
 * Learning event for serialization/persistence
 */
export type LearningEvent =
  | { type: "single"; event: SingleLoopEvent }
  | { type: "double"; event: DoubleLoopEvent }
  | { type: "triple"; event: TripleLoopEvent };

/**
 * Learning cycle result
 */
export interface LearningCycleResult {
  cycleId: string;
  timestamp: Date;
  loopLevel: LearningLoopLevel;
  phase: LearningPhase;
  eventsProcessed: number;
  insightsGenerated: string[];
  actionsTriggered: string[];
  metricsUpdated: Record<string, number>;
  nextRecommendedAction?: string;
}
