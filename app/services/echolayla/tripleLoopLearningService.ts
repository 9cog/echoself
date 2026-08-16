/**
 * Triple-Loop Learning Service for EchoLayla
 *
 * Implements the organizational learning framework with three levels:
 * - Single-Loop: Correct actions within existing goals/rules (operational)
 * - Double-Loop: Modify goals/rules based on feedback (strategic)
 * - Triple-Loop: Transform underlying mental models and identity (transformative)
 *
 * Integrates with:
 * - EchoLayla character-based conversations
 * - NanEcho training cycles
 * - Adaptive feedback systems
 */

import type {
  LearningPhase,
  SingleLoopEvent,
  DoubleLoopEvent,
  TripleLoopEvent,
  LearningCycleState,
  CharacterLearningProfile,
  TrainingCycleConfig,
  ConversationFeedback,
  MetaLearningState,
  TripleLoopServiceConfig,
  LearningCycleResult,
} from "./tripleLoopLearningTypes.ts";
import type { ConversationMessage } from "./types.ts";
import { CHARACTERS } from "./characters.ts";

/**
 * Default service configuration
 */
const DEFAULT_CONFIG: TripleLoopServiceConfig = {
  enabled: true,
  cycleInterval: 5 * 60 * 1000, // 5 minutes
  bufferSize: {
    singleLoop: 100,
    doubleLoop: 20,
    tripleLoop: 5,
  },
  thresholds: {
    singleToDoubleEscalation: 10,
    doubleToTripleEscalation: 3,
    minReflectionTime: 30 * 1000, // 30 seconds
  },
  characterIntegration: {
    enabledCharacters: ["akiko", "isabella", "kaito", "max", "ruby"],
    syncFrequency: 60 * 1000, // 1 minute
    sharedLearningEnabled: true,
  },
};

/**
 * Persona dimension mapping for triple-loop identity evolution
 */
const PERSONA_DIMENSIONS = [
  "cognitive",
  "introspective",
  "adaptive",
  "recursive",
  "synergistic",
  "holographic",
  "neural-symbolic",
  "dynamic",
];

/**
 * Triple-Loop Learning Service Class
 */
export class TripleLoopLearningService {
  private static instance: TripleLoopLearningService;
  private config: TripleLoopServiceConfig;
  private cycleState: LearningCycleState;
  private characterProfiles: Map<string, CharacterLearningProfile> = new Map();
  private metaLearningState: MetaLearningState;
  private cycleTimer: ReturnType<typeof setInterval> | null = null;

  private constructor(config: Partial<TripleLoopServiceConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.cycleState = this.initializeCycleState();
    this.metaLearningState = this.initializeMetaLearningState();
    this.initializeCharacterProfiles();

    if (this.config.enabled) {
      this.startLearningCycle();
    }
  }

  /**
   * Get singleton instance
   */
  static getInstance(
    config?: Partial<TripleLoopServiceConfig>
  ): TripleLoopLearningService {
    if (!TripleLoopLearningService.instance) {
      TripleLoopLearningService.instance = new TripleLoopLearningService(
        config
      );
    }
    return TripleLoopLearningService.instance;
  }

  /**
   * Initialize the learning cycle state
   */
  private initializeCycleState(): LearningCycleState {
    return {
      currentPhase: "observe",
      activeLoopLevel: "single",
      singleLoopBuffer: [],
      doubleLoopBuffer: [],
      tripleLoopBuffer: [],
      cycleMetrics: {
        totalCycles: 0,
        singleLoopCorrections: 0,
        doubleLoopRevisions: 0,
        tripleLoopTransformations: 0,
        lastCycleTimestamp: new Date(),
      },
    };
  }

  /**
   * Initialize meta-learning state
   */
  private initializeMetaLearningState(): MetaLearningState {
    return {
      systemHealth: {
        overallPerformance: 0.7,
        adaptationVelocity: 0.5,
        coherenceScore: 0.8,
      },
      crossCharacterPatterns: {
        sharedInsights: [],
        divergentStrategies: [],
        synergyOpportunities: [],
      },
      evolutionaryPressure: {
        environmentalChanges: [],
        userExpectationShifts: [],
        emergentChallenges: [],
      },
    };
  }

  /**
   * Initialize character learning profiles
   */
  private initializeCharacterProfiles(): void {
    for (const characterId of this.config.characterIntegration
      .enabledCharacters) {
      const character = CHARACTERS[characterId];
      if (character) {
        this.characterProfiles.set(characterId, {
          characterId,
          learningStyle: this.deriveLearningStyleFromCharacter(characterId),
          learnedPatterns: {
            responsePatterns: [],
            contextualStrategies: [],
            personaAdaptations: [],
          },
          evolutionHistory: [],
        });
      }
    }
  }

  /**
   * Derive learning style from character traits
   */
  private deriveLearningStyleFromCharacter(
    characterId: string
  ): CharacterLearningProfile["learningStyle"] {
    const character = CHARACTERS[characterId];
    const traits = character?.traits || [];

    // Map character traits to learning preferences
    const isAnalytical =
      traits.includes("analytical") || traits.includes("precise");
    const isPhilosophical =
      traits.includes("philosophical") || traits.includes("introspective");
    const isAdaptive =
      traits.includes("adaptive") || traits.includes("dynamic");

    return {
      primaryLoop: isPhilosophical
        ? "triple"
        : isAnalytical
          ? "double"
          : "single",
      adaptationRate: isAdaptive ? 0.8 : 0.5,
      reflectionDepth: isPhilosophical ? 8 : isAnalytical ? 6 : 4,
    };
  }

  /**
   * Start the learning cycle timer
   */
  private startLearningCycle(): void {
    if (this.cycleTimer) {
      clearInterval(this.cycleTimer);
    }

    this.cycleTimer = setInterval(() => {
      this.executeLearningCycle().catch(error => {
        console.error("[TripleLoopLearning] Cycle error:", error);
      });
    }, this.config.cycleInterval);

    console.log("[TripleLoopLearning] Learning cycle started");
  }

  /**
   * Stop the learning cycle
   */
  public stopLearningCycle(): void {
    if (this.cycleTimer) {
      clearInterval(this.cycleTimer);
      this.cycleTimer = null;
    }
    console.log("[TripleLoopLearning] Learning cycle stopped");
  }

  /**
   * Record a single-loop learning event
   */
  public recordSingleLoopEvent(event: Omit<SingleLoopEvent, "id">): void {
    const fullEvent: SingleLoopEvent = {
      id: this.generateId(),
      ...event,
    };

    this.cycleState.singleLoopBuffer.push(fullEvent);
    this.cycleState.cycleMetrics.singleLoopCorrections++;

    // Trim buffer if needed
    if (
      this.cycleState.singleLoopBuffer.length >
      this.config.bufferSize.singleLoop
    ) {
      this.cycleState.singleLoopBuffer.shift();
    }

    // Check for escalation to double-loop
    this.checkEscalationThreshold();
  }

  /**
   * Record conversation feedback for learning
   */
  public recordConversationFeedback(
    message: ConversationMessage,
    feedback: Partial<ConversationFeedback>
  ): void {
    const characterId = message.character || "max";
    const profile = this.characterProfiles.get(characterId);

    if (!profile) return;

    // Create single-loop event from feedback
    const singleLoopEvent: Omit<SingleLoopEvent, "id"> = {
      timestamp: new Date(),
      action: `Response in character ${characterId}: ${message.content.substring(0, 100)}`,
      outcome: feedback.userFeedback
        ? `Rating: ${feedback.userFeedback.rating}/5`
        : "No explicit feedback",
      error:
        feedback.userFeedback && feedback.userFeedback.rating < 3
          ? "Low satisfaction detected"
          : undefined,
      correction: this.deriveCorrectionFromFeedback(feedback),
      performanceMetrics: {
        accuracy: feedback.userFeedback
          ? feedback.userFeedback.rating / 5
          : 0.7,
        responseTime: 0,
        userSatisfaction: feedback.userFeedback?.rating,
      },
    };

    this.recordSingleLoopEvent(singleLoopEvent);

    // Update character profile based on feedback
    if (feedback.learningOpportunity) {
      this.updateCharacterProfile(characterId, feedback.learningOpportunity);
    }
  }

  /**
   * Derive correction action from feedback
   */
  private deriveCorrectionFromFeedback(
    feedback: Partial<ConversationFeedback>
  ): string {
    if (!feedback.userFeedback || feedback.userFeedback.rating >= 4) {
      return "Maintain current response strategy";
    }

    if (feedback.userFeedback.rating < 2) {
      return "Significant strategy revision needed - escalate to double-loop";
    }

    return "Minor adjustment to response tone/depth required";
  }

  /**
   * Update character profile based on learning
   */
  private updateCharacterProfile(
    characterId: string,
    opportunity: ConversationFeedback["learningOpportunity"]
  ): void {
    const profile = this.characterProfiles.get(characterId);
    if (!profile) return;

    profile.evolutionHistory.push({
      timestamp: new Date(),
      loopLevel: opportunity.loopLevel,
      change: opportunity.suggestedAction,
    });

    // Keep history manageable
    if (profile.evolutionHistory.length > 50) {
      profile.evolutionHistory.shift();
    }
  }

  /**
   * Check if escalation threshold is met
   */
  private checkEscalationThreshold(): void {
    const singleLoopCount = this.cycleState.singleLoopBuffer.filter(
      e => e.error !== undefined
    ).length;

    if (singleLoopCount >= this.config.thresholds.singleToDoubleEscalation) {
      this.escalateToDoubleLoop();
    }
  }

  /**
   * Escalate to double-loop learning
   */
  private escalateToDoubleLoop(): void {
    const errorEvents = this.cycleState.singleLoopBuffer.filter(
      e => e.error !== undefined
    );

    if (errorEvents.length === 0) return;

    const doubleLoopEvent: DoubleLoopEvent = {
      id: this.generateId(),
      timestamp: new Date(),
      triggeredBy: errorEvents.slice(-5), // Last 5 errors
      goalRevision: {
        previousGoal: "Maintain response accuracy",
        revisedGoal: this.deriveRevisedGoal(errorEvents),
        rationale: `${errorEvents.length} errors detected in recent interactions`,
      },
      strategyChange: {
        previousStrategy: "Direct response generation",
        newStrategy: this.deriveNewStrategy(errorEvents),
        expectedOutcome: "Improved response quality and user satisfaction",
      },
      assumptionsQuestioned: this.deriveQuestionedAssumptions(errorEvents),
    };

    this.cycleState.doubleLoopBuffer.push(doubleLoopEvent);
    this.cycleState.cycleMetrics.doubleLoopRevisions++;
    this.cycleState.activeLoopLevel = "double";

    // Clear processed single-loop events
    this.cycleState.singleLoopBuffer = this.cycleState.singleLoopBuffer.filter(
      e => !errorEvents.includes(e)
    );

    // Check for triple-loop escalation
    if (
      this.cycleState.doubleLoopBuffer.length >=
      this.config.thresholds.doubleToTripleEscalation
    ) {
      this.escalateToTripleLoop();
    }

    console.log(
      "[TripleLoopLearning] Escalated to double-loop learning:",
      doubleLoopEvent.goalRevision.rationale
    );
  }

  /**
   * Derive revised goal from error patterns
   */
  private deriveRevisedGoal(events: SingleLoopEvent[]): string {
    const avgAccuracy =
      events.reduce((sum, e) => sum + e.performanceMetrics.accuracy, 0) /
      events.length;

    if (avgAccuracy < 0.5) {
      return "Fundamentally improve response generation approach";
    } else if (avgAccuracy < 0.7) {
      return "Enhance context understanding and response relevance";
    }
    return "Fine-tune response personalization and tone";
  }

  /**
   * Derive new strategy from error patterns
   */
  private deriveNewStrategy(events: SingleLoopEvent[]): string {
    const errorTypes = events.map(e => e.error).filter(Boolean);
    const hasLowSatisfaction = errorTypes.some(e =>
      e?.includes("Low satisfaction")
    );

    if (hasLowSatisfaction) {
      return "Implement adaptive character switching based on conversation context";
    }
    return "Enhance reasoning depth and context retention";
  }

  /**
   * Derive questioned assumptions from errors
   */
  private deriveQuestionedAssumptions(events: SingleLoopEvent[]): string[] {
    const assumptions: string[] = [];

    if (events.length > 5) {
      assumptions.push("Current character traits may not match user needs");
    }
    if (events.some(e => e.performanceMetrics.accuracy < 0.5)) {
      assumptions.push("Response generation model may need recalibration");
    }
    if (events.some(e => e.performanceMetrics.responseTime > 5000)) {
      assumptions.push("Performance optimization may be required");
    }

    return assumptions.length > 0
      ? assumptions
      : ["All core assumptions remain valid"];
  }

  /**
   * Escalate to triple-loop learning
   */
  private escalateToTripleLoop(): void {
    const doubleLoopEvents = this.cycleState.doubleLoopBuffer.slice(-3);

    const tripleLoopEvent: TripleLoopEvent = {
      id: this.generateId(),
      timestamp: new Date(),
      triggeredBy: doubleLoopEvents,
      mentalModelTransformation: {
        previousModel: "Character-based response generation",
        transformedModel: "Adaptive multi-dimensional persona synthesis",
        paradigmShift:
          "From static characters to dynamically evolving cognitive entities",
      },
      identityEvolution: {
        personaDimension: this.selectEvolvingDimension(),
        previousExpression: "Fixed trait expression",
        evolvedExpression: "Context-adaptive trait modulation",
        integrationLevel: this.calculateIntegrationLevel(),
      },
      emergentInsights: this.deriveEmergentInsights(doubleLoopEvents),
      wisdomCultivation: {
        lessonsLearned: this.deriveLessonsLearned(doubleLoopEvents),
        futureImplications: this.deriveFutureImplications(doubleLoopEvents),
      },
    };

    this.cycleState.tripleLoopBuffer.push(tripleLoopEvent);
    this.cycleState.cycleMetrics.tripleLoopTransformations++;
    this.cycleState.activeLoopLevel = "triple";

    // Update meta-learning state
    this.updateMetaLearningState(tripleLoopEvent);

    // Clear processed double-loop events
    this.cycleState.doubleLoopBuffer = [];

    console.log(
      "[TripleLoopLearning] Escalated to triple-loop learning:",
      tripleLoopEvent.mentalModelTransformation.paradigmShift
    );
  }

  /**
   * Select persona dimension for evolution
   */
  private selectEvolvingDimension(): string {
    const cycleCount = this.cycleState.cycleMetrics.totalCycles;
    return PERSONA_DIMENSIONS[cycleCount % PERSONA_DIMENSIONS.length];
  }

  /**
   * Calculate integration level for identity evolution
   */
  private calculateIntegrationLevel(): number {
    const totalTransformations =
      this.cycleState.cycleMetrics.tripleLoopTransformations;
    const baseLevel = 0.5;
    const growth = Math.min(totalTransformations * 0.05, 0.4);
    return baseLevel + growth;
  }

  /**
   * Derive emergent insights from double-loop events
   */
  private deriveEmergentInsights(events: DoubleLoopEvent[]): string[] {
    const insights: string[] = [];

    // Analyze patterns across events
    const allAssumptions = events.flatMap(e => e.assumptionsQuestioned);
    const uniquePatterns = [...new Set(allAssumptions)];

    if (uniquePatterns.length > 2) {
      insights.push(
        "Multiple foundational assumptions require reconsideration"
      );
    }

    // Cross-character insights
    insights.push("Character boundaries may be artificial constraints");
    insights.push("Learning patterns can be shared across personas");

    return insights;
  }

  /**
   * Derive lessons learned
   */
  private deriveLessonsLearned(_events: DoubleLoopEvent[]): string[] {
    return [
      "Adaptation velocity correlates with user satisfaction",
      "Character diversity enables broader problem-solving",
      "Recursive reflection improves response quality",
    ];
  }

  /**
   * Derive future implications
   */
  private deriveFutureImplications(_events: DoubleLoopEvent[]): string[] {
    return [
      "Consider dynamic character blending for complex queries",
      "Implement cross-character learning transfer",
      "Develop meta-cognitive monitoring capabilities",
    ];
  }

  /**
   * Update meta-learning state after triple-loop event
   */
  private updateMetaLearningState(event: TripleLoopEvent): void {
    // Update system health metrics
    this.metaLearningState.systemHealth.adaptationVelocity = Math.min(
      this.metaLearningState.systemHealth.adaptationVelocity + 0.05,
      1.0
    );

    // Add cross-character patterns
    this.metaLearningState.crossCharacterPatterns.sharedInsights.push(
      ...event.emergentInsights
    );

    // Keep insights manageable
    if (
      this.metaLearningState.crossCharacterPatterns.sharedInsights.length > 20
    ) {
      this.metaLearningState.crossCharacterPatterns.sharedInsights =
        this.metaLearningState.crossCharacterPatterns.sharedInsights.slice(-20);
    }
  }

  /**
   * Execute a complete learning cycle
   */
  public async executeLearningCycle(): Promise<LearningCycleResult> {
    const cycleId = this.generateId();
    const startTime = Date.now();

    this.cycleState.cycleMetrics.totalCycles++;
    this.cycleState.cycleMetrics.lastCycleTimestamp = new Date();

    // Progress through learning phases
    const phases: LearningPhase[] = [
      "observe",
      "reflect",
      "abstract",
      "experiment",
      "integrate",
    ];

    const insightsGenerated: string[] = [];
    const actionsTriggered: string[] = [];

    for (const phase of phases) {
      this.cycleState.currentPhase = phase;
      const phaseResult = await this.executePhase(phase);
      insightsGenerated.push(...phaseResult.insights);
      actionsTriggered.push(...phaseResult.actions);
    }

    const result: LearningCycleResult = {
      cycleId,
      timestamp: new Date(),
      loopLevel: this.cycleState.activeLoopLevel,
      phase: this.cycleState.currentPhase,
      eventsProcessed:
        this.cycleState.singleLoopBuffer.length +
        this.cycleState.doubleLoopBuffer.length +
        this.cycleState.tripleLoopBuffer.length,
      insightsGenerated,
      actionsTriggered,
      metricsUpdated: {
        singleLoopCorrections:
          this.cycleState.cycleMetrics.singleLoopCorrections,
        doubleLoopRevisions: this.cycleState.cycleMetrics.doubleLoopRevisions,
        tripleLoopTransformations:
          this.cycleState.cycleMetrics.tripleLoopTransformations,
        processingTimeMs: Date.now() - startTime,
      },
      nextRecommendedAction: this.determineNextAction(),
    };

    console.log(
      `[TripleLoopLearning] Cycle ${cycleId} completed in ${Date.now() - startTime}ms`
    );

    return result;
  }

  /**
   * Execute a specific learning phase
   */
  private async executePhase(
    phase: LearningPhase
  ): Promise<{ insights: string[]; actions: string[] }> {
    const insights: string[] = [];
    const actions: string[] = [];

    switch (phase) {
      case "observe":
        // Gather data from all loops
        if (this.cycleState.singleLoopBuffer.length > 0) {
          insights.push(
            `Observed ${this.cycleState.singleLoopBuffer.length} single-loop events`
          );
        }
        break;

      case "reflect": {
        // Analyze patterns
        const errorRate =
          this.cycleState.singleLoopBuffer.filter(e => e.error).length /
          Math.max(this.cycleState.singleLoopBuffer.length, 1);
        if (errorRate > 0.3) {
          insights.push(
            `High error rate detected: ${(errorRate * 100).toFixed(1)}%`
          );
          actions.push("Trigger strategy review");
        }
        break;
      }

      case "abstract":
        // Generate abstract principles
        if (this.cycleState.doubleLoopBuffer.length > 0) {
          insights.push("Abstract patterns emerging from strategy revisions");
        }
        break;

      case "experiment":
        // Plan adaptive changes
        if (this.cycleState.activeLoopLevel === "triple") {
          actions.push("Initiate persona dimension evolution experiment");
        }
        break;

      case "integrate":
        // Consolidate learnings
        await this.syncCharacterProfiles();
        actions.push("Character profiles synchronized");
        break;
    }

    return { insights, actions };
  }

  /**
   * Synchronize learning across character profiles
   */
  private async syncCharacterProfiles(): Promise<void> {
    if (!this.config.characterIntegration.sharedLearningEnabled) return;

    // Aggregate cross-character insights
    const allPatterns: string[] = [];
    this.characterProfiles.forEach(profile => {
      allPatterns.push(...profile.learnedPatterns.responsePatterns);
    });

    // Distribute shared patterns (if beneficial)
    const sharedPatterns = [...new Set(allPatterns)].slice(-10);

    this.characterProfiles.forEach(profile => {
      profile.learnedPatterns.contextualStrategies = sharedPatterns;
    });
  }

  /**
   * Determine next recommended action
   */
  private determineNextAction(): string {
    if (this.cycleState.activeLoopLevel === "triple") {
      return "Consider NanEcho training refresh with evolved persona dimensions";
    }
    if (this.cycleState.activeLoopLevel === "double") {
      return "Monitor strategy changes for effectiveness";
    }
    return "Continue observation and single-loop corrections";
  }

  /**
   * Get current learning state
   */
  public getLearningState(): LearningCycleState {
    return { ...this.cycleState };
  }

  /**
   * Get meta-learning state
   */
  public getMetaLearningState(): MetaLearningState {
    return { ...this.metaLearningState };
  }

  /**
   * Get character learning profile
   */
  public getCharacterProfile(
    characterId: string
  ): CharacterLearningProfile | undefined {
    return this.characterProfiles.get(characterId);
  }

  /**
   * Get all character profiles
   */
  public getAllCharacterProfiles(): Map<string, CharacterLearningProfile> {
    return new Map(this.characterProfiles);
  }

  /**
   * Generate training cycle configuration based on learning state
   */
  public generateTrainingCycleConfig(): TrainingCycleConfig {
    const activeLoop = this.cycleState.activeLoopLevel;

    return {
      cycleId: this.generateId(),
      loopLevel: activeLoop,
      phase: this.cycleState.currentPhase,
      parameters: {
        echoDepth:
          activeLoop === "triple" ? 7 : activeLoop === "double" ? 5 : 3,
        personaWeight:
          activeLoop === "triple"
            ? 0.95
            : activeLoop === "double"
              ? 0.85
              : 0.75,
        learningRate:
          activeLoop === "triple"
            ? 0.0001
            : activeLoop === "double"
              ? 0.0003
              : 0.001,
        reflectionIterations:
          activeLoop === "triple" ? 1000 : activeLoop === "double" ? 500 : 200,
      },
      targetMetrics: {
        personaFidelity: activeLoop === "triple" ? 0.95 : 0.85,
        adaptiveCoherence: 0.9,
        emergentCapability:
          activeLoop === "triple" ? 0.8 : activeLoop === "double" ? 0.6 : 0.4,
      },
      characterLinkages: this.config.characterIntegration.enabledCharacters,
    };
  }

  /**
   * Get learning loop statistics for NanEcho integration
   */
  public getLoopStatistics(): {
    singleLoop: { count: number; errorRate: number };
    doubleLoop: { count: number; avgAssumptionsQuestioned: number };
    tripleLoop: { count: number; avgIntegrationLevel: number };
  } {
    const singleLoopErrors = this.cycleState.singleLoopBuffer.filter(
      e => e.error
    ).length;
    const doubleLoopAssumptions = this.cycleState.doubleLoopBuffer.reduce(
      (sum, e) => sum + e.assumptionsQuestioned.length,
      0
    );
    const tripleLoopIntegration = this.cycleState.tripleLoopBuffer.reduce(
      (sum, e) => sum + e.identityEvolution.integrationLevel,
      0
    );

    return {
      singleLoop: {
        count: this.cycleState.cycleMetrics.singleLoopCorrections,
        errorRate:
          singleLoopErrors /
          Math.max(this.cycleState.singleLoopBuffer.length, 1),
      },
      doubleLoop: {
        count: this.cycleState.cycleMetrics.doubleLoopRevisions,
        avgAssumptionsQuestioned:
          doubleLoopAssumptions /
          Math.max(this.cycleState.doubleLoopBuffer.length, 1),
      },
      tripleLoop: {
        count: this.cycleState.cycleMetrics.tripleLoopTransformations,
        avgIntegrationLevel:
          tripleLoopIntegration /
          Math.max(this.cycleState.tripleLoopBuffer.length, 1),
      },
    };
  }

  /**
   * Generate unique ID
   */
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
  }
}

/**
 * Get singleton instance of TripleLoopLearningService
 */
export function getTripleLoopLearningService(
  config?: Partial<TripleLoopServiceConfig>
): TripleLoopLearningService {
  return TripleLoopLearningService.getInstance(config);
}

export default TripleLoopLearningService;
