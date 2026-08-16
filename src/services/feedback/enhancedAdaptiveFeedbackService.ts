/**
 * Enhanced Adaptive Feedback Service with Relevance Realization Integration
 * ==========================================================================
 *
 * Extends AdaptiveFeedbackService with integration to the Python
 * RelevanceRealizationEngine for principled relevance optimization.
 *
 * This implements the IMPLEMENTATION_GUIDE.md Phase 1 integration:
 * - Python-TypeScript bridge for RR engine
 * - RR engine integration with adaptive feedback
 * - Opponent process management
 * - Circular causality tracking
 *
 * @author Deep Tree Echo
 * @date June 2026
 */

import {
  AdaptiveFeedbackService,
  ProjectModel,
} from "./adaptiveFeedbackService.ts";
import {
  RelevanceRealizationClient,
  Possibility,
  RelevanceContext,
  OpponentStates,
  EngineState,
} from "../relevanceRealizationClient.ts";

// Re-export types
export type { RelevanceContext, OpponentStates };

export interface EnhancedModelScore {
  model: ProjectModel;
  rrScore: number;
  criteria: {
    goal_alignment: number;
    predictive_power: number;
    cognitive_economy: number;
    novelty_value: number;
    contextual_fit: number;
  };
  future_relevance: number;
}

export interface FeedbackCycleResult {
  timestamp: Date;
  originalCount: number;
  filteredCount: number;
  processingTimeMs: number;
  opponentStates: OpponentStates;
  topModels: EnhancedModelScore[];
}

/**
 * Enhanced Adaptive Feedback Service with Relevance Realization
 */
export class EnhancedAdaptiveFeedbackService {
  private static instance: EnhancedAdaptiveFeedbackService;
  private baseService: AdaptiveFeedbackService;
  private rrClient: RelevanceRealizationClient;
  private rrAvailable: boolean = false;
  private lastResult: FeedbackCycleResult | null = null;
  private cycleHistory: FeedbackCycleResult[] = [];

  private constructor() {
    this.baseService = AdaptiveFeedbackService.getInstance();
    this.rrClient = new RelevanceRealizationClient();

    this.initializeRRConnection();
  }

  public static getInstance(): EnhancedAdaptiveFeedbackService {
    if (!EnhancedAdaptiveFeedbackService.instance) {
      EnhancedAdaptiveFeedbackService.instance =
        new EnhancedAdaptiveFeedbackService();
    }
    return EnhancedAdaptiveFeedbackService.instance;
  }

  /**
   * Initialize connection to RR engine
   */
  private async initializeRRConnection(): Promise<void> {
    try {
      this.rrAvailable = await this.rrClient.isHealthy();
      if (this.rrAvailable) {
        console.log("✅ Relevance Realization Engine connected");

        // Set up event listeners
        this.rrClient.on("relevance:realized", data => {
          console.log(
            `🎯 RR: ${data.filteredCount}/${data.originalCount} relevant ` +
              `(${data.processingTime.toFixed(1)}ms)`
          );
        });

        this.rrClient.on("feedback:provided", data => {
          console.log(`📊 RR: Feedback provided for ${data.count} items`);
        });
      } else {
        console.warn("⚠️ RR Engine not available, using fallback salience");
      }
    } catch (error) {
      console.warn("⚠️ Failed to connect to RR Engine:", error);
      this.rrAvailable = false;
    }
  }

  /**
   * Check RR availability
   */
  public async checkRRAvailability(): Promise<boolean> {
    this.rrAvailable = await this.rrClient.isHealthy();
    return this.rrAvailable;
  }

  /**
   * Enhanced feedback loop with RR integration
   */
  public async executeFeedbackCycle(): Promise<FeedbackCycleResult> {
    console.log("🔄 Starting enhanced feedback cycle with RR integration...");

    // Collect models using base service mechanisms
    const allModels = await this.collectAllPossibleModels();

    // Build context from current system state
    const context = await this.buildRelevanceContext();

    let result: FeedbackCycleResult;

    if (this.rrAvailable) {
      // Use RR engine for principled relevance optimization
      result = await this.processWithRREngine(allModels, context);
    } else {
      // Fallback to heuristic-based salience
      result = await this.processWithFallback(allModels, context);
    }

    result.timestamp = new Date();
    this.lastResult = result;
    this.cycleHistory.push(result);

    // Keep only last 100 results
    if (this.cycleHistory.length > 100) {
      this.cycleHistory.shift();
    }

    console.log(
      `✅ Enhanced feedback cycle complete: ${result.filteredCount}/${result.originalCount} ` +
        `relevant (${result.processingTimeMs.toFixed(1)}ms)`
    );

    return result;
  }

  /**
   * Process models using RR engine
   */
  private async processWithRREngine(
    models: ProjectModel[],
    context: RelevanceContext
  ): Promise<FeedbackCycleResult> {
    // Convert ProjectModels to Possibilities
    const possibilities: Possibility[] = models.map(model => ({
      id: model.id,
      data: {
        name: model.name,
        description: model.description,
        version: model.version,
        usageCount: model.usageCount,
        feedbackCount: model.communityFeedback?.length || 0,
        salienceScore: model.salienceScore,
      },
    }));

    // Set context on RR engine
    await this.rrClient.setContext(context);

    // Realize relevance
    const rrResult = await this.rrClient.realizeRelevance(
      possibilities,
      context
    );

    // Convert back to enhanced scores
    const topModels: EnhancedModelScore[] = rrResult.possibilities.map(p => {
      const originalModel = models.find(m => m.id === p.id);
      return {
        model: originalModel!,
        rrScore: p.criteria?.score || 0,
        criteria: {
          goal_alignment: p.criteria?.goal_alignment || 0,
          predictive_power: p.criteria?.predictive_power || 0,
          cognitive_economy: p.criteria?.cognitive_economy || 0,
          novelty_value: p.criteria?.novelty_value || 0,
          contextual_fit: p.criteria?.contextual_fit || 0,
        },
        future_relevance: p.future_relevance || 0,
      };
    });

    return {
      timestamp: new Date(),
      originalCount: rrResult.original_count,
      filteredCount: rrResult.filtered_count,
      processingTimeMs: rrResult.processing_time_ms,
      opponentStates: rrResult.opponent_states,
      topModels,
    };
  }

  /**
   * Fallback processing without RR engine
   */
  private async processWithFallback(
    models: ProjectModel[],
    _context: RelevanceContext
  ): Promise<FeedbackCycleResult> {
    const startTime = Date.now();

    // Simple heuristic scoring
    const scored = models.map(model => ({
      model,
      rrScore: this.calculateFallbackScore(model),
      criteria: {
        goal_alignment: model.salienceScore * 0.3,
        predictive_power: 0.5,
        cognitive_economy: Math.min(1, 1 / (1 + model.usageCount * 0.1)),
        novelty_value: model.salienceScore > 0.5 ? 0.7 : 0.3,
        contextual_fit: 0.5,
      },
      future_relevance: model.salienceScore * 0.8,
    }));

    // Sort by score and take top items
    scored.sort((a, b) => b.rrScore - a.rrScore);
    const topModels = scored.slice(0, 20);

    return {
      timestamp: new Date(),
      originalCount: models.length,
      filteredCount: topModels.length,
      processingTimeMs: Date.now() - startTime,
      opponentStates: {
        exploration_exploitation: 0.5,
        breadth_depth: 0.5,
        speed_accuracy: 0.5,
        certainty_openness: 0.6,
      },
      topModels,
    };
  }

  /**
   * Calculate fallback salience score
   */
  private calculateFallbackScore(model: ProjectModel): number {
    const feedbackWeight = Math.min(
      (model.communityFeedback?.length || 0) * 0.1,
      0.3
    );
    const usageWeight = Math.min(model.usageCount * 0.05, 0.2);
    return model.salienceScore * 0.5 + feedbackWeight + usageWeight;
  }

  /**
   * Collect all possible models for processing
   */
  private async collectAllPossibleModels(): Promise<ProjectModel[]> {
    // Get from base service
    const salientModels = await (
      this.baseService as any
    ).collectSalientModels();
    return salientModels || [];
  }

  /**
   * Build relevance context from current system state
   */
  private async buildRelevanceContext(): Promise<RelevanceContext> {
    const thresholds = (this.baseService as any).adaptiveThresholds;

    return {
      goals: [
        {
          id: "improve_models",
          description: "Improve model quality",
          priority: 0.9,
        },
        {
          id: "integrate_feedback",
          description: "Address community feedback",
          priority: 0.8,
        },
        {
          id: "optimize_performance",
          description: "Optimize system performance",
          priority: 0.7,
        },
      ],
      resources: {
        cognitive_load: thresholds?.cognitiveLoad || 0.5,
        time_pressure: 0.3,
      },
      cognitive_load: thresholds?.cognitiveLoad || 0.5,
      novelty_needed: Math.random() > 0.5, // Could be based on exploration phase
      precision_needed: Math.random() > 0.7,
    };
  }

  // ================== OPPONENT PROCESS CONTROL ==================

  /**
   * Get current opponent process states
   */
  public async getOpponentStates(): Promise<OpponentStates | null> {
    if (!this.rrAvailable) {
      return null;
    }
    return this.rrClient.getOpponentStates();
  }

  /**
   * Shift toward exploration mode
   */
  public async shiftTowardExploration(amount: number = 0.1): Promise<void> {
    if (this.rrAvailable) {
      await this.rrClient.shiftTowardExploration(amount);
    }
  }

  /**
   * Shift toward exploitation mode
   */
  public async shiftTowardExploitation(amount: number = 0.1): Promise<void> {
    if (this.rrAvailable) {
      await this.rrClient.shiftTowardExploitation(amount);
    }
  }

  /**
   * Shift toward breadth (broader coverage)
   */
  public async shiftTowardBreadth(amount: number = 0.1): Promise<void> {
    if (this.rrAvailable) {
      await this.rrClient.shiftTowardBreadth(amount);
    }
  }

  /**
   * Shift toward depth (more thorough)
   */
  public async shiftTowardDepth(amount: number = 0.1): Promise<void> {
    if (this.rrAvailable) {
      await this.rrClient.shiftTowardDepth(amount);
    }
  }

  // ================== FEEDBACK & LEARNING ==================

  /**
   * Provide feedback on processing outcomes for learning
   */
  public async provideFeedback(
    processedModels: EnhancedModelScore[],
    outcomes: Array<{ success: boolean; quality?: number }>
  ): Promise<void> {
    if (!this.rrAvailable) {
      return;
    }

    const chosen = processedModels.map(m => ({
      id: m.model.id,
      data: m.model as unknown as Record<string, unknown>,
      criteria: m.criteria,
    }));

    await this.rrClient.provideFeedback(chosen, outcomes);
  }

  // ================== STATE & DIAGNOSTICS ==================

  /**
   * Get last feedback cycle result
   */
  public getLastResult(): FeedbackCycleResult | null {
    return this.lastResult;
  }

  /**
   * Get cycle history
   */
  public getCycleHistory(): FeedbackCycleResult[] {
    return [...this.cycleHistory];
  }

  /**
   * Get RR engine state
   */
  public async getRRState(): Promise<EngineState | null> {
    if (!this.rrAvailable) {
      return null;
    }
    return this.rrClient.getState();
  }

  /**
   * Get comprehensive service state
   */
  public async getState(): Promise<{
    rrAvailable: boolean;
    lastCycleTimestamp: Date | null;
    cycleHistoryLength: number;
    opponentStates: OpponentStates | null;
    rrEngineState: EngineState | null;
  }> {
    return {
      rrAvailable: this.rrAvailable,
      lastCycleTimestamp: this.lastResult?.timestamp || null,
      cycleHistoryLength: this.cycleHistory.length,
      opponentStates: await this.getOpponentStates(),
      rrEngineState: await this.getRRState(),
    };
  }
}

/**
 * Get singleton instance
 */
export function getEnhancedAdaptiveFeedbackService(): EnhancedAdaptiveFeedbackService {
  return EnhancedAdaptiveFeedbackService.getInstance();
}

export default EnhancedAdaptiveFeedbackService;
