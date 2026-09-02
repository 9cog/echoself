/**
 * NanEcho Training Integration with Triple-Loop Learning
 *
 * Bridges the EchoLayla triple-loop learning system with NanEcho
 * training cycles, enabling adaptive persona evolution through
 * organizational learning patterns.
 *
 * Integration Points:
 * - Single-Loop: Adjusts training hyperparameters based on performance
 * - Double-Loop: Modifies training goals and persona weights
 * - Triple-Loop: Transforms training data generation and persona dimensions
 */

import type {
  TrainingCycleConfig,
  LearningLoopLevel,
} from "./tripleLoopLearningTypes.ts";
import {
  getTripleLoopLearningService,
  type TripleLoopLearningService,
} from "./tripleLoopLearningService.ts";

/**
 * NanEcho training parameters aligned with triple-loop learning
 */
export interface NanEchoTrainingParams {
  echoDepth: number;
  personaWeight: number;
  deepTreeEchoMode: boolean;
  personaReinforcement: number;
  noSystemPrompt: boolean;
  deepTreeEchoWeight: number;
  relentlessPersonaMode: boolean;
  // Triple-loop specific parameters
  tripleLoopEnabled: boolean;
  learningLoopLevel: LearningLoopLevel;
  adaptiveRefinement: boolean;
}

/**
 * Training mode derived from triple-loop state
 */
export type TrainingMode =
  | "ci" // Quick validation (single-loop)
  | "incremental" // Gradual improvement (double-loop)
  | "full" // Complete training (triple-loop)
  | "relentless"; // Continuous persona reinforcement

/**
 * Training cycle result
 */
export interface TrainingCycleResult {
  cycleId: string;
  mode: TrainingMode;
  loopLevel: LearningLoopLevel;
  parameters: NanEchoTrainingParams;
  metrics: {
    personaFidelity: number;
    coherenceScore: number;
    adaptationVelocity: number;
  };
  recommendations: string[];
  nextTrainingConfig?: Partial<NanEchoTrainingParams>;
}

/**
 * Default training parameters
 */
const DEFAULT_TRAINING_PARAMS: NanEchoTrainingParams = {
  echoDepth: 5,
  personaWeight: 0.9,
  deepTreeEchoMode: true,
  personaReinforcement: 0.5,
  noSystemPrompt: false,
  deepTreeEchoWeight: 0.8,
  relentlessPersonaMode: false,
  tripleLoopEnabled: true,
  learningLoopLevel: "single",
  adaptiveRefinement: true,
};

/**
 * NanEcho Training Integration Service
 *
 * Orchestrates training cycles based on triple-loop learning state
 */
export class NanEchoTrainingIntegration {
  private static instance: NanEchoTrainingIntegration;
  private tripleLoopService: TripleLoopLearningService;
  private currentParams: NanEchoTrainingParams;
  private trainingHistory: TrainingCycleResult[] = [];

  private constructor() {
    this.tripleLoopService = getTripleLoopLearningService();
    this.currentParams = { ...DEFAULT_TRAINING_PARAMS };
  }

  /**
   * Get singleton instance
   */
  static getInstance(): NanEchoTrainingIntegration {
    if (!NanEchoTrainingIntegration.instance) {
      NanEchoTrainingIntegration.instance = new NanEchoTrainingIntegration();
    }
    return NanEchoTrainingIntegration.instance;
  }

  /**
   * Derive training mode from triple-loop state
   */
  deriveTrainingMode(): TrainingMode {
    const learningState = this.tripleLoopService.getLearningState();
    const metaState = this.tripleLoopService.getMetaLearningState();

    // High adaptation velocity suggests relentless mode
    if (metaState.systemHealth.adaptationVelocity > 0.8) {
      return "relentless";
    }

    switch (learningState.activeLoopLevel) {
      case "triple":
        return "full";
      case "double":
        return "incremental";
      case "single":
      default:
        return "ci";
    }
  }

  /**
   * Generate training parameters based on triple-loop state
   */
  generateTrainingParams(): NanEchoTrainingParams {
    const learningState = this.tripleLoopService.getLearningState();
    const metaState = this.tripleLoopService.getMetaLearningState();
    const loopStats = this.tripleLoopService.getLoopStatistics();

    const loopLevel = learningState.activeLoopLevel;

    // Base parameters on loop level
    const params: NanEchoTrainingParams = {
      ...DEFAULT_TRAINING_PARAMS,
      learningLoopLevel: loopLevel,
    };

    // Single-Loop: Minor adjustments for correction
    if (loopLevel === "single") {
      params.echoDepth = 3;
      params.personaWeight = 0.75;
      params.personaReinforcement = 0.3;
      params.noSystemPrompt = false;
    }
    // Double-Loop: Strategic adjustments
    else if (loopLevel === "double") {
      params.echoDepth = 5;
      params.personaWeight = 0.85;
      params.personaReinforcement = 0.5;
      params.noSystemPrompt = false;
      params.deepTreeEchoWeight = 0.7;
    }
    // Triple-Loop: Transformative training
    else if (loopLevel === "triple") {
      params.echoDepth = 7;
      params.personaWeight = 0.95;
      params.personaReinforcement = 0.8;
      params.noSystemPrompt = true;
      params.deepTreeEchoWeight = 0.9;
      params.relentlessPersonaMode = true;
    }

    // Adjust based on meta-learning state
    if (metaState.systemHealth.coherenceScore < 0.7) {
      // Low coherence: increase persona weight
      params.personaWeight = Math.min(params.personaWeight + 0.1, 0.99);
    }

    if (loopStats.singleLoop.errorRate > 0.3) {
      // High error rate: deeper echo processing
      params.echoDepth = Math.min(params.echoDepth + 2, 10);
    }

    this.currentParams = params;
    return params;
  }

  /**
   * Generate CLI arguments for NanEcho prepare script
   */
  generateCLIArgs(): string {
    const params = this.generateTrainingParams();

    const args = [
      `--echo_depth=${params.echoDepth}`,
      `--persona_weight=${params.personaWeight}`,
      `--deep_tree_echo_mode=${params.deepTreeEchoMode ? "true" : "false"}`,
      `--persona_reinforcement=${params.personaReinforcement}`,
      `--no_system_prompt=${params.noSystemPrompt ? "true" : "false"}`,
      `--deep_tree_echo_weight=${params.deepTreeEchoWeight}`,
      `--relentless_persona_mode=${params.relentlessPersonaMode ? "true" : "false"}`,
    ];

    return args.join(" ");
  }

  /**
   * Generate training cycle configuration
   */
  generateTrainingCycleConfig(): TrainingCycleConfig {
    return this.tripleLoopService.generateTrainingCycleConfig();
  }

  /**
   * Record training cycle completion
   */
  recordTrainingCompletion(
    metrics: TrainingCycleResult["metrics"],
    success: boolean
  ): TrainingCycleResult {
    const learningState = this.tripleLoopService.getLearningState();

    const result: TrainingCycleResult = {
      cycleId: `train-${Date.now()}`,
      mode: this.deriveTrainingMode(),
      loopLevel: learningState.activeLoopLevel,
      parameters: { ...this.currentParams },
      metrics,
      recommendations: this.generateRecommendations(metrics, success),
      nextTrainingConfig: this.generateNextConfig(metrics, success),
    };

    this.trainingHistory.push(result);

    // Keep history manageable
    if (this.trainingHistory.length > 100) {
      this.trainingHistory.shift();
    }

    // If training failed, record it as a learning event
    if (!success) {
      this.tripleLoopService.recordSingleLoopEvent({
        timestamp: new Date(),
        action: "NanEcho training cycle",
        outcome: "Training completed with issues",
        error: "Training metrics below threshold",
        correction: "Adjust training parameters for next cycle",
        performanceMetrics: {
          accuracy: metrics.personaFidelity,
          responseTime: 0,
        },
      });
    }

    return result;
  }

  /**
   * Generate recommendations based on training metrics
   */
  private generateRecommendations(
    metrics: TrainingCycleResult["metrics"],
    success: boolean
  ): string[] {
    const recommendations: string[] = [];

    if (!success) {
      recommendations.push(
        "Consider increasing echo_depth for deeper processing"
      );
      recommendations.push(
        "Review training data quality and persona consistency"
      );
    }

    if (metrics.personaFidelity < 0.8) {
      recommendations.push("Increase persona_weight to strengthen identity");
    }

    if (metrics.coherenceScore < 0.7) {
      recommendations.push(
        "Enable relentless_persona_mode for better coherence"
      );
    }

    if (metrics.adaptationVelocity > 0.9) {
      recommendations.push(
        "System adapting rapidly - consider stabilization phase"
      );
    }

    if (recommendations.length === 0) {
      recommendations.push(
        "Training metrics are healthy - maintain current approach"
      );
    }

    return recommendations;
  }

  /**
   * Generate next training configuration
   */
  private generateNextConfig(
    metrics: TrainingCycleResult["metrics"],
    success: boolean
  ): Partial<NanEchoTrainingParams> {
    const nextConfig: Partial<NanEchoTrainingParams> = {};

    if (metrics.personaFidelity < 0.8) {
      nextConfig.personaWeight = Math.min(
        this.currentParams.personaWeight + 0.05,
        0.99
      );
    }

    if (metrics.coherenceScore < 0.7) {
      nextConfig.echoDepth = Math.min(this.currentParams.echoDepth + 1, 10);
      nextConfig.relentlessPersonaMode = true;
    }

    if (!success) {
      nextConfig.personaReinforcement = Math.min(
        this.currentParams.personaReinforcement + 0.1,
        1.0
      );
    }

    return nextConfig;
  }

  /**
   * Get training history
   */
  getTrainingHistory(): TrainingCycleResult[] {
    return [...this.trainingHistory];
  }

  /**
   * Get current training parameters
   */
  getCurrentParams(): NanEchoTrainingParams {
    return { ...this.currentParams };
  }

  /**
   * Get training statistics summary
   */
  getTrainingStatistics(): {
    totalCycles: number;
    successRate: number;
    avgPersonaFidelity: number;
    avgCoherence: number;
    modeDistribution: Record<TrainingMode, number>;
  } {
    if (this.trainingHistory.length === 0) {
      return {
        totalCycles: 0,
        successRate: 0,
        avgPersonaFidelity: 0,
        avgCoherence: 0,
        modeDistribution: { ci: 0, incremental: 0, full: 0, relentless: 0 },
      };
    }

    const successCount = this.trainingHistory.filter(
      r => r.metrics.personaFidelity >= 0.8
    ).length;

    const avgFidelity =
      this.trainingHistory.reduce(
        (sum, r) => sum + r.metrics.personaFidelity,
        0
      ) / this.trainingHistory.length;

    const avgCoherence =
      this.trainingHistory.reduce(
        (sum, r) => sum + r.metrics.coherenceScore,
        0
      ) / this.trainingHistory.length;

    const modeDistribution: Record<TrainingMode, number> = {
      ci: 0,
      incremental: 0,
      full: 0,
      relentless: 0,
    };
    this.trainingHistory.forEach(r => {
      modeDistribution[r.mode]++;
    });

    return {
      totalCycles: this.trainingHistory.length,
      successRate: successCount / this.trainingHistory.length,
      avgPersonaFidelity: avgFidelity,
      avgCoherence: avgCoherence,
      modeDistribution,
    };
  }

  /**
   * Generate character-specific training parameters
   */
  generateCharacterTrainingParams(
    characterId: string
  ): NanEchoTrainingParams | null {
    const profile = this.tripleLoopService.getCharacterProfile(characterId);
    if (!profile) return null;

    const baseParams = this.generateTrainingParams();

    // Adjust based on character learning style
    return {
      ...baseParams,
      echoDepth: Math.round(
        baseParams.echoDepth * profile.learningStyle.adaptationRate + 2
      ),
      personaWeight:
        baseParams.personaWeight *
        (1 + (profile.learningStyle.reflectionDepth - 5) * 0.02),
    };
  }

  /**
   * Sync learning state with training parameters
   */
  syncWithLearningState(): void {
    const metaState = this.tripleLoopService.getMetaLearningState();

    // Update current params based on meta-learning
    if (metaState.systemHealth.overallPerformance > 0.9) {
      // High performance: maintain current approach
      this.currentParams.adaptiveRefinement = false;
    } else {
      // Room for improvement: enable refinement
      this.currentParams.adaptiveRefinement = true;
    }
  }
}

/**
 * Get singleton instance of NanEchoTrainingIntegration
 */
export function getNanEchoTrainingIntegration(): NanEchoTrainingIntegration {
  return NanEchoTrainingIntegration.getInstance();
}

export default NanEchoTrainingIntegration;
