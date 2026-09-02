/**
 * Evolutionary Configuration System
 * ==================================
 *
 * Dynamic, self-evolving configuration system for EchoSelf cognitive architecture.
 * Replaces hardcoded values with adaptive parameters that evolve based on:
 * - Performance metrics
 * - Environmental feedback
 * - Cognitive load patterns
 * - Learning trajectory analysis
 *
 * Core Principles:
 * 1. All parameters have bounds and mutation rates
 * 2. Configurations adapt through gradient-free optimization
 * 3. History tracking enables rollback and analysis
 * 4. Cross-module coordination prevents conflicting adaptations
 */

export interface EvolutionaryBounds {
  min: number;
  max: number;
  mutationRate: number; // Rate of change per evolution step
  decayFactor?: number; // Optional decay toward default
  defaultValue: number;
}

export interface EvolutionaryParameter {
  value: number;
  bounds: EvolutionaryBounds;
  history: Array<{ value: number; timestamp: number; fitness: number }>;
  lastMutation: number;
  adaptationVelocity: number; // Current direction and magnitude of change
}

export interface CognitiveConfig {
  // Attention and Salience
  attentionThreshold: EvolutionaryParameter;
  cognitiveLoadFactor: EvolutionaryParameter;
  salienceDecayRate: EvolutionaryParameter;

  // Memory Parameters
  memoryBufferSize: EvolutionaryParameter;
  memoryRetentionThreshold: EvolutionaryParameter;
  consolidationAccessThreshold: EvolutionaryParameter;
  episodicBufferLimit: EvolutionaryParameter;

  // Resonance and Patterns
  resonanceIntensityMin: EvolutionaryParameter;
  resonanceIntensityMax: EvolutionaryParameter;
  patternThreshold: EvolutionaryParameter;
  spreadingFactor: EvolutionaryParameter;

  // Learning and Adaptation
  explorationFactor: EvolutionaryParameter;
  exploitationFactor: EvolutionaryParameter;
  learningRateMultiplier: EvolutionaryParameter;

  // Agent System
  agentAttentionBudget: EvolutionaryParameter;
  agentDecayRate: EvolutionaryParameter;
  maxCopilotRequests: EvolutionaryParameter;

  // Network Parameters
  spectralRadius: EvolutionaryParameter;
  connectivity: EvolutionaryParameter;
  leakingRate: EvolutionaryParameter;
  ridgeRegularization: EvolutionaryParameter;

  // Toroidal Processing
  toroidalBufferSize: EvolutionaryParameter;
  hemisphereBalanceDefault: EvolutionaryParameter;
  responseTimeout: EvolutionaryParameter;

  // Wisdom Cultivation
  confidenceBaseline: EvolutionaryParameter;
  virtueEvaluationThreshold: EvolutionaryParameter;

  // Service Configuration
  apiPort: EvolutionaryParameter;
  retryAttempts: EvolutionaryParameter;
  requestTimeout: EvolutionaryParameter;
}

export interface EvolutionaryMetrics {
  overallFitness: number;
  cognitiveEfficiency: number;
  adaptationRate: number;
  stabilityScore: number;
  explorationExploitationBalance: number;
}

/**
 * Creates an evolutionary parameter with bounds and initial value
 */
function createParameter(
  defaultValue: number,
  min: number,
  max: number,
  mutationRate: number = 0.1,
  decayFactor?: number
): EvolutionaryParameter {
  return {
    value: defaultValue,
    bounds: {
      min,
      max,
      mutationRate,
      decayFactor,
      defaultValue,
    },
    history: [{ value: defaultValue, timestamp: Date.now(), fitness: 0.5 }],
    lastMutation: Date.now(),
    adaptationVelocity: 0,
  };
}

/**
 * Default evolutionary configuration with adaptive bounds
 */
export function createDefaultConfig(): CognitiveConfig {
  return {
    // Attention and Salience - dynamically adjust based on cognitive load
    attentionThreshold: createParameter(0.5, 0.1, 0.95, 0.05, 0.98),
    cognitiveLoadFactor: createParameter(0.3, 0.05, 0.8, 0.08),
    salienceDecayRate: createParameter(0.95, 0.8, 0.999, 0.02),

    // Memory Parameters - grow with accumulated knowledge
    memoryBufferSize: createParameter(50, 10, 500, 0.15),
    memoryRetentionThreshold: createParameter(0.6, 0.3, 0.9, 0.05),
    consolidationAccessThreshold: createParameter(3, 1, 10, 0.1),
    episodicBufferLimit: createParameter(1000, 100, 10000, 0.1),

    // Resonance and Patterns - adapt to pattern complexity
    resonanceIntensityMin: createParameter(0.7, 0.3, 0.9, 0.05),
    resonanceIntensityMax: createParameter(0.9, 0.7, 1.0, 0.03),
    patternThreshold: createParameter(0.7, 0.4, 0.95, 0.05),
    spreadingFactor: createParameter(0.1, 0.01, 0.5, 0.08),

    // Learning and Adaptation - balance exploration vs exploitation
    explorationFactor: createParameter(0.1, 0.01, 0.5, 0.1),
    exploitationFactor: createParameter(0.1, 0.01, 0.5, 0.1),
    learningRateMultiplier: createParameter(1.0, 0.1, 3.0, 0.08),

    // Agent System - scale with system complexity
    agentAttentionBudget: createParameter(1000, 100, 10000, 0.12),
    agentDecayRate: createParameter(0.95, 0.8, 0.999, 0.03),
    maxCopilotRequests: createParameter(5, 1, 20, 0.1),

    // Network Parameters - optimize for stability and expressiveness
    spectralRadius: createParameter(0.99, 0.5, 0.999, 0.02),
    connectivity: createParameter(0.1, 0.01, 0.5, 0.05),
    leakingRate: createParameter(1.0, 0.1, 1.0, 0.05),
    ridgeRegularization: createParameter(1e-6, 1e-10, 1e-3, 0.15),

    // Toroidal Processing - balance hemispheric integration
    toroidalBufferSize: createParameter(50, 10, 200, 0.1),
    hemisphereBalanceDefault: createParameter(0, -0.5, 0.5, 0.08),
    responseTimeout: createParameter(1200, 500, 5000, 0.1),

    // Wisdom Cultivation - grow confidence through experience
    confidenceBaseline: createParameter(0.5, 0.1, 0.9, 0.05, 0.99),
    virtueEvaluationThreshold: createParameter(3, 1, 10, 0.08),

    // Service Configuration - adapt to environment
    apiPort: createParameter(8765, 1024, 65535, 0.01),
    retryAttempts: createParameter(3, 1, 10, 0.1),
    requestTimeout: createParameter(30000, 5000, 120000, 0.08),
  };
}

/**
 * Evolutionary Configuration Manager
 * Central orchestrator for adaptive parameter evolution
 */
export class EvolutionaryConfigManager {
  private static instance: EvolutionaryConfigManager;
  private config: CognitiveConfig;
  private metrics: EvolutionaryMetrics;
  private evolutionHistory: Array<{
    timestamp: number;
    config: Partial<CognitiveConfig>;
    metrics: EvolutionaryMetrics;
  }> = [];
  private evolutionInterval: NodeJS.Timeout | null = null;
  private listeners: Map<string, Set<(value: number) => void>> = new Map();

  private constructor() {
    this.config = createDefaultConfig();
    this.metrics = {
      overallFitness: 0.5,
      cognitiveEfficiency: 0.5,
      adaptationRate: 0.0,
      stabilityScore: 1.0,
      explorationExploitationBalance: 0.5,
    };
    this.startEvolutionCycle();
  }

  public static getInstance(): EvolutionaryConfigManager {
    if (!EvolutionaryConfigManager.instance) {
      EvolutionaryConfigManager.instance = new EvolutionaryConfigManager();
    }
    return EvolutionaryConfigManager.instance;
  }

  /**
   * Get current value of a parameter
   */
  public get<K extends keyof CognitiveConfig>(key: K): number {
    return this.config[key].value;
  }

  /**
   * Get parameter with full evolutionary metadata
   */
  public getParameter<K extends keyof CognitiveConfig>(
    key: K
  ): EvolutionaryParameter {
    return { ...this.config[key] };
  }

  /**
   * Manually set a parameter value (within bounds)
   */
  public set<K extends keyof CognitiveConfig>(key: K, value: number): void {
    const param = this.config[key];
    const clampedValue = Math.max(
      param.bounds.min,
      Math.min(param.bounds.max, value)
    );

    param.value = clampedValue;
    param.history.push({
      value: clampedValue,
      timestamp: Date.now(),
      fitness: this.metrics.overallFitness,
    });

    // Trim history to prevent memory bloat
    if (param.history.length > 100) {
      param.history = param.history.slice(-50);
    }

    // Notify listeners
    this.notifyListeners(key as string, clampedValue);
  }

  /**
   * Register a listener for parameter changes
   */
  public subscribe<K extends keyof CognitiveConfig>(
    key: K,
    callback: (value: number) => void
  ): () => void {
    const keyStr = key as string;
    if (!this.listeners.has(keyStr)) {
      this.listeners.set(keyStr, new Set());
    }
    this.listeners.get(keyStr)!.add(callback);

    // Return unsubscribe function
    return () => {
      this.listeners.get(keyStr)?.delete(callback);
    };
  }

  private notifyListeners(key: string, value: number): void {
    this.listeners.get(key)?.forEach(callback => callback(value));
  }

  /**
   * Report fitness feedback for parameter evolution
   */
  public reportFitness(feedback: {
    cognitiveEfficiency?: number;
    responseQuality?: number;
    processingSpeed?: number;
    memoryUtilization?: number;
    errorRate?: number;
  }): void {
    // Compute weighted fitness
    const weights = {
      cognitiveEfficiency: 0.3,
      responseQuality: 0.25,
      processingSpeed: 0.2,
      memoryUtilization: 0.15,
      errorRate: 0.1,
    };

    let totalWeight = 0;
    let weightedFitness = 0;

    if (feedback.cognitiveEfficiency !== undefined) {
      weightedFitness +=
        feedback.cognitiveEfficiency * weights.cognitiveEfficiency;
      totalWeight += weights.cognitiveEfficiency;
      this.metrics.cognitiveEfficiency = feedback.cognitiveEfficiency;
    }
    if (feedback.responseQuality !== undefined) {
      weightedFitness += feedback.responseQuality * weights.responseQuality;
      totalWeight += weights.responseQuality;
    }
    if (feedback.processingSpeed !== undefined) {
      weightedFitness += feedback.processingSpeed * weights.processingSpeed;
      totalWeight += weights.processingSpeed;
    }
    if (feedback.memoryUtilization !== undefined) {
      weightedFitness += feedback.memoryUtilization * weights.memoryUtilization;
      totalWeight += weights.memoryUtilization;
    }
    if (feedback.errorRate !== undefined) {
      // Invert error rate (lower is better)
      weightedFitness += (1 - feedback.errorRate) * weights.errorRate;
      totalWeight += weights.errorRate;
    }

    if (totalWeight > 0) {
      const newFitness = weightedFitness / totalWeight;
      // Smooth update
      this.metrics.overallFitness =
        this.metrics.overallFitness * 0.8 + newFitness * 0.2;
    }
  }

  /**
   * Start the evolution cycle
   */
  private startEvolutionCycle(): void {
    // Evolution cycle runs every 5 minutes
    const EVOLUTION_INTERVAL = 5 * 60 * 1000;

    this.evolutionInterval = setInterval(() => {
      this.evolve();
    }, EVOLUTION_INTERVAL);
  }

  /**
   * Execute one evolution step across all parameters
   */
  public evolve(): void {
    const timestamp = Date.now();
    const previousFitness = this.metrics.overallFitness;

    // Determine evolution strategy based on fitness
    const strategy = this.determineEvolutionStrategy();

    // Evolve each parameter
    for (const [key, param] of Object.entries(this.config) as [
      keyof CognitiveConfig,
      EvolutionaryParameter,
    ][]) {
      this.evolveParameter(key, param, strategy);
    }

    // Update stability score
    const fitnessChange = Math.abs(
      this.metrics.overallFitness - previousFitness
    );
    this.metrics.stabilityScore = Math.max(
      0.1,
      this.metrics.stabilityScore - fitnessChange + 0.01
    );

    // Record evolution history
    this.evolutionHistory.push({
      timestamp,
      config: this.getConfigSnapshot(),
      metrics: { ...this.metrics },
    });

    // Trim history
    if (this.evolutionHistory.length > 100) {
      this.evolutionHistory = this.evolutionHistory.slice(-50);
    }

    // Update adaptation rate
    this.metrics.adaptationRate = this.calculateAdaptationRate();
  }

  private determineEvolutionStrategy(): "explore" | "exploit" | "stabilize" {
    if (this.metrics.overallFitness < 0.3) {
      return "explore"; // Low fitness - try different values
    } else if (
      this.metrics.overallFitness > 0.7 &&
      this.metrics.stabilityScore > 0.8
    ) {
      return "stabilize"; // High fitness, stable - minor adjustments
    } else {
      return "exploit"; // Moderate - refine current direction
    }
  }

  private evolveParameter(
    key: keyof CognitiveConfig,
    param: EvolutionaryParameter,
    strategy: "explore" | "exploit" | "stabilize"
  ): void {
    const { bounds } = param;
    let mutation = 0;

    switch (strategy) {
      case "explore":
        // Random mutation within bounds
        mutation = (Math.random() * 2 - 1) * bounds.mutationRate * 2;
        param.adaptationVelocity = mutation;
        break;

      case "exploit":
        // Follow current velocity with some randomness
        mutation =
          param.adaptationVelocity * 0.8 +
          (Math.random() * 2 - 1) * bounds.mutationRate * 0.5;
        param.adaptationVelocity = mutation;
        break;

      case "stabilize":
        // Small random walk, decay toward default if configured
        mutation = (Math.random() * 2 - 1) * bounds.mutationRate * 0.2;
        if (bounds.decayFactor) {
          const decayPull =
            (bounds.defaultValue - param.value) * (1 - bounds.decayFactor);
          mutation += decayPull;
        }
        param.adaptationVelocity = mutation * 0.5;
        break;
    }

    // Apply mutation
    const range = bounds.max - bounds.min;
    const newValue = Math.max(
      bounds.min,
      Math.min(bounds.max, param.value + mutation * range)
    );

    if (newValue !== param.value) {
      param.value = newValue;
      param.lastMutation = Date.now();
      param.history.push({
        value: newValue,
        timestamp: Date.now(),
        fitness: this.metrics.overallFitness,
      });

      // Notify listeners
      this.notifyListeners(key as string, newValue);
    }
  }

  private calculateAdaptationRate(): number {
    const recentHistory = this.evolutionHistory.slice(-10);
    if (recentHistory.length < 2) return 0;

    let totalChange = 0;
    for (let i = 1; i < recentHistory.length; i++) {
      const fitnessChange = Math.abs(
        recentHistory[i].metrics.overallFitness -
          recentHistory[i - 1].metrics.overallFitness
      );
      totalChange += fitnessChange;
    }

    return totalChange / (recentHistory.length - 1);
  }

  private getConfigSnapshot(): Partial<CognitiveConfig> {
    const snapshot: Partial<
      Record<keyof CognitiveConfig, EvolutionaryParameter>
    > = {};
    for (const [key, param] of Object.entries(this.config)) {
      snapshot[key as keyof CognitiveConfig] = {
        ...param,
        history: [], // Don't include full history in snapshot
      };
    }
    return snapshot as Partial<CognitiveConfig>;
  }

  /**
   * Get current metrics
   */
  public getMetrics(): EvolutionaryMetrics {
    return { ...this.metrics };
  }

  /**
   * Get all current parameter values
   */
  public getAllValues(): Record<keyof CognitiveConfig, number> {
    const values: Partial<Record<keyof CognitiveConfig, number>> = {};
    for (const [key, param] of Object.entries(this.config)) {
      values[key as keyof CognitiveConfig] = param.value;
    }
    return values as Record<keyof CognitiveConfig, number>;
  }

  /**
   * Export configuration for persistence
   */
  public exportConfig(): string {
    return JSON.stringify(
      {
        config: this.config,
        metrics: this.metrics,
        timestamp: Date.now(),
      },
      null,
      2
    );
  }

  /**
   * Import configuration from persistence
   */
  public importConfig(json: string): boolean {
    try {
      const data = JSON.parse(json);
      if (data.config) {
        // Merge with defaults to handle new parameters
        const defaultConfig = createDefaultConfig();
        for (const [key, param] of Object.entries(data.config)) {
          if (defaultConfig[key as keyof CognitiveConfig]) {
            this.config[key as keyof CognitiveConfig] =
              param as EvolutionaryParameter;
          }
        }
      }
      if (data.metrics) {
        this.metrics = data.metrics;
      }
      return true;
    } catch (e) {
      console.error("Failed to import evolutionary config:", e);
      return false;
    }
  }

  /**
   * Reset to default configuration
   */
  public reset(): void {
    this.config = createDefaultConfig();
    this.metrics = {
      overallFitness: 0.5,
      cognitiveEfficiency: 0.5,
      adaptationRate: 0.0,
      stabilityScore: 1.0,
      explorationExploitationBalance: 0.5,
    };
    this.evolutionHistory = [];
  }

  /**
   * Cleanup on shutdown
   */
  public shutdown(): void {
    if (this.evolutionInterval) {
      clearInterval(this.evolutionInterval);
      this.evolutionInterval = null;
    }
  }
}

// Export singleton getter
export const getEvolutionaryConfig = () =>
  EvolutionaryConfigManager.getInstance();

// Export convenience functions for common operations
export const getConfig = <K extends keyof CognitiveConfig>(key: K): number => {
  return EvolutionaryConfigManager.getInstance().get(key);
};

export const setConfig = <K extends keyof CognitiveConfig>(
  key: K,
  value: number
): void => {
  EvolutionaryConfigManager.getInstance().set(key, value);
};

export const reportFitness = (
  feedback: Parameters<EvolutionaryConfigManager["reportFitness"]>[0]
): void => {
  EvolutionaryConfigManager.getInstance().reportFitness(feedback);
};

export default EvolutionaryConfigManager;
