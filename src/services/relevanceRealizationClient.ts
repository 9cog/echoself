/**
 * Relevance Realization Client for TypeScript
 * =============================================
 *
 * TypeScript client for interacting with the Python RelevanceRealizationEngine
 * via the REST API bridge. Integrates with AdaptiveFeedbackService.
 *
 * @author Deep Tree Echo
 * @date June 2026
 */

import { EventEmitter } from "events";

// Types matching the Python API
export interface RelevanceCriteria {
  goal_alignment: number;
  predictive_power: number;
  cognitive_economy: number;
  novelty_value: number;
  contextual_fit: number;
  score: number;
}

export interface Possibility {
  id: string;
  data: Record<string, unknown>;
  criteria?: Partial<RelevanceCriteria>;
  constraints_satisfied?: boolean;
  future_relevance?: number;
}

export interface RelevanceResult {
  possibilities: Possibility[];
  filtered_count: number;
  original_count: number;
  opponent_states: OpponentStates;
  processing_time_ms: number;
}

export interface OpponentStates {
  exploration_exploitation: number;
  breadth_depth: number;
  speed_accuracy: number;
  certainty_openness: number;
}

export interface RelevanceContext {
  goals?: Array<{ id: string; description: string; priority: number }>;
  resources?: { cognitive_load: number; time_pressure: number };
  cognitive_load?: number;
  novelty_needed?: boolean;
  precision_needed?: boolean;
  [key: string]: unknown;
}

export interface FeedbackOutcome {
  success: boolean;
  quality?: number;
  error?: string;
  [key: string]: unknown;
}

export interface EngineState {
  opponent_states: OpponentStates;
  context: Record<string, unknown>;
  history: {
    relevance_history_size: number;
    processing_history_size: number;
    outcome_history_size: number;
  };
  cost_functions: string[];
  statistics: {
    request_count: number;
    total_processing_time_ms: number;
  };
}

/**
 * Client for the Python Relevance Realization Engine
 */
export class RelevanceRealizationClient extends EventEmitter {
  private baseUrl: string;
  private timeout: number;
  private lastRequestTime: number = 0;
  private requestQueue: Array<() => Promise<void>> = [];
  private processing: boolean = false;

  constructor(
    baseUrl: string = "http://localhost:8766",
    timeout: number = 30000
  ) {
    super();
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeout = timeout;
  }

  /**
   * Make HTTP request to the RR engine
   */
  private async request<T>(
    method: "GET" | "POST",
    path: string,
    body?: Record<string, unknown>
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const options: RequestInit = {
      method,
      headers: {
        "Content-Type": "application/json",
      },
      signal: AbortSignal.timeout(this.timeout),
    };

    if (body) {
      options.body = JSON.stringify(body);
    }

    this.lastRequestTime = Date.now();
    this.emit("request:start", { method, path });

    try {
      const response = await fetch(url, options);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          `HTTP ${response.status}: ${errorData.error || response.statusText}`
        );
      }

      const data = await response.json();
      this.emit("request:complete", { method, path, data });
      return data as T;
    } catch (error) {
      this.emit("request:error", { method, path, error });
      throw error;
    }
  }

  // ================== CORE OPERATIONS ==================

  /**
   * Check if the RR engine is healthy and available
   */
  async isHealthy(): Promise<boolean> {
    try {
      const health = await this.request<{ status: string }>("GET", "/health");
      return health.status === "healthy";
    } catch {
      return false;
    }
  }

  /**
   * Core relevance realization - filter and prioritize possibilities
   */
  async realizeRelevance(
    possibilities: Possibility[],
    context?: RelevanceContext
  ): Promise<RelevanceResult> {
    const result = await this.request<RelevanceResult>("POST", "/realize", {
      possibilities,
      context,
    });

    this.emit("relevance:realized", {
      originalCount: result.original_count,
      filteredCount: result.filtered_count,
      processingTime: result.processing_time_ms,
    });

    return result;
  }

  /**
   * Provide feedback on relevance decisions for learning
   */
  async provideFeedback(
    chosen: Possibility[],
    outcomes: FeedbackOutcome[]
  ): Promise<{
    status: string;
    opponent_states: OpponentStates;
    history_size: number;
  }> {
    const result = await this.request<{
      status: string;
      opponent_states: OpponentStates;
      history_size: number;
    }>("POST", "/feedback", { chosen, outcomes });

    this.emit("feedback:provided", { count: outcomes.length });

    return result;
  }

  /**
   * Set the current context for relevance realization
   */
  async setContext(context: RelevanceContext): Promise<{
    status: string;
    context_keys: string[];
    goals_count: number;
  }> {
    return this.request("POST", "/context", { context });
  }

  // ================== OPPONENT PROCESS CONTROL ==================

  /**
   * Get current opponent process states
   */
  async getOpponentStates(): Promise<OpponentStates> {
    return this.request<OpponentStates>("GET", "/opponents");
  }

  /**
   * Adjust an opponent process balance
   */
  async adjustOpponentProcess(
    process: keyof OpponentStates,
    delta: number
  ): Promise<{
    process: string;
    old_balance: number;
    new_balance: number;
    delta: number;
  }> {
    return this.request("POST", "/opponent/adjust", { process, delta });
  }

  /**
   * Shift toward exploration (more novelty seeking)
   */
  async shiftTowardExploration(amount: number = 0.1): Promise<void> {
    await this.adjustOpponentProcess("exploration_exploitation", -amount);
  }

  /**
   * Shift toward exploitation (more focused)
   */
  async shiftTowardExploitation(amount: number = 0.1): Promise<void> {
    await this.adjustOpponentProcess("exploration_exploitation", amount);
  }

  /**
   * Shift toward breadth (more coverage)
   */
  async shiftTowardBreadth(amount: number = 0.1): Promise<void> {
    await this.adjustOpponentProcess("breadth_depth", -amount);
  }

  /**
   * Shift toward depth (more thorough)
   */
  async shiftTowardDepth(amount: number = 0.1): Promise<void> {
    await this.adjustOpponentProcess("breadth_depth", amount);
  }

  // ================== STATE & DIAGNOSTICS ==================

  /**
   * Get full engine state
   */
  async getState(): Promise<EngineState> {
    return this.request<EngineState>("GET", "/state");
  }

  /**
   * Get statistics about engine usage
   */
  async getStatistics(): Promise<{
    request_count: number;
    total_processing_time_ms: number;
    avg_processing_time_ms: number;
  }> {
    const state = await this.getState();
    const stats = state.statistics;

    return {
      ...stats,
      avg_processing_time_ms:
        stats.request_count > 0
          ? stats.total_processing_time_ms / stats.request_count
          : 0,
    };
  }
}

/**
 * Integration with AdaptiveFeedbackService
 */
export interface AdaptiveFeedbackServiceIntegration {
  relevanceClient: RelevanceRealizationClient;
  processSalientModels(models: Possibility[]): Promise<RelevanceResult>;
  feedbackLoop(
    models: Possibility[],
    context: RelevanceContext
  ): Promise<RelevanceResult>;
}

/**
 * Create integrated service with AdaptiveFeedbackService
 */
export function createAdaptiveFeedbackIntegration(
  rrClient: RelevanceRealizationClient
): AdaptiveFeedbackServiceIntegration {
  return {
    relevanceClient: rrClient,

    async processSalientModels(
      models: Possibility[]
    ): Promise<RelevanceResult> {
      // Use RR engine for intelligent filtering
      return rrClient.realizeRelevance(models);
    },

    async feedbackLoop(
      models: Possibility[],
      context: RelevanceContext
    ): Promise<RelevanceResult> {
      // Set context first
      await rrClient.setContext(context);

      // Realize relevance
      const result = await rrClient.realizeRelevance(models, context);

      // Emit event for the feedback loop
      rrClient.emit("feedbackLoop:complete", {
        inputCount: models.length,
        outputCount: result.filtered_count,
        context,
      });

      return result;
    },
  };
}

// Singleton instance
let clientInstance: RelevanceRealizationClient | null = null;

/**
 * Get or create the singleton client
 */
export function getRelevanceClient(
  baseUrl?: string
): RelevanceRealizationClient {
  if (!clientInstance) {
    clientInstance = new RelevanceRealizationClient(baseUrl);
  }
  return clientInstance;
}

/**
 * Reset the singleton client
 */
export function resetRelevanceClient(): void {
  clientInstance = null;
}

export default RelevanceRealizationClient;
