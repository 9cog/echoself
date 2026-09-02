/**
 * Echogenesis Service
 * ===================
 *
 * TypeScript service for interacting with the Python echogenesis
 * backend through REST API.
 *
 * Provides methods for:
 * - Adaptive dimensional embedding
 * - Optimal cognitive grip
 * - Perspectival knowing
 * - Wisdom cultivation
 * - Full cognitive cycles
 *
 * @author Deep Tree Echo
 * @date June 2026
 */

import { EventEmitter } from "events";
import { getConfig, getEvolutionaryConfig } from "./evolutionaryConfig.ts";

// Configuration - dynamic defaults from evolutionary config
const DEFAULT_API_HOST = "localhost";

// Types
export interface TruthValue {
  strength: number;
  confidence: number;
}

export interface AttentionValue {
  sti: number;
  lti: number;
  vlti?: boolean;
}

export interface EmbeddingResult {
  projected: number[];
  state: EmbeddingState;
}

export interface EmbeddingState {
  current_effective_dim: number;
  projection_mode: string;
  cognitive_load: number;
  attention_threshold: number;
}

export interface MultiScaleEmbedding {
  local?: number[];
  context?: number[];
  global?: number[];
}

export interface EmbodimentManifold {
  manifold: number[];
  dimension: number;
}

export interface RelevanceResult {
  ranked: Array<{
    possibility: any;
    relevance_score: number;
    filtered: boolean;
  }>;
  grip_quality: number;
  state: GripState;
}

export interface GripState {
  optimal_grip: number;
  grip_quality: number;
  opponent_balances: Record<string, number>;
  cost_breakdown: Record<string, number>;
}

export interface FrameState {
  current_frame: string | null;
  available_frames: string[];
  frame_history: string[];
  salience_landscape: Record<string, number>;
}

export interface WisdomState {
  sophrosyne_level: number;
  wisdom_score: number;
  beliefs_count: number;
  deception_count: number;
  insights_count: number;
  transformation_history: any[];
}

export interface CognitiveState {
  embedding: EmbeddingState;
  grip: GripState;
  perspective: FrameState;
  wisdom: WisdomState;
}

export interface CycleModeConfig {
  phase: string;
  echoDepth: number;
  attentionThreshold: number;
}

export interface CycleResult {
  result: any;
  state: CognitiveState;
}

/**
 * Configuration for EchogenesisService
 */
export interface EchogenesisConfig {
  host?: string;
  port?: number;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
}

/**
 * EchogenesisService
 *
 * Main service class for echogenesis interactions.
 */
export class EchogenesisService extends EventEmitter {
  private baseUrl: string;
  private config: Required<EchogenesisConfig>;
  private connected: boolean = false;
  private healthCheckInterval: NodeJS.Timeout | null = null;

  constructor(config: EchogenesisConfig = {}) {
    super();

    // Use evolutionary config for defaults
    const DEFAULT_API_PORT = Math.round(getConfig("apiPort"));
    const DEFAULT_TIMEOUT = Math.round(getConfig("requestTimeout"));
    const DEFAULT_RETRY_ATTEMPTS = Math.round(getConfig("retryAttempts"));

    this.config = {
      host: config.host || DEFAULT_API_HOST,
      port: config.port || DEFAULT_API_PORT,
      timeout: config.timeout || DEFAULT_TIMEOUT,
      retryAttempts: config.retryAttempts || DEFAULT_RETRY_ATTEMPTS,
      retryDelay: config.retryDelay || 1000,
    };

    this.baseUrl = `http://${this.config.host}:${this.config.port}`;

    // Subscribe to port changes for dynamic reconfiguration
    getEvolutionaryConfig().subscribe("apiPort", newPort => {
      const roundedPort = Math.round(newPort);
      if (this.config.port !== roundedPort) {
        this.config.port = roundedPort;
        this.baseUrl = `http://${this.config.host}:${roundedPort}`;
        this.emit("config:changed", { port: roundedPort });
      }
    });
  }

  /**
   * Initialize service and check connection
   */
  async initialize(): Promise<boolean> {
    try {
      const healthy = await this.healthCheck();
      if (healthy) {
        this.connected = true;
        this.emit("connected");
        this.startHealthMonitor();
      }
      return healthy;
    } catch (error) {
      this.emit("error", error);
      return false;
    }
  }

  /**
   * Shutdown service
   */
  async shutdown(): Promise<void> {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
    this.connected = false;
    this.emit("disconnected");
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.fetch("/health", "GET");
      return response.status === "healthy";
    } catch {
      return false;
    }
  }

  /**
   * Start health monitoring
   */
  private startHealthMonitor(): void {
    this.healthCheckInterval = setInterval(async () => {
      const healthy = await this.healthCheck();
      if (!healthy && this.connected) {
        this.connected = false;
        this.emit("disconnected");
      } else if (healthy && !this.connected) {
        this.connected = true;
        this.emit("reconnected");
      }
    }, 30000);
  }

  // ================== EMBEDDING API ==================

  /**
   * Perform adaptive dimensional projection
   */
  async adaptiveProject(
    data: number[],
    cognitiveLoad: number,
    attentionThreshold: number,
    context?: Record<string, any>
  ): Promise<EmbeddingResult> {
    return this.fetch("/embedding/project", "POST", {
      data,
      cognitive_load: cognitiveLoad,
      attention_threshold: attentionThreshold,
      context,
    });
  }

  /**
   * Create multi-scale embeddings
   */
  async multiScaleEmbed(
    data: number[],
    scales?: string[]
  ): Promise<MultiScaleEmbedding> {
    return this.fetch("/embedding/multiscale", "POST", {
      data,
      scales,
    });
  }

  /**
   * Create embodiment manifold
   */
  async createEmbodimentManifold(
    sensory: number[],
    motor: number[],
    cognitive: number[]
  ): Promise<EmbodimentManifold> {
    return this.fetch("/embedding/manifold", "POST", {
      sensory,
      motor,
      cognitive,
    });
  }

  // ================== GRIP API ==================

  /**
   * Realize relevance across possibilities
   */
  async realizeRelevance(
    possibilities: any[],
    goals?: any[],
    constraints?: any[]
  ): Promise<RelevanceResult> {
    return this.fetch("/grip/realize", "POST", {
      possibilities,
      goals,
      constraints,
    });
  }

  /**
   * Get top-k most relevant possibilities
   */
  async getTopRelevant(possibilities: any[], k: number = 5): Promise<any> {
    return this.fetch("/grip/top", "POST", {
      possibilities,
      k,
    });
  }

  // ================== PERSPECTIVE API ==================

  /**
   * Switch cognitive frame
   */
  async switchFrame(
    frameName: string,
    context?: Record<string, any>
  ): Promise<{
    success: boolean;
    current_frame: string | null;
    state: FrameState;
  }> {
    return this.fetch("/perspective/switch", "POST", {
      frame: frameName,
      context,
    });
  }

  /**
   * Perceive data through current frame
   */
  async perceive(data: Record<string, any>): Promise<{
    perceived: any;
    frame: string | null;
  }> {
    return this.fetch("/perspective/perceive", "POST", {
      data,
    });
  }

  /**
   * See data as particular aspect
   */
  async seeAs(
    data: Record<string, any>,
    aspect: string,
    patternType?: string
  ): Promise<{
    perceived: any;
    aspect: string;
  }> {
    return this.fetch("/perspective/see_as", "POST", {
      data,
      aspect,
      pattern_type: patternType,
    });
  }

  /**
   * Get available frames
   */
  async getAvailableFrames(): Promise<string[]> {
    const result = await this.fetch("/frames", "GET");
    return result.frames;
  }

  // ================== WISDOM API ==================

  /**
   * Add a belief
   */
  async addBelief(
    id: string,
    content: string,
    confidence: number = 0.5
  ): Promise<{
    belief_id: string;
    content: string;
    confidence: number;
  }> {
    return this.fetch("/wisdom/belief", "POST", {
      id,
      content,
      confidence,
    });
  }

  /**
   * Perform self-examination
   */
  async examineSelf(): Promise<{
    insights: Array<{ question: string; discovery: string }>;
    count: number;
  }> {
    return this.fetch("/wisdom/examine", "POST", {});
  }

  /**
   * Detect self-deceptions
   */
  async detectDeceptions(): Promise<{
    deceptions: any[];
    count: number;
  }> {
    return this.fetch("/wisdom/deceptions", "POST", {});
  }

  /**
   * Run wisdom cultivation cycle
   */
  async cultivateWisdom(): Promise<{
    result: any;
    wisdom_score: number;
  }> {
    return this.fetch("/wisdom/cultivate", "POST", {});
  }

  /**
   * Get wisdom state
   */
  async getWisdomState(): Promise<WisdomState> {
    return this.fetch("/wisdom", "GET");
  }

  // ================== FULL CYCLE API ==================

  /**
   * Execute complete cognitive cycle
   */
  async cognitiveCycle(input: Record<string, any>): Promise<CycleResult> {
    return this.fetch("/cycle", "POST", {
      input,
    });
  }

  /**
   * Get full system state
   */
  async getState(): Promise<CognitiveState> {
    return this.fetch("/state", "GET");
  }

  // ================== HELPERS ==================

  /**
   * Internal fetch wrapper
   */
  private async fetch(
    endpoint: string,
    method: "GET" | "POST",
    body?: any
  ): Promise<any> {
    const url = `${this.baseUrl}${endpoint}`;

    const options: RequestInit = {
      method,
      headers: {
        "Content-Type": "application/json",
      },
    };

    if (body && method === "POST") {
      options.body = JSON.stringify(body);
    }

    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.config.retryAttempts; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(
          () => controller.abort(),
          this.config.timeout
        );

        const response = await fetch(url, {
          ...options,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP error: ${response.status}`);
        }

        return await response.json();
      } catch (error) {
        lastError = error as Error;

        if (attempt < this.config.retryAttempts - 1) {
          await new Promise(resolve =>
            setTimeout(resolve, this.config.retryDelay * (attempt + 1))
          );
        }
      }
    }

    throw lastError || new Error("Failed to fetch");
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connected;
  }
}

/**
 * Create echogenesis service singleton
 */
let serviceInstance: EchogenesisService | null = null;

export function getEchogenesisService(
  config?: EchogenesisConfig
): EchogenesisService {
  if (!serviceInstance) {
    serviceInstance = new EchogenesisService(config);
  }
  return serviceInstance;
}

export function resetEchogenesisService(): void {
  if (serviceInstance) {
    serviceInstance.shutdown();
    serviceInstance = null;
  }
}

export default EchogenesisService;
