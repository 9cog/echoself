/**
 * Cognitive Grip Service
 * ======================
 *
 * TypeScript service for optimal cognitive grip management.
 * Provides high-level interface for relevance realization
 * and attention allocation.
 *
 * @author Deep Tree Echo
 * @date June 2026
 */

import { EventEmitter } from "events";
import {
  EchogenesisService,
  getEchogenesisService,
  GripState,
} from "./echogenesisService.ts";
import { getConfig, getEvolutionaryConfig } from "./evolutionaryConfig.ts";

// Types
export interface CognitivePossibility {
  id: string;
  content: any;
  metadata?: Record<string, any>;
  initialRelevance?: number;
}

export interface RankedPossibility extends CognitivePossibility {
  relevanceScore: number;
  filtered: boolean;
  ranking: number;
}

export interface CognitiveGoal {
  id: string;
  description: string;
  weight: number;
  targetState?: any;
}

export interface CognitiveConstraint {
  id: string;
  type: "hard" | "soft";
  description: string;
  weight: number;
  predicate?: (possibility: CognitivePossibility) => boolean;
}

export interface OpponentBalance {
  name: string;
  current: number;
  target: number;
  adjustment: number;
}

export interface GripMetrics {
  quality: number;
  filteringEfficiency: number;
  attentionCoherence: number;
  relevancePrecision: number;
}

export interface GripConfiguration {
  costWeights: {
    goalAlignment: number;
    predictivePower: number;
    cognitiveEconomy: number;
    noveltyValue: number;
    contextualFit: number;
  };
  opponentProcesses: Array<{
    positive: string;
    negative: string;
    balance: number;
  }>;
  thresholds: {
    relevance: number;
    attention: number;
    filtering: number;
  };
}

/**
 * CognitiveGripService
 *
 * Manages optimal cognitive grip through relevance realization.
 */
export class CognitiveGripService extends EventEmitter {
  private echogenesis: EchogenesisService;
  private goals: Map<string, CognitiveGoal> = new Map();
  private constraints: Map<string, CognitiveConstraint> = new Map();
  private opponentBalances: Map<string, number> = new Map();

  // Dynamic grip from evolutionary config
  private get defaultGrip(): number {
    return getConfig("confidenceBaseline");
  }
  private _currentGrip: number;

  constructor(echogenesis?: EchogenesisService) {
    super();
    this.echogenesis = echogenesis || getEchogenesisService();
    this._currentGrip = this.defaultGrip;

    // Default opponent process balances
    this.opponentBalances.set("exploration_exploitation", 0.5);
    this.opponentBalances.set("breadth_depth", 0.5);
    this.opponentBalances.set("speed_accuracy", 0.5);
    this.opponentBalances.set("certainty_openness", 0.5);

    // Subscribe to evolutionary config changes
    getEvolutionaryConfig().subscribe("confidenceBaseline", value => {
      // Adjust current grip proportionally when baseline changes
      // Guard against division by zero
      const currentDefault = this.defaultGrip || 0.5;
      const ratio = this._currentGrip / currentDefault;
      this._currentGrip = value * ratio;
    });
  }

  // ================== GOAL MANAGEMENT ==================

  /**
   * Add a cognitive goal
   */
  addGoal(goal: CognitiveGoal): void {
    this.goals.set(goal.id, goal);
    this.emit("goal:added", goal);
  }

  /**
   * Remove a goal
   */
  removeGoal(goalId: string): boolean {
    const removed = this.goals.delete(goalId);
    if (removed) {
      this.emit("goal:removed", goalId);
    }
    return removed;
  }

  /**
   * Update goal weight
   */
  updateGoalWeight(goalId: string, weight: number): boolean {
    const goal = this.goals.get(goalId);
    if (goal) {
      goal.weight = Math.max(0, Math.min(1, weight));
      this.emit("goal:updated", goal);
      return true;
    }
    return false;
  }

  /**
   * Get all active goals
   */
  getGoals(): CognitiveGoal[] {
    return Array.from(this.goals.values());
  }

  // ================== CONSTRAINT MANAGEMENT ==================

  /**
   * Add a constraint
   */
  addConstraint(constraint: CognitiveConstraint): void {
    this.constraints.set(constraint.id, constraint);
    this.emit("constraint:added", constraint);
  }

  /**
   * Remove a constraint
   */
  removeConstraint(constraintId: string): boolean {
    const removed = this.constraints.delete(constraintId);
    if (removed) {
      this.emit("constraint:removed", constraintId);
    }
    return removed;
  }

  /**
   * Get all constraints
   */
  getConstraints(): CognitiveConstraint[] {
    return Array.from(this.constraints.values());
  }

  // ================== OPPONENT PROCESSES ==================

  /**
   * Adjust opponent balance
   */
  adjustOpponentBalance(process: string, direction: number): number {
    const current = this.opponentBalances.get(process) || 0.5;
    const newBalance = Math.max(0, Math.min(1, current + direction * 0.1));
    this.opponentBalances.set(process, newBalance);
    this.emit("opponent:adjusted", { process, balance: newBalance });
    return newBalance;
  }

  /**
   * Get opponent balances
   */
  getOpponentBalances(): OpponentBalance[] {
    return Array.from(this.opponentBalances.entries()).map(
      ([name, current]) => ({
        name,
        current,
        target: 0.5,
        adjustment: 0.5 - current,
      })
    );
  }

  /**
   * Shift toward exploration
   */
  explore(): void {
    this.adjustOpponentBalance("exploration_exploitation", 1);
    this.adjustOpponentBalance("breadth_depth", 1);
    this.adjustOpponentBalance("certainty_openness", -1);
  }

  /**
   * Shift toward exploitation
   */
  exploit(): void {
    this.adjustOpponentBalance("exploration_exploitation", -1);
    this.adjustOpponentBalance("breadth_depth", -1);
    this.adjustOpponentBalance("certainty_openness", 1);
  }

  // ================== RELEVANCE REALIZATION ==================

  /**
   * Realize relevance across possibilities
   */
  async realizeRelevance(
    possibilities: CognitivePossibility[]
  ): Promise<RankedPossibility[]> {
    // Prepare API call
    const goalsArray = Array.from(this.goals.values());
    const constraintsArray = Array.from(this.constraints.values());

    // Apply local constraint predicates
    const preFiltered = possibilities.filter(p => {
      const hardConstraints = constraintsArray.filter(
        c => c.type === "hard" && c.predicate
      );
      return hardConstraints.every(c => c.predicate!(p));
    });

    try {
      // Call echogenesis API
      const result = await this.echogenesis.realizeRelevance(
        preFiltered.map(p => ({
          id: p.id,
          content: p.content,
          metadata: p.metadata,
          initial_relevance: p.initialRelevance,
        })),
        goalsArray,
        constraintsArray.filter(c => !c.predicate) // Send non-predicate constraints
      );

      // Update current grip
      this._currentGrip = result.grip_quality;
      this.emit("grip:updated", this._currentGrip);

      // Map results
      const ranked: RankedPossibility[] = result.ranked.map((r, idx) => ({
        id: r.possibility.id,
        content: r.possibility.content,
        metadata: r.possibility.metadata,
        relevanceScore: r.relevance_score,
        filtered: r.filtered,
        ranking: idx + 1,
      }));

      return ranked;
    } catch (error) {
      this.emit("error", error);
      throw error;
    }
  }

  /**
   * Get top-k most relevant
   */
  async getTopRelevant(
    possibilities: CognitivePossibility[],
    k: number = 5
  ): Promise<RankedPossibility[]> {
    const ranked = await this.realizeRelevance(possibilities);
    return ranked.slice(0, k);
  }

  /**
   * Filter by threshold
   */
  async filterByThreshold(
    possibilities: CognitivePossibility[],
    threshold?: number
  ): Promise<RankedPossibility[]> {
    const effectiveThreshold = threshold ?? getConfig("attentionThreshold");
    const ranked = await this.realizeRelevance(possibilities);
    return ranked.filter(p => p.relevanceScore >= effectiveThreshold);
  }

  // ================== GRIP METRICS ==================

  /**
   * Get current grip quality
   */
  getCurrentGrip(): number {
    return this._currentGrip;
  }

  /**
   * Get comprehensive grip metrics
   */
  async getGripMetrics(): Promise<GripMetrics> {
    const state = await this.echogenesis.getState();
    const grip = state.grip;

    // Calculate derived metrics
    const filteringEfficiency =
      1 -
      Object.values(grip.opponent_balances).reduce((a, b) => a + b, 0) /
        Object.keys(grip.opponent_balances).length;

    const attentionCoherence = 1 - Math.abs(0.5 - grip.grip_quality) * 2; // Peaks at 0.5

    return {
      quality: grip.grip_quality,
      filteringEfficiency,
      attentionCoherence,
      relevancePrecision: grip.optimal_grip,
    };
  }

  /**
   * Get grip state from backend
   */
  async getGripState(): Promise<GripState> {
    const state = await this.echogenesis.getState();
    return state.grip;
  }

  // ================== CONFIGURATION ==================

  /**
   * Apply grip configuration
   */
  applyConfiguration(config: GripConfiguration): void {
    // Apply opponent balances
    for (const op of config.opponentProcesses) {
      const key = `${op.positive}_${op.negative}`;
      this.opponentBalances.set(key, op.balance);
    }

    this.emit("config:applied", config);
  }

  /**
   * Reset to default state
   */
  reset(): void {
    this.goals.clear();
    this.constraints.clear();
    this._currentGrip = this.defaultGrip;

    this.opponentBalances.set("exploration_exploitation", 0.5);
    this.opponentBalances.set("breadth_depth", 0.5);
    this.opponentBalances.set("speed_accuracy", 0.5);
    this.opponentBalances.set("certainty_openness", 0.5);

    this.emit("reset");
  }
}

/**
 * Create service singleton
 */
let serviceInstance: CognitiveGripService | null = null;

export function getCognitiveGripService(
  echogenesis?: EchogenesisService
): CognitiveGripService {
  if (!serviceInstance) {
    serviceInstance = new CognitiveGripService(echogenesis);
  }
  return serviceInstance;
}

export function resetCognitiveGripService(): void {
  serviceInstance = null;
}

export default CognitiveGripService;
