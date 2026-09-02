/**
 * Wisdom Cultivation Client for TypeScript
 * ==========================================
 *
 * TypeScript client for interacting with the Python WisdomCultivationSystem.
 * Provides wisdom cultivation capabilities to the EchoSelf cognitive architecture.
 *
 * @author Deep Tree Echo
 * @date June 2026
 */

import { EventEmitter } from "events";

// Types matching the Python API
export type VirtueType =
  | "intellectual_humility"
  | "intellectual_courage"
  | "intellectual_empathy"
  | "intellectual_perseverance"
  | "intellectual_autonomy"
  | "fairmindedness"
  | "confidence_in_reason";

export type WisdomDimension = "morality" | "meaning" | "mastery";

export interface Belief {
  id: string;
  content: string;
  confidence: number;
  evidence?: string[];
  reality_tested?: boolean;
  last_examined?: number;
  revision_count?: number;
  caring_about_truth?: number;
}

export interface Insight {
  question: string;
  discovery: string;
  implications?: string[];
  confidence?: number;
  timestamp?: number;
}

export interface SelfDeception {
  belief_id: string;
  type: string;
  severity: number;
  recommendation: string;
}

export interface VirtueState {
  virtue: VirtueType;
  level: number;
  practice_count?: number;
  last_practiced?: number;
}

export interface WisdomScore {
  overall: number;
  morality: number;
  meaning: number;
  mastery: number;
  sophrosyne: number;
  virtue_average?: number;
}

export interface RegulationAssessment {
  extremes: Array<{
    dimension: string;
    value: number;
    direction: "too_low" | "too_high";
  }>;
  well_regulated: Array<{ dimension: string; value: number }>;
  balance_score: number;
  recommendation: string;
}

export interface CultivationResult {
  timestamp: number;
  insights: Insight[];
  deceptions: SelfDeception[];
  virtue_updates: Record<string, number>;
  regulation_assessment: RegulationAssessment;
  wisdom_score: WisdomScore;
}

export interface WisdomState {
  beliefs_count: number;
  insights_count: number;
  virtues: Record<string, number>;
  weakest_virtues: VirtueType[];
  sophrosyne_balance: number;
  deceptions_detected: number;
  humility: {
    uncertainty_acknowledgments: number;
    overconfidence_detections: number;
  };
  wisdom_score: WisdomScore;
  history_size: number;
}

/**
 * WisdomCultivationClient
 *
 * Client for the Python Wisdom Cultivation System.
 * Manages beliefs, detects self-deception, and cultivates cognitive virtues.
 */
export class WisdomCultivationClient extends EventEmitter {
  private baseUrl: string;
  private timeout: number;
  private localBeliefs: Map<string, Belief> = new Map();
  private localInsights: Insight[] = [];
  private lastCultivation: CultivationResult | null = null;

  constructor(
    baseUrl: string = "http://localhost:8767",
    timeout: number = 30000
  ) {
    super();
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeout = timeout;
  }

  /**
   * Make HTTP request to the wisdom cultivation server
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

    try {
      const response = await fetch(url, options);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          `HTTP ${response.status}: ${errorData.error || response.statusText}`
        );
      }

      return (await response.json()) as T;
    } catch (error) {
      this.emit("error", error);
      throw error;
    }
  }

  // ================== BELIEF MANAGEMENT ==================

  /**
   * Add a belief to track
   */
  async addBelief(
    id: string,
    content: string,
    confidence: number = 0.5,
    evidence?: string[]
  ): Promise<Belief> {
    const belief: Belief = {
      id,
      content,
      confidence,
      evidence,
      reality_tested: false,
      revision_count: 0,
      caring_about_truth: 1.0,
    };

    this.localBeliefs.set(id, belief);
    this.emit("belief:added", belief);

    try {
      return await this.request<Belief>("POST", "/belief", {
        id,
        content,
        confidence,
        evidence,
      });
    } catch {
      // Fall back to local belief
      return belief;
    }
  }

  /**
   * Get a tracked belief
   */
  getBelief(id: string): Belief | undefined {
    return this.localBeliefs.get(id);
  }

  /**
   * Get all tracked beliefs
   */
  getAllBeliefs(): Belief[] {
    return Array.from(this.localBeliefs.values());
  }

  /**
   * Update belief confidence
   */
  updateBeliefConfidence(id: string, confidence: number): boolean {
    const belief = this.localBeliefs.get(id);
    if (belief) {
      belief.confidence = Math.max(0, Math.min(1, confidence));
      belief.revision_count = (belief.revision_count || 0) + 1;
      this.emit("belief:updated", belief);
      return true;
    }
    return false;
  }

  /**
   * Mark belief as reality-tested
   */
  markBeliefTested(id: string, result: boolean): boolean {
    const belief = this.localBeliefs.get(id);
    if (belief) {
      belief.reality_tested = true;
      // Adjust confidence based on test result
      if (!result) {
        belief.confidence = Math.max(0, belief.confidence - 0.2);
      }
      belief.last_examined = Date.now();
      this.emit("belief:tested", { belief, result });
      return true;
    }
    return false;
  }

  // ================== SELF-EXAMINATION ==================

  /**
   * Perform Socratic self-examination
   */
  async examineSelf(): Promise<Insight[]> {
    try {
      const result = await this.request<{ insights: Insight[] }>(
        "POST",
        "/examine"
      );
      this.localInsights.push(...result.insights);
      this.emit("insights:gained", result.insights);
      return result.insights;
    } catch {
      // Fallback: generate local insights
      const insights = this.generateLocalInsights();
      this.localInsights.push(...insights);
      return insights;
    }
  }

  /**
   * Generate local insights without server
   */
  private generateLocalInsights(): Insight[] {
    const highConfidenceBeliefs = Array.from(this.localBeliefs.values()).filter(
      b => b.confidence > 0.7
    );

    const insights: Insight[] = [];

    if (highConfidenceBeliefs.length > 0) {
      insights.push({
        question: "What beliefs do I hold with high confidence?",
        discovery: `Currently holding ${highConfidenceBeliefs.length} high-confidence beliefs`,
        implications: ["Consider seeking disconfirming evidence"],
        confidence: 0.9,
        timestamp: Date.now(),
      });
    }

    const untestedBeliefs = Array.from(this.localBeliefs.values()).filter(
      b => !b.reality_tested
    );

    if (untestedBeliefs.length > 0) {
      insights.push({
        question: "Which beliefs have I not reality-tested?",
        discovery: `${untestedBeliefs.length} beliefs remain untested`,
        implications: ["Prioritize testing important beliefs"],
        confidence: 0.8,
        timestamp: Date.now(),
      });
    }

    return insights;
  }

  /**
   * Get Socratic questions for a belief
   */
  getSocraticQuestions(beliefId: string): string[] {
    const belief = this.localBeliefs.get(beliefId);
    if (!belief) return [];

    const content = belief.content.substring(0, 50);
    return [
      `What do I really mean when I say '${content}...'?`,
      `What evidence would make me change my mind about this?`,
      `What am I assuming that I haven't examined?`,
      `How might someone with an opposing view see this?`,
      `What are the implications if I'm wrong about this?`,
      `Is my confidence in this belief justified by the evidence?`,
    ];
  }

  // ================== DECEPTION DETECTION ==================

  /**
   * Detect self-deceptions in beliefs
   */
  async detectDeceptions(): Promise<SelfDeception[]> {
    try {
      const result = await this.request<{ deceptions: SelfDeception[] }>(
        "POST",
        "/deceptions"
      );
      this.emit("deceptions:detected", result.deceptions);
      return result.deceptions;
    } catch {
      // Fallback: local detection
      return this.detectLocalDeceptions();
    }
  }

  /**
   * Local deception detection
   */
  private detectLocalDeceptions(): SelfDeception[] {
    const deceptions: SelfDeception[] = [];

    for (const belief of this.localBeliefs.values()) {
      // Check for bullshit (Frankfurt's definition)
      if (!belief.reality_tested && (belief.caring_about_truth || 1) < 0.5) {
        deceptions.push({
          belief_id: belief.id,
          type: "bullshit",
          severity: 1.0 - (belief.caring_about_truth || 0.5),
          recommendation:
            "Reality-test this belief and consider whether you truly care about its truth",
        });
      }

      // Check for confirmation bias
      if (
        belief.evidence &&
        belief.evidence.length > 0 &&
        belief.evidence.every(
          e =>
            e.toLowerCase().includes("confirm") ||
            e.toLowerCase().includes("support")
        )
      ) {
        deceptions.push({
          belief_id: belief.id,
          type: "confirmation_bias",
          severity: 0.6,
          recommendation:
            "Actively seek disconfirming evidence for this belief",
        });
      }

      // Check for unfalsifiable beliefs
      if (belief.revision_count === 0 && belief.confidence > 0.9) {
        deceptions.push({
          belief_id: belief.id,
          type: "unfalsifiable_belief",
          severity: 0.5,
          recommendation:
            "Consider what evidence would change your mind about this",
        });
      }
    }

    return deceptions;
  }

  // ================== WISDOM CULTIVATION ==================

  /**
   * Run full wisdom cultivation cycle
   */
  async cultivate(): Promise<CultivationResult> {
    try {
      const result = await this.request<CultivationResult>(
        "POST",
        "/cultivate"
      );
      this.lastCultivation = result;
      this.emit("cultivation:complete", result);
      return result;
    } catch {
      // Fallback: local cultivation
      return this.localCultivate();
    }
  }

  /**
   * Local wisdom cultivation
   */
  private localCultivate(): CultivationResult {
    const insights = this.generateLocalInsights();
    const deceptions = this.detectLocalDeceptions();

    const result: CultivationResult = {
      timestamp: Date.now(),
      insights,
      deceptions,
      virtue_updates: {
        intellectual_humility: 0.5,
        intellectual_courage: 0.5,
        intellectual_empathy: 0.5,
        intellectual_perseverance: 0.5,
        intellectual_autonomy: 0.5,
        fairmindedness: 0.5,
        confidence_in_reason: 0.5,
      },
      regulation_assessment: {
        extremes: [],
        well_regulated: [],
        balance_score: 0.5,
        recommendation: "Maintain current balance",
      },
      wisdom_score: {
        overall: 0.5,
        morality: 0.5,
        meaning: 0.5,
        mastery: 0.5,
        sophrosyne: 0.5,
      },
    };

    this.lastCultivation = result;
    return result;
  }

  /**
   * Get wisdom score
   */
  async getWisdomScore(): Promise<WisdomScore> {
    try {
      return await this.request<WisdomScore>("GET", "/wisdom/score");
    } catch {
      return (
        this.lastCultivation?.wisdom_score || {
          overall: 0.5,
          morality: 0.5,
          meaning: 0.5,
          mastery: 0.5,
          sophrosyne: 0.5,
        }
      );
    }
  }

  // ================== VIRTUE CULTIVATION ==================

  /**
   * Practice a cognitive virtue
   */
  async practiceVirtue(virtue: VirtueType): Promise<VirtueState | null> {
    try {
      return await this.request<VirtueState>("POST", "/virtue/practice", {
        virtue,
      });
    } catch {
      return null;
    }
  }

  /**
   * Get virtue levels
   */
  async getVirtueLevels(): Promise<Record<VirtueType, number>> {
    try {
      return await this.request<Record<VirtueType, number>>("GET", "/virtues");
    } catch {
      return (
        (this.lastCultivation?.virtue_updates as Record<
          VirtueType,
          number
        >) || {
          intellectual_humility: 0.5,
          intellectual_courage: 0.5,
          intellectual_empathy: 0.5,
          intellectual_perseverance: 0.5,
          intellectual_autonomy: 0.5,
          fairmindedness: 0.5,
          confidence_in_reason: 0.5,
        }
      );
    }
  }

  /**
   * Get weakest virtues for focused development
   */
  async getWeakestVirtues(n: number = 3): Promise<VirtueType[]> {
    const virtues = await this.getVirtueLevels();
    const sorted = Object.entries(virtues)
      .sort(([, a], [, b]) => a - b)
      .slice(0, n);
    return sorted.map(([v]) => v as VirtueType);
  }

  // ================== STATE ==================

  /**
   * Get full wisdom state
   */
  async getState(): Promise<WisdomState> {
    try {
      return await this.request<WisdomState>("GET", "/state");
    } catch {
      // Return local state
      return {
        beliefs_count: this.localBeliefs.size,
        insights_count: this.localInsights.length,
        virtues: this.lastCultivation?.virtue_updates || {},
        weakest_virtues: [],
        sophrosyne_balance:
          this.lastCultivation?.regulation_assessment.balance_score || 0.5,
        deceptions_detected: 0,
        humility: {
          uncertainty_acknowledgments: 0,
          overconfidence_detections: 0,
        },
        wisdom_score: this.lastCultivation?.wisdom_score || {
          overall: 0.5,
          morality: 0.5,
          meaning: 0.5,
          mastery: 0.5,
          sophrosyne: 0.5,
        },
        history_size: 0,
      };
    }
  }

  /**
   * Check if server is healthy
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
   * Get last cultivation result
   */
  getLastCultivation(): CultivationResult | null {
    return this.lastCultivation;
  }

  /**
   * Get all accumulated insights
   */
  getInsights(): Insight[] {
    return [...this.localInsights];
  }
}

// Singleton instance
let wisdomClient: WisdomCultivationClient | null = null;

export function getWisdomClient(baseUrl?: string): WisdomCultivationClient {
  if (!wisdomClient) {
    wisdomClient = new WisdomCultivationClient(baseUrl);
  }
  return wisdomClient;
}

export function resetWisdomClient(): void {
  wisdomClient = null;
}

export default WisdomCultivationClient;
