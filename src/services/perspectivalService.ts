/**
 * Perspectival Service
 * ====================
 *
 * TypeScript service for frame-switching and perspectival knowing.
 * Manages cognitive frames, aspect perception, and gestalt shifts.
 *
 * @author Deep Tree Echo
 * @date June 2026
 */

import { EventEmitter } from "events";
import {
  EchogenesisService,
  getEchogenesisService,
  FrameState,
} from "./echogenesisService.ts";

// Types
export type FrameType =
  | "analytical"
  | "intuitive"
  | "embodied"
  | "relational"
  | "creative"
  | "contemplative"
  | "pragmatic"
  | "systemic";

export interface Frame {
  type: FrameType;
  name: string;
  description: string;
  salienceModifiers: Record<string, number>;
  activationLevel: number;
  history: Date[];
}

export interface SaliencePoint {
  key: string;
  value: number;
  normalized: number;
}

export interface SalienceLandscape {
  points: SaliencePoint[];
  peaks: SaliencePoint[];
  valleys: SaliencePoint[];
  totalSalience: number;
}

export interface Aspect {
  name: string;
  pattern: string;
  recognitionConfidence: number;
  implications: string[];
}

export interface GestaltShift {
  fromFrame: FrameType;
  toFrame: FrameType;
  trigger: string;
  timestamp: Date;
  confidence: number;
}

export interface PerceptionResult {
  perceived: any;
  frame: FrameType;
  salience: SalienceLandscape;
  aspects: Aspect[];
}

/**
 * PerspectivalService
 *
 * Manages cognitive perspectives and frame-switching.
 */
export class PerspectivalService extends EventEmitter {
  private echogenesis: EchogenesisService;
  private currentFrame: FrameType = "analytical";
  private frameHistory: GestaltShift[] = [];
  private customFrames: Map<string, Frame> = new Map();
  private salienceLandscape: Map<string, number> = new Map();

  constructor(echogenesis?: EchogenesisService) {
    super();
    this.echogenesis = echogenesis || getEchogenesisService();
    this.initializeDefaultSalience();
  }

  /**
   * Initialize default salience landscape
   */
  private initializeDefaultSalience(): void {
    const defaults: Record<string, number> = {
      novelty: 0.5,
      familiarity: 0.5,
      relevance: 0.5,
      complexity: 0.5,
      urgency: 0.3,
      importance: 0.5,
      emotion: 0.4,
      pattern: 0.5,
    };

    for (const [key, value] of Object.entries(defaults)) {
      this.salienceLandscape.set(key, value);
    }
  }

  // ================== FRAME MANAGEMENT ==================

  /**
   * Switch to a different cognitive frame
   */
  async switchFrame(
    frameType: FrameType,
    context?: Record<string, any>
  ): Promise<boolean> {
    const previousFrame = this.currentFrame;

    try {
      const result = await this.echogenesis.switchFrame(frameType, context);

      if (result.success) {
        this.currentFrame = frameType;

        // Record gestalt shift
        const shift: GestaltShift = {
          fromFrame: previousFrame,
          toFrame: frameType,
          trigger: context?.trigger || "explicit",
          timestamp: new Date(),
          confidence: 1.0,
        };

        this.frameHistory.push(shift);
        this.emit("frame:switched", shift);

        return true;
      }

      return false;
    } catch (error) {
      this.emit("error", error);
      return false;
    }
  }

  /**
   * Get current frame
   */
  getCurrentFrame(): FrameType {
    return this.currentFrame;
  }

  /**
   * Get frame history
   */
  getFrameHistory(): GestaltShift[] {
    return [...this.frameHistory];
  }

  /**
   * Get available frames
   */
  async getAvailableFrames(): Promise<FrameType[]> {
    const frames = await this.echogenesis.getAvailableFrames();
    return frames as FrameType[];
  }

  /**
   * Register custom frame
   */
  registerCustomFrame(frame: Frame): void {
    this.customFrames.set(frame.name, frame);
    this.emit("frame:registered", frame);
  }

  // ================== PERCEPTION ==================

  /**
   * Perceive data through current frame
   */
  async perceive(data: Record<string, any>): Promise<PerceptionResult> {
    const result = await this.echogenesis.perceive(data);

    // Build salience landscape
    const salience = this.computeSalienceLandscape(result.perceived);

    // Detect aspects
    const aspects = this.detectAspects(result.perceived);

    return {
      perceived: result.perceived,
      frame: (result.frame as FrameType) || this.currentFrame,
      salience,
      aspects,
    };
  }

  /**
   * See data as a particular aspect
   */
  async seeAs(
    data: Record<string, any>,
    aspect: string,
    patternType?: string
  ): Promise<{
    perceived: any;
    aspect: Aspect;
  }> {
    const result = await this.echogenesis.seeAs(data, aspect, patternType);

    return {
      perceived: result.perceived,
      aspect: {
        name: aspect,
        pattern: patternType || "default",
        recognitionConfidence: 0.8,
        implications: [],
      },
    };
  }

  /**
   * Multi-perspective perception
   */
  async perceiveMulti(
    data: Record<string, any>,
    frames: FrameType[]
  ): Promise<Map<FrameType, PerceptionResult>> {
    const results = new Map<FrameType, PerceptionResult>();
    const originalFrame = this.currentFrame;

    for (const frame of frames) {
      await this.switchFrame(frame);
      const perception = await this.perceive(data);
      results.set(frame, perception);
    }

    // Restore original frame
    await this.switchFrame(originalFrame);

    return results;
  }

  // ================== SALIENCE ==================

  /**
   * Compute salience landscape from perceived data
   */
  private computeSalienceLandscape(
    perceived: Record<string, any>
  ): SalienceLandscape {
    const points: SaliencePoint[] = [];
    let total = 0;

    // Extract salience from perceived data
    for (const [key, rawValue] of Object.entries(perceived)) {
      // Compute salience value
      let value = 0;
      if (typeof rawValue === "number") {
        value = Math.abs(rawValue);
      } else if (typeof rawValue === "string") {
        value = rawValue.length / 100;
      } else if (Array.isArray(rawValue)) {
        value = rawValue.length / 10;
      } else if (typeof rawValue === "object" && rawValue !== null) {
        value = Object.keys(rawValue).length / 10;
      }

      value = Math.min(1, Math.max(0, value));
      total += value;

      points.push({ key, value, normalized: 0 });
    }

    // Normalize
    if (total > 0) {
      for (const point of points) {
        point.normalized = point.value / total;
      }
    }

    // Find peaks and valleys
    points.sort((a, b) => b.value - a.value);
    const peaks = points.slice(0, 3);
    const valleys = points.slice(-3);

    return {
      points,
      peaks,
      valleys,
      totalSalience: total,
    };
  }

  /**
   * Update salience value
   */
  updateSalience(key: string, value: number): void {
    const normalized = Math.max(0, Math.min(1, value));
    this.salienceLandscape.set(key, normalized);
    this.emit("salience:updated", { key, value: normalized });
  }

  /**
   * Get current salience landscape
   */
  getSalienceLandscape(): SalienceLandscape {
    const points: SaliencePoint[] = [];
    let total = 0;

    for (const [key, value] of this.salienceLandscape.entries()) {
      total += value;
      points.push({ key, value, normalized: 0 });
    }

    if (total > 0) {
      for (const point of points) {
        point.normalized = point.value / total;
      }
    }

    points.sort((a, b) => b.value - a.value);

    return {
      points,
      peaks: points.slice(0, 3),
      valleys: points.slice(-3),
      totalSalience: total,
    };
  }

  // ================== ASPECT PERCEPTION ==================

  /**
   * Detect aspects in perceived data
   */
  private detectAspects(perceived: Record<string, any>): Aspect[] {
    const aspects: Aspect[] = [];

    // Pattern recognition heuristics
    const patterns = [
      { name: "structure", keys: ["shape", "form", "structure", "hierarchy"] },
      {
        name: "process",
        keys: ["flow", "sequence", "steps", "transformation"],
      },
      {
        name: "relation",
        keys: ["link", "connection", "reference", "between"],
      },
      { name: "value", keys: ["importance", "weight", "priority", "worth"] },
    ];

    for (const pattern of patterns) {
      const matchingKeys = pattern.keys.filter(
        k =>
          Object.keys(perceived).some(pk => pk.toLowerCase().includes(k)) ||
          JSON.stringify(perceived).toLowerCase().includes(k)
      );

      if (matchingKeys.length > 0) {
        aspects.push({
          name: pattern.name,
          pattern: matchingKeys.join(","),
          recognitionConfidence: matchingKeys.length / pattern.keys.length,
          implications: [],
        });
      }
    }

    return aspects;
  }

  /**
   * Register aspect detector
   */
  registerAspectDetector(
    name: string,
    detector: (data: Record<string, any>) => Aspect | null
  ): void {
    // Store detector for later use
    this.emit("aspect:detector:registered", { name, detector });
  }

  // ================== GESTALT DETECTION ==================

  /**
   * Detect potential gestalt shifts
   */
  detectGestaltOpportunity(data: Record<string, any>): FrameType[] {
    const opportunities: FrameType[] = [];
    const currentSalience = this.computeSalienceLandscape(data);

    // Heuristics for frame switching
    const hasEmotionalContent = currentSalience.points.some(
      p =>
        ["emotion", "feeling", "affect"].some(k =>
          p.key.toLowerCase().includes(k)
        ) && p.value > 0.5
    );

    const hasPatternContent = currentSalience.points.some(
      p =>
        ["pattern", "structure", "form"].some(k =>
          p.key.toLowerCase().includes(k)
        ) && p.value > 0.5
    );

    const hasRelationalContent = currentSalience.points.some(
      p =>
        ["relation", "connection", "link"].some(k =>
          p.key.toLowerCase().includes(k)
        ) && p.value > 0.5
    );

    if (hasEmotionalContent && this.currentFrame !== "intuitive") {
      opportunities.push("intuitive");
    }

    if (hasPatternContent && this.currentFrame !== "analytical") {
      opportunities.push("analytical");
    }

    if (hasRelationalContent && this.currentFrame !== "relational") {
      opportunities.push("relational");
    }

    // Always suggest complementary frame
    const complementary: Record<FrameType, FrameType> = {
      analytical: "intuitive",
      intuitive: "analytical",
      embodied: "contemplative",
      relational: "systemic",
      creative: "pragmatic",
      contemplative: "embodied",
      pragmatic: "creative",
      systemic: "relational",
    };

    const comp = complementary[this.currentFrame];
    if (comp && !opportunities.includes(comp)) {
      opportunities.push(comp);
    }

    return opportunities;
  }

  // ================== STATE ==================

  /**
   * Get full perspective state
   */
  async getState(): Promise<FrameState> {
    return {
      current_frame: this.currentFrame,
      available_frames: Array.from(this.customFrames.keys()),
      frame_history: this.frameHistory.map(h => h.toFrame),
      salience_landscape: Object.fromEntries(this.salienceLandscape),
    };
  }

  /**
   * Reset to default state
   */
  reset(): void {
    this.currentFrame = "analytical";
    this.frameHistory = [];
    this.customFrames.clear();
    this.initializeDefaultSalience();
    this.emit("reset");
  }
}

/**
 * Create service singleton
 */
let serviceInstance: PerspectivalService | null = null;

export function getPerspectivalService(
  echogenesis?: EchogenesisService
): PerspectivalService {
  if (!serviceInstance) {
    serviceInstance = new PerspectivalService(echogenesis);
  }
  return serviceInstance;
}

export function resetPerspectivalService(): void {
  serviceInstance = null;
}

export default PerspectivalService;
