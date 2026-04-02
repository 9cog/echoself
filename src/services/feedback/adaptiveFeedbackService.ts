/**
 * Adaptive Feedback Service
 *
 * Implements the adaptive, hypergraph-encoded feedback loop for
 * DeepTreeEcho's distributed cognition framework.
 *
 * Collects community feedback on cognitive models, scores their salience,
 * integrates with GitHub Copilot, and broadcasts refinements to the community.
 */

import { HypergraphSchemeCore } from "./hypergraphSchemeCore";

// --- Interfaces ---

export interface ProjectModel {
  id: string;
  name: string;
  description: string;
  version: string;
  lastModified: Date;
  usageCount: number;
  tags?: string[];
}

export interface CommunityFeedback {
  modelId: string;
  userId: string;
  type:
    | "improvement"
    | "bug"
    | "feature"
    | "validation"
    | "deprecation"
    | "performance"
    | "feature_request";
  priority: "low" | "medium" | "high" | "critical" | "urgent";
  description: string;
  votes: number;
  metadata?: Record<string, unknown>;
}

export interface CopilotRequest {
  modelId: string;
  requestType: "optimize" | "refactor" | "document" | "test" | "analyze";
  context: string;
  constraints?: string[];
  targetQuality?: number;
}

export interface CopilotResponse {
  requestId: string;
  modelId: string;
  suggestions: string[];
  codeChanges?: string;
  qualityScore: number;
  confidence: number;
  processingTime: number;
}

export interface AdaptiveThresholds {
  salienceThreshold: number;
  attentionThreshold: number;
  feedbackWeightThreshold: number;
  copilotConfidenceThreshold: number;
}

interface AdaptiveThresholdStatus {
  attentionThreshold: number;
  salienceThreshold: number;
  cognitiveLoad: number;
  recentActivity: number;
}

interface SystemStatus {
  projectModelsCount: number;
  communityFeedbackCount: number;
  pendingCopilotRequests: number;
  averageSalienceScore: number;
  attentionFilteredNodes: number;
  hypergraphNodesCount: number;
  adaptiveThresholds: AdaptiveThresholdStatus;
  lastFeedbackLoopRun: Date | null;
  lastFeedbackCycle: Date;
  isRunning: boolean;
}

// --- Service class ---

export class AdaptiveFeedbackService {
  private static instance: AdaptiveFeedbackService;

  private hypergraphCore: HypergraphSchemeCore;
  private projectModels: Map<string, ProjectModel> = new Map();
  private communityFeedback: CommunityFeedback[] = [];
  private copilotRequests: Map<string, CopilotRequest> = new Map();
  private thresholds: AdaptiveThresholds = {
    salienceThreshold: 0.4,
    attentionThreshold: 0.5,
    feedbackWeightThreshold: 0.3,
    copilotConfidenceThreshold: 0.7,
  };
  private lastFeedbackLoopRun: Date | null = null;
  private isRunning = false;
  private feedbackIntervalId: number | null = null;

  private constructor() {
    this.hypergraphCore = new HypergraphSchemeCore();
  }

  public static getInstance(): AdaptiveFeedbackService {
    if (!AdaptiveFeedbackService.instance) {
      AdaptiveFeedbackService.instance = new AdaptiveFeedbackService();
    }
    return AdaptiveFeedbackService.instance;
  }

  // --- Model management ---

  public registerProjectModel(model: ProjectModel): void {
    this.projectModels.set(model.id, model);

    // Add to hypergraph as a cognitive node
    this.hypergraphCore.createNode(
      model.id,
      "model",
      {
        name: model.name,
        description: model.description,
        version: model.version,
      },
      []
    );
  }

  public getProjectModel(id: string): ProjectModel | undefined {
    return this.projectModels.get(id);
  }

  public getAllProjectModels(): ProjectModel[] {
    return Array.from(this.projectModels.values());
  }

  // --- Community feedback ---

  public addCommunityFeedback(feedback: CommunityFeedback): void {
    this.communityFeedback.push({
      ...feedback,
      metadata: feedback.metadata ?? {},
    });
  }

  public getFeedbackForModel(modelId: string): CommunityFeedback[] {
    return this.communityFeedback.filter(f => f.modelId === modelId);
  }

  // --- Adaptive thresholds ---

  public updateThresholds(partial: Partial<AdaptiveThresholds>): void {
    this.thresholds = { ...this.thresholds, ...partial };
  }

  public getThresholds(): AdaptiveThresholds {
    return { ...this.thresholds };
  }

  // --- Copilot integration ---

  public submitCopilotRequest(request: CopilotRequest): string {
    const requestId = `req_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    this.copilotRequests.set(requestId, request);
    return requestId;
  }

  public async processCopilotRequest(
    requestId: string
  ): Promise<CopilotResponse | null> {
    const request = this.copilotRequests.get(requestId);
    if (!request) return null;

    const model = this.projectModels.get(request.modelId);
    const processingStart = Date.now();

    // Simulate Copilot processing using feedback data
    const feedback = this.getFeedbackForModel(request.modelId);
    const suggestions = feedback
      .filter(f => f.priority === "high" || f.priority === "critical")
      .slice(0, 3)
      .map(f => f.description);

    const response: CopilotResponse = {
      requestId,
      modelId: request.modelId,
      suggestions:
        suggestions.length > 0
          ? suggestions
          : [`Optimize ${model?.name ?? "model"} based on ${request.requestType} analysis`],
      qualityScore: 0.75,
      confidence: 0.8,
      processingTime: Date.now() - processingStart,
    };

    this.copilotRequests.delete(requestId);
    return response;
  }

  // --- Feedback loop ---

  public async triggerFeedbackLoop(): Promise<void> {
    if (this.isRunning) {
      console.warn("Feedback loop already running, skipping...");
      return;
    }

    this.isRunning = true;

    try {
      // 1. Calculate salience for all models
      for (const [modelId] of this.projectModels) {
        const feedback = this.getFeedbackForModel(modelId);
        const metrics = this.hypergraphCore.calculateSalienceMetrics(
          modelId,
          feedback.map(f => ({
            priority: f.priority,
            votes: f.votes,
            type: f.type,
          }))
        );

        // 2. Adapt attention thresholds based on cognitive load
        const cognitiveLoad =
          this.communityFeedback.length / Math.max(1, this.projectModels.size * 10);
        const recentActivity = feedback.length / Math.max(1, feedback.length + 1);
        const newThreshold = this.hypergraphCore.adaptiveAttention(
          cognitiveLoad,
          recentActivity
        );

        if (metrics.overall < this.thresholds.salienceThreshold) {
          console.log(
            `ℹ️ Model ${modelId} below salience threshold (${metrics.overall.toFixed(2)})`
          );
        }

        this.thresholds.attentionThreshold = newThreshold;
      }

      // 3. Mine cognitive patterns across all feedback
      const patterns = this.hypergraphCore.mineCognitivePatterns(
        this.thresholds.salienceThreshold
      );

      if (patterns.length > 0) {
        console.log(`🔍 Found ${patterns.length} cognitive patterns`);
      }

      this.lastFeedbackLoopRun = new Date();
    } finally {
      this.isRunning = false;
    }
  }

  // --- System status ---

  public getSystemStatus(): SystemStatus {
    let totalSalience = 0;

    for (const [modelId] of this.projectModels) {
      const metrics = this.hypergraphCore.calculateSalienceMetrics(modelId, []);
      totalSalience += metrics.overall;
    }

    const modelCount = this.projectModels.size;
    const averageSalience = modelCount > 0 ? totalSalience / modelCount : 0;
    const hypergraphNodesCount = modelCount; // one node per model

    const cognitiveLoad =
      this.communityFeedback.length /
      Math.max(1, this.projectModels.size * 10);
    const recentActivity =
      this.communityFeedback.length /
      Math.max(1, this.communityFeedback.length + 1);

    const attentionFiltered = Math.round(
      hypergraphNodesCount * this.thresholds.attentionThreshold
    );

    const now = new Date();

    return {
      projectModelsCount: this.projectModels.size,
      communityFeedbackCount: this.communityFeedback.length,
      pendingCopilotRequests: this.copilotRequests.size,
      averageSalienceScore: averageSalience,
      attentionFilteredNodes: attentionFiltered,
      hypergraphNodesCount,
      adaptiveThresholds: {
        attentionThreshold: this.thresholds.attentionThreshold,
        salienceThreshold: this.thresholds.salienceThreshold,
        cognitiveLoad,
        recentActivity,
      },
      lastFeedbackLoopRun: this.lastFeedbackLoopRun,
      lastFeedbackCycle: this.lastFeedbackLoopRun ?? now,
      isRunning: this.isRunning,
    };
  }

  /**
   * Set the interval for automatic feedback loop execution (in milliseconds).
   * Pass 0 to disable automatic execution.
   */
  public setFeedbackCycleInterval(intervalMs: number): void {
    if (this.feedbackIntervalId !== null) {
      clearInterval(this.feedbackIntervalId);
      this.feedbackIntervalId = null;
    }

    if (intervalMs > 0) {
      this.feedbackIntervalId = window.setInterval(() => {
        this.triggerFeedbackLoop().catch(err =>
          console.error("Feedback loop error:", err)
        );
      }, intervalMs);
    }
  }
}
