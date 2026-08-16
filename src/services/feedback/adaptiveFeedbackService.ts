/**
 * Adaptive Feedback Service
 *
 * Implements the adaptive, hypergraph-encoded feedback loop for
 * DeepTreeEcho's distributed cognition framework.
 *
 * Collects community feedback on cognitive models, scores their salience,
 * integrates with GitHub Copilot, and broadcasts refinements to the community.
 */

import {
  HypergraphSchemeCore,
  HypergraphNode,
} from "./hypergraphSchemeCore";
import { getConfig, getEvolutionaryConfig } from "../evolutionaryConfig";

// --- Interfaces ---

export interface ProjectModel {
  id: string;
  name: string;
  description: string;
  version: string;
  lastModified: Date;
  usageCount: number;
  tags?: string[];
  communityFeedback?: CommunityFeedback[];
  salienceScore?: number;
}

export interface CommunityFeedback {
  id?: string;
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
  timestamp?: Date;
  metadata?: Record<string, unknown>;
}

export interface CopilotRequest {
  modelId: string;
  requestType: "optimize" | "refactor" | "document" | "test" | "analyze";
  context: string;
  constraints?: string[];
  targetQuality?: number;
  priority?: number;
  requirements?: string[];
}

export interface CopilotResponse {
  requestId: string;
  modelId: string;
  suggestions: string[];
  codeChanges?: string;
  upgradedContent?: string;
  improvements?: string[];
  version?: string;
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
  feedbackUrgency: number;
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

const URGENCY_WEIGHTS: Record<CommunityFeedback["priority"], number> = {
  urgent: 1.0,
  critical: 0.95,
  high: 0.8,
  medium: 0.6,
  low: 0.3,
};

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
  private feedbackIntervalId: ReturnType<typeof setInterval> | null = null;
  private cognitiveLoad = 0;
  private recentActivity = 0.7;
  private feedbackUrgency = 0.5;

  private constructor() {
    this.hypergraphCore = new HypergraphSchemeCore();
    this.thresholds.attentionThreshold = getConfig("attentionThreshold");
    this.cognitiveLoad = getConfig("cognitiveLoadFactor");

    getEvolutionaryConfig().subscribe("attentionThreshold", value => {
      this.thresholds.attentionThreshold = value;
    });
    getEvolutionaryConfig().subscribe("cognitiveLoadFactor", value => {
      this.cognitiveLoad = value;
    });

    this.initializeFeedbackLoop();
  }

  public static getInstance(): AdaptiveFeedbackService {
    if (!AdaptiveFeedbackService.instance) {
      AdaptiveFeedbackService.instance = new AdaptiveFeedbackService();
    }
    return AdaptiveFeedbackService.instance;
  }

  private initializeFeedbackLoop(): void {
    this.hypergraphCore.createNode(
      "feedback-collector",
      "procedure",
      {
        description: "Collects salient project models and community feedback",
        priority: 0.9,
      },
      []
    );

    this.hypergraphCore.createNode(
      "salience-scorer",
      "procedure",
      {
        description: "Scores models using semantic salience",
        priority: 0.85,
      },
      ["feedback-collector"]
    );

    this.hypergraphCore.createNode(
      "copilot-interface",
      "procedure",
      {
        description: "Queries Copilot with prioritized wishlist",
        priority: 0.8,
      },
      ["salience-scorer"]
    );

    this.hypergraphCore.createNode(
      "model-integrator",
      "procedure",
      {
        description: "Integrates upgrades into local repository",
        priority: 0.75,
      },
      ["copilot-interface"]
    );

    this.hypergraphCore.createNode(
      "community-broadcaster",
      "procedure",
      {
        description: "Broadcasts improvements to community",
        priority: 0.7,
      },
      ["model-integrator"]
    );
  }

  // --- Model management ---

  public registerProjectModel(model: ProjectModel): void {
    const completeModel: ProjectModel = {
      ...model,
      communityFeedback: model.communityFeedback ?? [],
      salienceScore: model.salienceScore ?? 0.5,
    };
    this.projectModels.set(model.id, completeModel);

    this.hypergraphCore.createNode(
      model.id,
      "model",
      {
        name: model.name,
        description: model.description,
        version: model.version,
        usageCount: model.usageCount,
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
    const completeFeedback: CommunityFeedback = {
      ...feedback,
      id:
        feedback.id ??
        `feedback-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      timestamp: feedback.timestamp ?? new Date(),
      metadata: feedback.metadata ?? {},
    };
    this.communityFeedback.push(completeFeedback);

    const model = this.projectModels.get(feedback.modelId);
    if (model) {
      model.communityFeedback = [
        ...(model.communityFeedback ?? []),
        completeFeedback,
      ];
    }

    if (feedback.priority === "urgent" || feedback.priority === "critical") {
      this.feedbackUrgency = Math.min(this.feedbackUrgency + 0.2, 1.0);
    }
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

    const feedback = this.getFeedbackForModel(request.modelId);
    const suggestions = feedback
      .filter(
        f =>
          f.priority === "high" ||
          f.priority === "critical" ||
          f.priority === "urgent"
      )
      .slice(0, 3)
      .map(f => f.description);

    const response: CopilotResponse = {
      requestId,
      modelId: request.modelId,
      suggestions:
        suggestions.length > 0
          ? suggestions
          : [
              `Optimize ${model?.name ?? "model"} based on ${request.requestType} analysis`,
            ],
      qualityScore: 0.75,
      confidence: 0.8,
      processingTime: Date.now() - processingStart,
    };

    this.copilotRequests.delete(requestId);
    return response;
  }

  // --- Feedback loop ---

  public async triggerFeedbackLoop(): Promise<void> {
    return this.executeFeedbackLoop();
  }

  public async executeFeedbackLoop(): Promise<void> {
    if (this.isRunning) {
      console.warn("Feedback loop already running, skipping...");
      return;
    }

    this.isRunning = true;
    console.log("🔄 Starting adaptive feedback loop cycle...");

    try {
      this.updateAdaptiveThresholds();

      const salientModels = await this.collectSalientModels();
      const scoredModels = this.scoreModelsSalience(salientModels);
      const prioritizedRequests = this.buildCopilotWishlist(scoredModels);
      const copilotResponses = await this.queryCopilot(prioritizedRequests);
      const integratedModels = await this.integrateUpgrades(copilotResponses);
      await this.broadcastImprovements(integratedModels);
      this.updateHypergraphPatterns();

      console.log(
        `✅ Feedback loop completed. Processed ${scoredModels.length} models, integrated ${integratedModels.length} upgrades`
      );
    } catch (error) {
      console.error("❌ Error in feedback loop execution:", error);
    } finally {
      this.lastFeedbackLoopRun = new Date();
      this.isRunning = false;
    }
  }

  private updateAdaptiveThresholds(): void {
    const currentLoad = this.calculateCognitiveLoad();
    const recentActivity = this.calculateRecentActivity();
    const newThreshold = this.hypergraphCore.adaptiveAttention(
      currentLoad,
      recentActivity
    );
    this.thresholds.attentionThreshold = newThreshold;
    this.hypergraphCore.updateAttentionThreshold(newThreshold);

    console.log(
      `🧠 Adaptive thresholds updated: load=${currentLoad.toFixed(3)}, activity=${recentActivity.toFixed(3)}, threshold=${newThreshold.toFixed(3)}`
    );
  }

  private calculateCognitiveLoad(): number {
    const activeNodes = this.hypergraphCore.getAllNodes().length;
    const feedbackVolume = this.communityFeedback.length;
    const processingQueue = this.projectModels.size;
    const load = Math.min(
      (activeNodes + feedbackVolume + processingQueue) / 100,
      1.0
    );
    this.cognitiveLoad = load;
    return load;
  }

  private calculateRecentActivity(): number {
    const now = Date.now();
    const hourAgo = now - 60 * 60 * 1000;

    const recentFeedback = this.communityFeedback.filter(feedback => {
      const ts = feedback.timestamp?.getTime() ?? now;
      return ts > hourAgo;
    }).length;

    const recentModels = Array.from(this.projectModels.values()).filter(
      model => model.lastModified.getTime() > hourAgo
    ).length;

    const activity = Math.min((recentFeedback + recentModels) / 10, 1.0);
    this.recentActivity = activity;
    return activity;
  }

  private async collectSalientModels(): Promise<ProjectModel[]> {
    const attentionNodes = this.hypergraphCore.getAttentionFilteredNodes(
      this.thresholds.attentionThreshold
    );

    const salientModels: ProjectModel[] = [];

    for (const node of attentionNodes) {
      if (node.type === "model" || node.type === "concept") {
        const model = this.nodeToProjectModel(node);
        if (model) {
          salientModels.push(model);
        }
      }
    }

    salientModels.push(...this.getModelsWithUrgentFeedback());

    const uniqueModels = salientModels.filter(
      (model, index, self) => index === self.findIndex(m => m.id === model.id)
    );

    console.log(
      `📊 Collected ${uniqueModels.length} salient models (threshold: ${this.thresholds.attentionThreshold.toFixed(3)})`
    );

    return uniqueModels;
  }

  private nodeToProjectModel(node: HypergraphNode): ProjectModel | null {
    const existing = this.projectModels.get(node.id);
    if (existing) return existing;

    try {
      return {
        id: node.id,
        name: node.content.name || node.id,
        description:
          node.content.description || "Generated from hypergraph node",
        version: node.content.version || "1.0.0",
        lastModified: node.lastUpdated,
        usageCount: node.content.usageCount || 0,
        communityFeedback: this.getFeedbackForModel(node.id),
        salienceScore: node.salience,
      };
    } catch (error) {
      console.warn(
        `Warning: Failed to convert node ${node.id} to project model:`,
        error
      );
      return null;
    }
  }

  private getModelsWithUrgentFeedback(): ProjectModel[] {
    const urgentModelIds = Array.from(
      new Set(
        this.communityFeedback
          .filter(f => f.priority === "urgent" || f.priority === "critical")
          .map(f => f.modelId)
      )
    );

    return urgentModelIds
      .map(id => this.projectModels.get(id))
      .filter((model): model is ProjectModel => model !== undefined);
  }

  private scoreModelsSalience(models: ProjectModel[]): ProjectModel[] {
    return models
      .map(model => {
        const feedback = this.getFeedbackForModel(model.id);
        const metrics = this.hypergraphCore.calculateSalienceMetrics(
          model.id,
          feedback.map(f => ({
            priority: f.priority,
            votes: f.votes,
            type: f.type,
          }))
        );
        const feedbackUrgency = this.calculateFeedbackUrgency(model);
        const combinedScore =
          metrics.demand * 0.4 +
          metrics.freshness * 0.3 +
          feedbackUrgency * 0.3;
        model.salienceScore = combinedScore;
        return model;
      })
      .sort((a, b) => (b.salienceScore ?? 0) - (a.salienceScore ?? 0));
  }

  private calculateFeedbackUrgency(model: ProjectModel): number {
    const feedback = model.communityFeedback ?? this.getFeedbackForModel(model.id);
    if (feedback.length === 0) return 0;

    const weightedUrgency =
      feedback.reduce((sum, item) => {
        return sum + URGENCY_WEIGHTS[item.priority] * (1 + item.votes * 0.1);
      }, 0) / feedback.length;

    return Math.min(weightedUrgency, 1.0);
  }

  private buildCopilotWishlist(scoredModels: ProjectModel[]): CopilotRequest[] {
    const maxRequests = Math.round(getConfig("maxCopilotRequests")) || 5;

    return scoredModels.slice(0, maxRequests).map(model => ({
      modelId: model.id,
      priority: model.salienceScore,
      requestType: this.determineRequestType(model),
      context: this.buildModelContext(model),
      requirements: this.extractRequirements(model),
    }));
  }

  private determineRequestType(model: ProjectModel): CopilotRequest["requestType"] {
    const feedbackTypes = (
      model.communityFeedback ?? this.getFeedbackForModel(model.id)
    ).map(f => f.type);

    if (feedbackTypes.includes("performance")) return "optimize";
    if (feedbackTypes.includes("feature_request") || feedbackTypes.includes("feature"))
      return "analyze";
    if (feedbackTypes.includes("bug")) return "refactor";
    return "optimize";
  }

  private buildModelContext(model: ProjectModel): string {
    const feedback = (
      model.communityFeedback ?? this.getFeedbackForModel(model.id)
    )
      .map(f => `${f.type}: ${f.description}`)
      .join("; ");

    return `Model: ${model.name} (v${model.version})\nDescription: ${model.description}\nFeedback: ${feedback}`;
  }

  private extractRequirements(model: ProjectModel): string[] {
    return (model.communityFeedback ?? this.getFeedbackForModel(model.id))
      .filter(
        f =>
          f.priority === "high" ||
          f.priority === "urgent" ||
          f.priority === "critical"
      )
      .map(f => f.description);
  }

  private async queryCopilot(
    requests: CopilotRequest[]
  ): Promise<CopilotResponse[]> {
    console.log(
      `🤖 Querying Copilot with ${requests.length} prioritized requests...`
    );

    const responses: CopilotResponse[] = [];

    for (const request of requests) {
      const requestId = this.submitCopilotRequest(request);
      const processed = await this.processCopilotRequest(requestId);
      if (processed) {
        processed.upgradedContent = this.generateMockUpgrade(request);
        processed.improvements = this.generateMockImprovements(request);
        processed.version = this.incrementVersion(request.modelId);
        responses.push(processed);
      }
    }

    console.log(`✨ Received ${responses.length} Copilot responses`);
    return responses;
  }

  private generateMockUpgrade(request: CopilotRequest): string {
    return `
// Enhanced ${request.modelId} - Generated by Copilot
// Request Type: ${request.requestType}

// Improvements based on requirements:
${(request.requirements ?? []).map(req => `// - ${req}`).join("\n")}

// Mock implementation follows hypergraph-encoded patterns
(define (enhanced-${request.modelId.replace(/[^a-zA-Z0-9]/g, "-")} context)
  ;; Enhanced cognitive processing with improved salience
  (let ((processed-context (apply-salience-filter context)))
    (hypergraph-encode processed-context)))
`;
  }

  private generateMockImprovements(request: CopilotRequest): string[] {
    const baseImprovements = [
      "Enhanced hypergraph pattern encoding",
      "Improved adaptive attention allocation",
      "Optimized semantic salience calculation",
      "Better integration with cognitive framework",
    ];

    const typeSpecificImprovements: Record<
      CopilotRequest["requestType"],
      string[]
    > = {
      optimize: ["Reduced cognitive load", "Faster pattern recognition"],
      refactor: ["Updated core algorithms", "Enhanced performance metrics"],
      document: ["Expanded model documentation", "Clearer salience notes"],
      test: ["Added salience regression coverage", "Validated attention filters"],
      analyze: [
        "New community feedback integration",
        "Enhanced broadcasting capabilities",
      ],
    };

    return [
      ...baseImprovements,
      ...typeSpecificImprovements[request.requestType],
    ];
  }

  private incrementVersion(modelId: string): string {
    const model = this.projectModels.get(modelId);
    if (!model) return "1.0.1";

    const [major, minor, patch] = model.version.split(".").map(Number);
    return `${major}.${minor}.${(patch || 0) + 1}`;
  }

  private async integrateUpgrades(
    responses: CopilotResponse[]
  ): Promise<ProjectModel[]> {
    console.log(`🔧 Integrating ${responses.length} Copilot upgrades...`);

    const integratedModels: ProjectModel[] = [];

    for (const response of responses) {
      try {
        const existingModel = this.projectModels.get(response.modelId);
        const upgradedModel: ProjectModel = {
          id: response.modelId,
          name: existingModel?.name || response.modelId,
          description: `${existingModel?.description || "Model"} - Enhanced by Copilot`,
          version: response.version || existingModel?.version || "1.0.1",
          lastModified: new Date(),
          usageCount: existingModel?.usageCount || 0,
          communityFeedback: existingModel?.communityFeedback || [],
          salienceScore: existingModel?.salienceScore || 0.5,
        };

        this.projectModels.set(response.modelId, upgradedModel);

        this.hypergraphCore.createNode(
          `${response.modelId}-v${upgradedModel.version}`,
          "model",
          {
            content: response.upgradedContent,
            improvements: response.improvements,
            confidence: response.confidence,
            version: upgradedModel.version,
          },
          [response.modelId]
        );

        integratedModels.push(upgradedModel);
      } catch (error) {
        console.error(`❌ Failed to integrate ${response.modelId}:`, error);
      }
    }

    return integratedModels;
  }

  private async broadcastImprovements(models: ProjectModel[]): Promise<void> {
    console.log(
      `📡 Broadcasting ${models.length} model improvements to community...`
    );

    for (const model of models) {
      const broadcastMessage = {
        type: "model_improvement",
        modelId: model.id,
        version: model.version,
        timestamp: new Date(),
        improvements: this.extractImprovements(model),
        salienceScore: model.salienceScore,
      };

      for (const node of ["project-alpha", "project-beta", "community-hub"]) {
        await this.sendBroadcast(node, broadcastMessage);
      }
    }
  }

  private extractImprovements(model: ProjectModel): string[] {
    const node = this.hypergraphCore.getNode(`${model.id}-v${model.version}`);
    return (
      node?.content?.improvements || ["General improvements and optimizations"]
    );
  }

  private async sendBroadcast(
    nodeId: string,
    message: { type: string; modelId: string }
  ): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 0));
    console.log(
      `📤 Broadcast sent to ${nodeId}: ${message.type} for ${message.modelId}`
    );
  }

  private updateHypergraphPatterns(): void {
    const newPatterns = this.hypergraphCore.mineCognitivePatterns(
      this.thresholds.salienceThreshold
    );

    newPatterns
      .filter(pattern => pattern.strength > 0.8)
      .forEach(pattern => {
        this.hypergraphCore.embodyPattern(pattern);
      });

    const highSalienceNodes = this.hypergraphCore
      .getAllNodes()
      .filter(node => node.salience > 0.8);

    highSalienceNodes.forEach(node => {
      this.hypergraphCore.spreadAttention(node.id, 0.1);
    });

    console.log(
      `🧩 Updated hypergraph: ${newPatterns.length} new patterns, ${highSalienceNodes.length} attention spreads`
    );
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
    const hypergraphNodesCount = this.hypergraphCore.getAllNodes().length;

    const attentionFiltered = this.hypergraphCore.getAttentionFilteredNodes(
      this.thresholds.attentionThreshold
    ).length;

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
        cognitiveLoad: this.cognitiveLoad,
        recentActivity: this.recentActivity,
        feedbackUrgency: this.feedbackUrgency,
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
      this.feedbackIntervalId = setInterval(() => {
        this.triggerFeedbackLoop().catch(err =>
          console.error("Feedback loop error:", err)
        );
      }, Math.max(intervalMs, 30000));
    }
  }
}
