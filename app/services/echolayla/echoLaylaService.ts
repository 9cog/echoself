/**
 * EchoLayla Core Service
 *
 * Main service for managing EchoLayla AI assistant functionality,
 * integrating with EchoSelf's memory and cognitive systems.
 *
 * Enhanced with Triple-Loop Learning integration for adaptive
 * character evolution and training cycle linkages.
 */

import type {
  LaylaCharacter,
  ConversationMessage,
  ConversationContext,
  InferenceConfig,
  PrivacySettings,
  AutomationTask,
  TaskType,
} from "./types";
import type {
  ConversationFeedback,
  TrainingCycleConfig,
  LearningCycleResult,
} from "./tripleLoopLearningTypes";
import { getCharacter, getDefaultCharacter } from "./characters";
import { getDefaultAdapter, type AIAdapter } from "./aiIntegration";
import {
  getTripleLoopLearningService,
  type TripleLoopLearningService,
} from "./tripleLoopLearningService";

/**
 * Default inference configuration
 */
const DEFAULT_INFERENCE_CONFIG: InferenceConfig = {
  model: "gpt-3.5-turbo",
  temperature: 0.7,
  maxTokens: 1000,
  topP: 0.9,
  streaming: true,
};

/**
 * Default privacy settings (privacy-first approach)
 */
const DEFAULT_PRIVACY_SETTINGS: PrivacySettings = {
  localProcessingOnly: false,
  dataRetentionDays: 30,
  enableVoiceRecording: false,
  enableVisionCapture: false,
  shareDataWithEcho: true,
};

/**
 * EchoLayla Service Class
 *
 * Enhanced with Triple-Loop Learning integration for adaptive
 * character evolution and training cycle linkages.
 */
export class EchoLaylaService {
  private activeCharacter: LaylaCharacter = "max";
  private currentContext?: ConversationContext;
  private inferenceConfig: InferenceConfig = DEFAULT_INFERENCE_CONFIG;
  private privacySettings: PrivacySettings = DEFAULT_PRIVACY_SETTINGS;
  private tasks: Map<string, AutomationTask> = new Map();
  private aiAdapter: AIAdapter = getDefaultAdapter();
  private tripleLoopService: TripleLoopLearningService | null = null;
  private learningEnabled: boolean = true;

  /**
   * Initialize the service
   */
  async initialize(): Promise<void> {
    console.log("[EchoLayla] Initializing service...");

    // Load saved settings from localStorage if available
    if (typeof window !== "undefined") {
      this.loadSettings();
    }

    // Initialize Triple-Loop Learning after settings so learningEnabled is respected
    if (this.learningEnabled) {
      this.tripleLoopService = getTripleLoopLearningService();
      console.log("[EchoLayla] Triple-Loop Learning service initialized");
    }

    console.log(
      `[EchoLayla] Initialized with character: ${this.activeCharacter}`
    );
  }

  /**
   * Switch to a different character
   */
  setCharacter(characterId: LaylaCharacter): void {
    const character = getCharacter(characterId);
    if (!character) {
      throw new Error(`Unknown character: ${characterId}`);
    }

    this.activeCharacter = characterId;

    // Update system prompt in inference config
    this.inferenceConfig.systemPrompt = character.systemPrompt;

    // Start a new conversation context
    this.startNewContext();

    this.saveSettings();
    console.log(`[EchoLayla] Switched to character: ${character.name}`);
  }

  /**
   * Get current active character
   */
  getActiveCharacter(): LaylaCharacter {
    return this.activeCharacter;
  }

  /**
   * Get current character profile
   */
  getActiveCharacterProfile() {
    return getCharacter(this.activeCharacter) || getDefaultCharacter();
  }

  /**
   * Start a new conversation context
   */
  startNewContext(): ConversationContext {
    this.currentContext = {
      id: this.generateId(),
      character: this.activeCharacter,
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
      metadata: {},
    };

    return this.currentContext;
  }

  /**
   * Get current conversation context
   */
  getContext(): ConversationContext | undefined {
    return this.currentContext;
  }

  /**
   * Add a message to the current context
   */
  addMessage(
    role: "user" | "assistant" | "system",
    content: string,
    metadata?: Record<string, unknown>
  ): ConversationMessage {
    if (!this.currentContext) {
      this.startNewContext();
    }

    const message: ConversationMessage = {
      id: this.generateId(),
      role,
      content,
      character: this.activeCharacter,
      mode: "text",
      timestamp: new Date(),
      metadata,
    };

    this.currentContext!.messages.push(message);
    this.currentContext!.updatedAt = new Date();

    return message;
  }

  /**
   * Send a message and get AI response
   */
  async sendMessage(userMessage: string): Promise<ConversationMessage> {
    // Add user message
    this.addMessage("user", userMessage);

    // TODO: Integrate with actual AI inference service
    // For now, return a placeholder response
    const response = await this.generateResponse(userMessage);

    return this.addMessage("assistant", response);
  }

  /**
   * Generate AI response using AI adapter
   */
  private async generateResponse(_userMessage: string): Promise<string> {
    const character = this.getActiveCharacterProfile();

    if (!this.currentContext) {
      this.startNewContext();
    }

    try {
      // Build messages for AI
      const messages = [
        {
          role: "system",
          content: this.inferenceConfig.systemPrompt || character.systemPrompt,
        },
        ...this.currentContext!.messages.map(msg => ({
          role: msg.role === "assistant" ? "assistant" : "user",
          content: msg.content,
        })),
      ];

      // Generate response
      const response = await this.aiAdapter.generate(
        messages,
        this.inferenceConfig
      );

      // Update token usage in context metadata
      if (response.usage && this.currentContext) {
        this.currentContext.metadata.totalTokens = response.usage.totalTokens;
      }

      return response.content;
    } catch (error) {
      console.error("[EchoLayla] Error generating response:", error);
      return `[${character.name}] I apologize, but I encountered an error processing your request. Please try again.`;
    }
  }

  /**
   * Set AI adapter (for testing or custom providers)
   */
  setAIAdapter(adapter: AIAdapter): void {
    this.aiAdapter = adapter;
  }

  /**
   * Create an automation task
   */
  createTask(
    type: TaskType,
    description: string,
    input: unknown
  ): AutomationTask {
    const task: AutomationTask = {
      id: this.generateId(),
      type,
      description,
      status: "pending",
      character: this.activeCharacter,
      input,
      createdAt: new Date(),
    };

    this.tasks.set(task.id, task);

    // Start processing task asynchronously
    this.processTask(task.id).catch(console.error);

    return task;
  }

  /**
   * Process an automation task
   */
  private async processTask(taskId: string): Promise<void> {
    const task = this.tasks.get(taskId);
    if (!task) return;

    task.status = "processing";

    try {
      // TODO: Implement actual task processing logic
      // This would dispatch to different handlers based on task.type

      // Placeholder processing
      await new Promise(resolve => setTimeout(resolve, 1000));

      task.status = "completed";
      task.completedAt = new Date();
      task.output = { result: "Task completed successfully" };
    } catch (error) {
      task.status = "failed";
      task.output = { error: String(error) };
    }
  }

  /**
   * Get task by ID
   */
  getTask(taskId: string): AutomationTask | undefined {
    return this.tasks.get(taskId);
  }

  /**
   * Get all tasks
   */
  getAllTasks(): AutomationTask[] {
    return Array.from(this.tasks.values());
  }

  /**
   * Update inference configuration
   */
  setInferenceConfig(config: Partial<InferenceConfig>): void {
    this.inferenceConfig = { ...this.inferenceConfig, ...config };
    this.saveSettings();
  }

  /**
   * Get current inference configuration
   */
  getInferenceConfig(): InferenceConfig {
    return { ...this.inferenceConfig };
  }

  /**
   * Update privacy settings
   */
  setPrivacySettings(settings: Partial<PrivacySettings>): void {
    this.privacySettings = { ...this.privacySettings, ...settings };
    this.saveSettings();
  }

  /**
   * Get current privacy settings
   */
  getPrivacySettings(): PrivacySettings {
    return { ...this.privacySettings };
  }

  /**
   * Save settings to localStorage
   */
  private saveSettings(): void {
    if (typeof window === "undefined") return;

    const settings = {
      activeCharacter: this.activeCharacter,
      inferenceConfig: this.inferenceConfig,
      privacySettings: this.privacySettings,
      learningEnabled: this.learningEnabled,
    };

    localStorage.setItem("echolayla:settings", JSON.stringify(settings));
  }

  /**
   * Load settings from localStorage
   */
  private loadSettings(): void {
    if (typeof window === "undefined") return;

    const saved = localStorage.getItem("echolayla:settings");
    if (!saved) return;

    try {
      const settings = JSON.parse(saved);

      if (settings.activeCharacter) {
        this.activeCharacter = settings.activeCharacter;
      }

      if (settings.inferenceConfig) {
        this.inferenceConfig = {
          ...DEFAULT_INFERENCE_CONFIG,
          ...settings.inferenceConfig,
        };
      }

      if (settings.privacySettings) {
        this.privacySettings = {
          ...DEFAULT_PRIVACY_SETTINGS,
          ...settings.privacySettings,
        };
      }

      if (settings.learningEnabled !== undefined) {
        this.learningEnabled = settings.learningEnabled;
      }
    } catch (error) {
      console.error("[EchoLayla] Failed to load settings:", error);
    }
  }

  /**
   * Generate a unique ID
   */
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
  }

  // ==========================================
  // Triple-Loop Learning Integration Methods
  // ==========================================

  /**
   * Enable or disable learning
   */
  setLearningEnabled(enabled: boolean): void {
    this.learningEnabled = enabled;
    if (enabled && !this.tripleLoopService) {
      this.tripleLoopService = getTripleLoopLearningService();
    } else if (!enabled && this.tripleLoopService) {
      this.tripleLoopService.stopLearningCycle();
    }
    this.saveSettings();
  }

  /**
   * Check if learning is enabled
   */
  isLearningEnabled(): boolean {
    return this.learningEnabled;
  }

  /**
   * Record user feedback for a message (triggers learning)
   */
  recordFeedback(messageId: string, rating: number, comment?: string): void {
    if (!this.tripleLoopService || !this.currentContext) return;

    const message = this.currentContext.messages.find(m => m.id === messageId);
    if (!message) return;

    const feedback: Partial<ConversationFeedback> = {
      messageId,
      characterId: message.character || this.activeCharacter,
      userFeedback: {
        rating,
        comment,
      },
      implicitSignals: {
        responseAcceptance: rating >= 3,
        followUpDepth: this.currentContext.messages.length,
        contextRetention: this.calculateContextRetention(),
      },
      learningOpportunity: {
        loopLevel: rating < 2 ? "double" : rating < 3 ? "single" : "single",
        suggestedAction:
          rating < 2
            ? "Significant strategy revision needed"
            : rating < 4
              ? "Minor adjustment to response style"
              : "Reinforce current approach",
        priority: rating < 2 ? "high" : rating < 4 ? "medium" : "low",
      },
    };

    this.tripleLoopService.recordConversationFeedback(message, feedback);
    console.log(
      `[EchoLayla] Recorded feedback for message ${messageId}: rating=${rating}`
    );
  }

  /**
   * Calculate context retention score
   */
  private calculateContextRetention(): number {
    if (!this.currentContext) return 0;
    const messageCount = this.currentContext.messages.length;
    // Higher retention for longer conversations
    return Math.min(1, messageCount / 10);
  }

  /**
   * Get learning state summary
   */
  getLearningState(): {
    enabled: boolean;
    activeLoopLevel: string;
    cycleMetrics: object;
    characterProfile?: object;
  } | null {
    if (!this.tripleLoopService) return null;

    const state = this.tripleLoopService.getLearningState();
    const profile = this.tripleLoopService.getCharacterProfile(
      this.activeCharacter
    );

    return {
      enabled: this.learningEnabled,
      activeLoopLevel: state.activeLoopLevel,
      cycleMetrics: state.cycleMetrics,
      characterProfile: profile,
    };
  }

  /**
   * Get training cycle configuration for NanEcho integration
   */
  getTrainingCycleConfig(): TrainingCycleConfig | null {
    if (!this.tripleLoopService) return null;
    return this.tripleLoopService.generateTrainingCycleConfig();
  }

  /**
   * Manually trigger a learning cycle
   */
  async triggerLearningCycle(): Promise<LearningCycleResult | null> {
    if (!this.tripleLoopService) return null;
    return this.tripleLoopService.executeLearningCycle();
  }

  /**
   * Get loop statistics for monitoring
   */
  getLoopStatistics(): object | null {
    if (!this.tripleLoopService) return null;
    return this.tripleLoopService.getLoopStatistics();
  }

  /**
   * Get the triple-loop learning service instance
   */
  getTripleLoopService(): TripleLoopLearningService | null {
    return this.tripleLoopService;
  }
}

/**
 * Singleton instance
 */
let echoLaylaInstance: EchoLaylaService | null = null;

/**
 * Get or create the EchoLayla service instance
 */
export function getEchoLaylaService(): EchoLaylaService {
  if (!echoLaylaInstance) {
    echoLaylaInstance = new EchoLaylaService();
  }
  return echoLaylaInstance;
}
