/**
 * EchoLayla Type Definitions
 *
 * Type system for the EchoLayla integration - bridging Layla AI assistant
 * capabilities with EchoSelf's cognitive architecture.
 */

/**
 * Character personalities available in EchoLayla
 */
export type LaylaCharacter = "akiko" | "isabella" | "kaito" | "max" | "ruby";

/**
 * Character profile with personality traits
 */
export interface CharacterProfile {
  id: LaylaCharacter;
  name: string;
  description: string;
  traits: string[];
  systemPrompt: string;
  voiceId?: string;
  avatarUrl?: string;
}

/**
 * Multi-modal interaction types
 */
export type InteractionMode = "text" | "voice" | "vision" | "multimodal";

/**
 * Conversation message structure
 */
export interface ConversationMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  character?: LaylaCharacter;
  mode: InteractionMode;
  timestamp: Date;
  metadata?: Record<string, unknown>;
}

/**
 * Conversation context for multi-turn interactions
 */
export interface ConversationContext {
  id: string;
  character: LaylaCharacter;
  messages: ConversationMessage[];
  createdAt: Date;
  updatedAt: Date;
  metadata: {
    totalTokens?: number;
    mood?: string;
    topics?: string[];
  };
}

/**
 * AI inference configuration
 */
export interface InferenceConfig {
  model: string;
  temperature: number;
  maxTokens: number;
  topP: number;
  topK?: number;
  streaming: boolean;
  systemPrompt?: string;
}

/**
 * Task automation types
 */
export type TaskType =
  | "remember"
  | "summarize"
  | "analyze"
  | "create"
  | "schedule"
  | "remind";

export interface AutomationTask {
  id: string;
  type: TaskType;
  description: string;
  status: "pending" | "processing" | "completed" | "failed";
  character: LaylaCharacter;
  input: unknown;
  output?: unknown;
  createdAt: Date;
  completedAt?: Date;
}

/**
 * Privacy and security settings
 */
export interface PrivacySettings {
  localProcessingOnly: boolean;
  dataRetentionDays: number;
  enableVoiceRecording: boolean;
  enableVisionCapture: boolean;
  shareDataWithEcho: boolean;
}

/**
 * EchoLayla service state
 */
export interface EchoLaylaState {
  initialized: boolean;
  activeCharacter: LaylaCharacter;
  currentContext?: ConversationContext;
  privacySettings: PrivacySettings;
  inferenceConfig: InferenceConfig;
}
