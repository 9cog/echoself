/**
 * Agents Module
 * =============
 *
 * Exports all agent-related components for echogenesis.
 *
 * @author Deep Tree Echo
 * @date June 2026
 */

// Agent system - classes and functions
export {
  CognitiveAgent,
  CognitiveArena,
  EchoProtocol,
  createDefaultArena,
} from "./agentSystem.ts";

// Agent system - types
export type {
  AgentRole,
  AgentStatus,
  AgentMessage,
  AgentCapability,
  AgentMetrics,
  Agent,
  ArenaConfig,
} from "./agentSystem.ts";

// Specialized agents
export {
  PerceptionAgent,
  ReasoningAgent,
  ActionAgent,
  MemoryAgent,
  AttentionAgent,
  MetaAgent,
  createSpecializedAgents,
} from "./specializedAgents.ts";
