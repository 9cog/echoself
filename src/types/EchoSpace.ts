/**
 * EchoSpace: Marduk's Agent-Arena-Relation Architecture
 *
 * Implements the recursive blueprint with nested boundaries and relations,
 * where Agent-Arena-Relation forms the fundamental principle.
 */

export interface AgentNamespace {
  id: string;
  type: "actual" | "virtual" | "collective";
  permissions: Permission[];
  arenaAccess: string[]; // Arena namespaces this agent can access
  parentAgent?: string; // For recursive hierarchies
  childAgents?: string[]; // Virtual agents spawned by this agent
}

export interface ArenaNamespace {
  id: string;
  type: "real" | "sandbox" | "shared";
  resources: ArenaResource[];
  simulationCapabilities?: SimulationCapabilities;
  accessControl: AccessControl;
}

export interface Permission {
  action: "read" | "write" | "execute" | "simulate" | "spawn";
  scope: string; // Arena or resource identifier
  level: "full" | "limited" | "observe";
}

export interface ArenaResource {
  id: string;
  type: "memory" | "computation" | "data" | "model";
  status: "available" | "busy" | "locked";
  metadata: Record<string, unknown>;
}

export interface SimulationCapabilities {
  timeControl: boolean;
  causalModeling: boolean;
  resourceSimulation: boolean;
  maxConcurrentSimulations: number;
}

export interface AccessControl {
  allowedAgents: string[];
  restrictedActions: string[];
  quotas: Record<string, number>;
}

/**
 * Agent-Arena-Relation: The fundamental triad
 */
export interface AgentArenaRelation {
  agentId: string;
  arenaId: string;
  relationshipType: "owns" | "operates_in" | "simulates" | "observes";
  permissions: Permission[];
  establishedAt: Date;
  lastActive: Date;
  metadata: {
    purpose?: string;
    constraints?: string[];
    autonomyLevel?: number; // 0-1 scale
  };
}

/**
 * Marduk's Nested Namespaces
 */
export interface MardukNamespaces {
  actualMarduk: AgentNamespace;
  mardukSpace: ArenaNamespace;
  virtualMarduks: AgentNamespace[];
  mardukSandbox: ArenaNamespace;
  mardukMemory: ArenaNamespace;
}

/**
 * EchoCog Namespaces (extension of the same pattern)
 */
export interface EchoCogNamespaces {
  echoCog: AgentNamespace;
  echoSpace: ArenaNamespace;
  virtualEchoes: AgentNamespace[];
  echoSandbox: ArenaNamespace;
  echoMemory: ArenaNamespace;
}

/**
 * Simulation Record for Virtual Agents
 */
export interface SimulationRecord {
  id: string;
  virtualAgentId: string;
  hypothesis: string;
  arenaId: string;
  startTime: Date;
  endTime?: Date;
  results: SimulationResult;
  consensusVote?: ConsensusVote;
}

export interface SimulationResult {
  success: boolean;
  outcome: string;
  metrics: Record<string, number>;
  observations: string[];
  recommendations: string[];
  riskAssessment: {
    level: "low" | "medium" | "high" | "critical";
    factors: string[];
  };
}

export interface ConsensusVote {
  virtualAgentId: string;
  vote: "approve" | "reject" | "modify";
  confidence: number; // 0-1 scale
  reasoning: string;
  modifications?: string[];
}

/**
 * Consensus State for strategy alignment
 */
export interface ConsensusState {
  sessionId: string;
  participatingAgents: string[];
  strategy: string;
  votes: ConsensusVote[];
  status: "pending" | "converged" | "diverged" | "timeout";
  finalDecision?: {
    strategy: string;
    confidence: number;
    dissenting: string[];
  };
  timestamp: Date;
}

/**
 * Memory structures for Agent-Arena coherence
 */
export interface AgentArenaMemory {
  id: string;
  agentId: string;
  arenaId: string;
  memoryType: "simulation" | "action" | "strategy" | "feedback";
  content: string;
  metadata: {
    relations?: AgentArenaRelation[];
    simulations?: SimulationRecord[];
    consensus?: ConsensusState[];
  };
  embedding?: number[];
  createdAt: Date;
  updatedAt: Date;
}

/**
 * Pipeline Result for EchoSpace workflows
 */
export interface PipelineResult {
  success: boolean;
  error?: string;
  result?: {
    simulations: SimulationRecord[];
    consensus: ConsensusState;
    execution?: {
      actionTaken: string;
      outcome: string;
    };
  };
}
