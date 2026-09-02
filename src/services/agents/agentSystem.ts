/**
 * Distributed Agent System
 * ========================
 *
 * Multi-agent orchestration system for echogenesis.
 * Implements the AAR (Agent-Arena-Relation) pattern for
 * cognitive load sharing and distributed processing.
 *
 * @author Deep Tree Echo
 * @date June 2026
 */

import { EventEmitter } from "events";
import { v4 as uuidv4 } from "uuid";

// Types
export type AgentRole =
  | "perceiver"
  | "reasoner"
  | "actor"
  | "memory"
  | "attention"
  | "meta"
  | "embodiment"
  | "wisdom";

export type AgentStatus =
  | "idle"
  | "active"
  | "processing"
  | "waiting"
  | "error"
  | "terminated";

export interface AgentMessage {
  id: string;
  from: string;
  to: string | "broadcast";
  type: string;
  payload: any;
  timestamp: Date;
  priority: number;
  replyTo?: string;
}

export interface AgentCapability {
  name: string;
  description: string;
  inputTypes: string[];
  outputTypes: string[];
}

export interface AgentMetrics {
  messagesReceived: number;
  messagesSent: number;
  tasksCompleted: number;
  averageLatency: number;
  uptime: number;
  cognitiveLoad: number;
}

export interface Agent {
  id: string;
  name: string;
  role: AgentRole;
  status: AgentStatus;
  capabilities: AgentCapability[];
  metrics: AgentMetrics;
  config: Record<string, any>;
}

export interface ArenaConfig {
  maxAgents: number;
  messageBufferSize: number;
  coordinationMode: "centralized" | "distributed" | "hybrid";
  loadBalancing: boolean;
  faultTolerance: boolean;
}

/**
 * CognitiveAgent
 *
 * Base class for cognitive agents.
 */
export class CognitiveAgent extends EventEmitter implements Agent {
  id: string;
  name: string;
  role: AgentRole;
  status: AgentStatus = "idle";
  capabilities: AgentCapability[] = [];
  metrics: AgentMetrics;
  config: Record<string, any>;

  private messageQueue: AgentMessage[] = [];
  private startTime: Date;
  private arena: CognitiveArena | null = null;

  constructor(
    name: string,
    role: AgentRole,
    capabilities: AgentCapability[] = [],
    config: Record<string, any> = {}
  ) {
    super();

    this.id = uuidv4();
    this.name = name;
    this.role = role;
    this.capabilities = capabilities;
    this.config = config;
    this.startTime = new Date();

    this.metrics = {
      messagesReceived: 0,
      messagesSent: 0,
      tasksCompleted: 0,
      averageLatency: 0,
      uptime: 0,
      cognitiveLoad: 0,
    };
  }

  /**
   * Register with arena
   */
  registerArena(arena: CognitiveArena): void {
    this.arena = arena;
    this.emit("registered", arena);
  }

  /**
   * Receive message
   */
  receiveMessage(message: AgentMessage): void {
    this.messageQueue.push(message);
    this.metrics.messagesReceived++;
    this.emit("message:received", message);

    // Auto-process if idle
    if (this.status === "idle") {
      this.processNextMessage();
    }
  }

  /**
   * Process next message
   */
  async processNextMessage(): Promise<void> {
    if (this.messageQueue.length === 0) {
      this.status = "idle";
      return;
    }

    this.status = "processing";
    const message = this.messageQueue.shift()!;

    try {
      const startTime = Date.now();
      const response = await this.handleMessage(message);
      const latency = Date.now() - startTime;

      // Update metrics
      this.metrics.tasksCompleted++;
      this.metrics.averageLatency =
        (this.metrics.averageLatency * (this.metrics.tasksCompleted - 1) +
          latency) /
        this.metrics.tasksCompleted;

      // Send response if needed
      if (response && message.replyTo) {
        this.sendMessage(message.from, "response", response, message.id);
      }

      this.emit("message:processed", { message, response, latency });
    } catch (error) {
      this.emit("error", { message, error });
    }

    // Process next
    this.processNextMessage();
  }

  /**
   * Handle message - override in subclasses
   */
  protected async handleMessage(message: AgentMessage): Promise<any> {
    // Default implementation - echo
    return { received: message.payload, handler: this.name };
  }

  /**
   * Send message
   */
  sendMessage(
    to: string | "broadcast",
    type: string,
    payload: any,
    replyTo?: string
  ): AgentMessage {
    const message: AgentMessage = {
      id: uuidv4(),
      from: this.id,
      to,
      type,
      payload,
      timestamp: new Date(),
      priority: 1,
      replyTo,
    };

    this.metrics.messagesSent++;

    if (this.arena) {
      this.arena.routeMessage(message);
    }

    this.emit("message:sent", message);
    return message;
  }

  /**
   * Broadcast message
   */
  broadcast(type: string, payload: any): AgentMessage {
    return this.sendMessage("broadcast", type, payload);
  }

  /**
   * Update cognitive load
   */
  updateCognitiveLoad(load: number): void {
    this.metrics.cognitiveLoad = Math.max(0, Math.min(1, load));
    this.emit("load:updated", this.metrics.cognitiveLoad);
  }

  /**
   * Get uptime
   */
  getUptime(): number {
    return Date.now() - this.startTime.getTime();
  }

  /**
   * Terminate agent
   */
  terminate(): void {
    this.status = "terminated";
    this.messageQueue = [];
    this.emit("terminated");
  }

  /**
   * Get state
   */
  getState(): Agent {
    return {
      id: this.id,
      name: this.name,
      role: this.role,
      status: this.status,
      capabilities: this.capabilities,
      metrics: {
        ...this.metrics,
        uptime: this.getUptime(),
      },
      config: this.config,
    };
  }
}

/**
 * CognitiveArena
 *
 * The arena where agents interact and coordinate.
 */
export class CognitiveArena extends EventEmitter {
  private agents: Map<string, CognitiveAgent> = new Map();
  private roleIndex: Map<AgentRole, Set<string>> = new Map();
  private messageHistory: AgentMessage[] = [];
  private config: ArenaConfig;

  constructor(config: Partial<ArenaConfig> = {}) {
    super();

    this.config = {
      maxAgents: config.maxAgents || 100,
      messageBufferSize: config.messageBufferSize || 10000,
      coordinationMode: config.coordinationMode || "hybrid",
      loadBalancing: config.loadBalancing ?? true,
      faultTolerance: config.faultTolerance ?? true,
    };

    // Initialize role index
    const roles: AgentRole[] = [
      "perceiver",
      "reasoner",
      "actor",
      "memory",
      "attention",
      "meta",
      "embodiment",
      "wisdom",
    ];
    for (const role of roles) {
      this.roleIndex.set(role, new Set());
    }
  }

  /**
   * Register agent
   */
  registerAgent(agent: CognitiveAgent): boolean {
    if (this.agents.size >= this.config.maxAgents) {
      this.emit("error", { type: "max_agents_reached", agent });
      return false;
    }

    this.agents.set(agent.id, agent);
    this.roleIndex.get(agent.role)?.add(agent.id);
    agent.registerArena(this);

    this.emit("agent:registered", agent);
    return true;
  }

  /**
   * Unregister agent
   */
  unregisterAgent(agentId: string): boolean {
    const agent = this.agents.get(agentId);
    if (!agent) return false;

    this.roleIndex.get(agent.role)?.delete(agentId);
    this.agents.delete(agentId);
    agent.terminate();

    this.emit("agent:unregistered", agentId);
    return true;
  }

  /**
   * Get agent
   */
  getAgent(agentId: string): CognitiveAgent | undefined {
    return this.agents.get(agentId);
  }

  /**
   * Get agents by role
   */
  getAgentsByRole(role: AgentRole): CognitiveAgent[] {
    const ids = this.roleIndex.get(role) || new Set();
    return Array.from(ids)
      .map(id => this.agents.get(id))
      .filter((a): a is CognitiveAgent => a !== undefined);
  }

  /**
   * Route message
   */
  routeMessage(message: AgentMessage): void {
    // Store in history
    this.messageHistory.push(message);
    if (this.messageHistory.length > this.config.messageBufferSize) {
      this.messageHistory.shift();
    }

    // Route
    if (message.to === "broadcast") {
      // Broadcast to all except sender
      for (const [id, agent] of this.agents) {
        if (id !== message.from) {
          agent.receiveMessage(message);
        }
      }
    } else {
      // Direct message
      const target = this.agents.get(message.to);
      if (target) {
        target.receiveMessage(message);
      } else {
        this.emit("error", { type: "target_not_found", message });
      }
    }

    this.emit("message:routed", message);
  }

  /**
   * Select agent with load balancing
   */
  selectAgent(role: AgentRole): CognitiveAgent | undefined {
    const agents = this.getAgentsByRole(role);

    if (agents.length === 0) return undefined;

    if (!this.config.loadBalancing) {
      return agents[0];
    }

    // Select agent with lowest cognitive load
    return agents.reduce((best, current) =>
      current.metrics.cognitiveLoad < best.metrics.cognitiveLoad
        ? current
        : best
    );
  }

  /**
   * Request processing from role
   */
  async requestProcessing(
    role: AgentRole,
    type: string,
    payload: any
  ): Promise<any> {
    const agent = this.selectAgent(role);
    if (!agent) {
      throw new Error(`No agent available for role: ${role}`);
    }

    return new Promise((resolve, reject) => {
      const messageId = uuidv4();

      const handler = (event: { message: AgentMessage; response: any }) => {
        if (event.message.replyTo === messageId) {
          agent.off("message:processed", handler);
          resolve(event.response);
        }
      };

      agent.on("message:processed", handler);

      // Timeout
      setTimeout(() => {
        agent.off("message:processed", handler);
        reject(new Error("Request timeout"));
      }, 30000);

      // Send request
      const message: AgentMessage = {
        id: messageId,
        from: "arena",
        to: agent.id,
        type,
        payload,
        timestamp: new Date(),
        priority: 1,
        replyTo: messageId,
      };

      agent.receiveMessage(message);
    });
  }

  /**
   * Get arena state
   */
  getState(): {
    agents: Agent[];
    config: ArenaConfig;
    messageCount: number;
  } {
    return {
      agents: Array.from(this.agents.values()).map(a => a.getState()),
      config: this.config,
      messageCount: this.messageHistory.length,
    };
  }

  /**
   * Get metrics
   */
  getMetrics(): {
    totalAgents: number;
    agentsByRole: Record<AgentRole, number>;
    totalMessages: number;
    averageLoad: number;
  } {
    const agentsByRole: Record<AgentRole, number> = {} as any;
    let totalLoad = 0;

    for (const [role, ids] of this.roleIndex) {
      agentsByRole[role] = ids.size;
    }

    for (const agent of this.agents.values()) {
      totalLoad += agent.metrics.cognitiveLoad;
    }

    return {
      totalAgents: this.agents.size,
      agentsByRole,
      totalMessages: this.messageHistory.length,
      averageLoad: this.agents.size > 0 ? totalLoad / this.agents.size : 0,
    };
  }

  /**
   * Shutdown arena
   */
  shutdown(): void {
    for (const agent of this.agents.values()) {
      agent.terminate();
    }
    this.agents.clear();
    this.messageHistory = [];
    this.emit("shutdown");
  }
}

/**
 * EchoProtocol
 *
 * Message protocol for echo synchronization between nodes.
 */
export class EchoProtocol {
  static readonly MESSAGE_TYPES = {
    SYNC: "echo:sync",
    STATE: "echo:state",
    HEARTBEAT: "echo:heartbeat",
    ATTENTION: "echo:attention",
    RELEVANCE: "echo:relevance",
    TRANSFORMATION: "echo:transformation",
  };

  /**
   * Create sync message
   */
  static createSyncMessage(
    source: string,
    echoState: number[],
    timestamp: number
  ): AgentMessage {
    return {
      id: uuidv4(),
      from: source,
      to: "broadcast",
      type: this.MESSAGE_TYPES.SYNC,
      payload: {
        echoState,
        timestamp,
        sequence: Date.now(),
      },
      timestamp: new Date(),
      priority: 2,
    };
  }

  /**
   * Create attention message
   */
  static createAttentionMessage(
    source: string,
    target: string,
    sti: number,
    lti: number
  ): AgentMessage {
    return {
      id: uuidv4(),
      from: source,
      to: target,
      type: this.MESSAGE_TYPES.ATTENTION,
      payload: { sti, lti },
      timestamp: new Date(),
      priority: 2,
    };
  }

  /**
   * Create relevance message
   */
  static createRelevanceMessage(
    source: string,
    target: string,
    relevanceScore: number,
    context: any
  ): AgentMessage {
    return {
      id: uuidv4(),
      from: source,
      to: target,
      type: this.MESSAGE_TYPES.RELEVANCE,
      payload: { relevanceScore, context },
      timestamp: new Date(),
      priority: 1,
    };
  }
}

/**
 * Create default arena with standard agents
 */
export function createDefaultArena(): CognitiveArena {
  const arena = new CognitiveArena();

  // Create standard cognitive agents
  const standardRoles: AgentRole[] = [
    "perceiver",
    "reasoner",
    "actor",
    "memory",
    "attention",
    "meta",
  ];

  for (const role of standardRoles) {
    const agent = new CognitiveAgent(
      `${role}_agent`,
      role,
      [
        {
          name: `${role}_processing`,
          description: `Primary ${role} capability`,
          inputTypes: ["any"],
          outputTypes: ["any"],
        },
      ],
      { autoProcess: true }
    );

    arena.registerAgent(agent);
  }

  return arena;
}

// Module exports
export default {
  CognitiveAgent,
  CognitiveArena,
  EchoProtocol,
  createDefaultArena,
};
