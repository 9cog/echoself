/**
 * Specialized Agents
 * ==================
 *
 * Specialized cognitive agents for echogenesis system.
 * Each agent implements specific cognitive functions.
 *
 * @author Deep Tree Echo
 * @date June 2026
 */

import { CognitiveAgent, AgentMessage, AgentRole } from "./agentSystem.ts";
import { getEchogenesisService } from "../echogenesisService.ts";
import { getCognitiveGripService } from "../cognitiveGripService.ts";
import { getPerspectivalService } from "../perspectivalService.ts";
import { getConfig } from "../evolutionaryConfig.ts";

/**
 * PerceptionAgent
 *
 * Specialized agent for sensory processing and perception.
 */
export class PerceptionAgent extends CognitiveAgent {
  private perspectival = getPerspectivalService();

  constructor(config: Record<string, any> = {}) {
    super(
      "perception_agent",
      "perceiver",
      [
        {
          name: "perceive",
          description: "Process sensory input through cognitive frames",
          inputTypes: ["sensory_data", "multimodal"],
          outputTypes: ["perception_result"],
        },
        {
          name: "frame_switch",
          description: "Switch cognitive perspective frame",
          inputTypes: ["frame_request"],
          outputTypes: ["frame_state"],
        },
        {
          name: "aspect_detection",
          description: "Detect aspects in perceived data",
          inputTypes: ["perception_data"],
          outputTypes: ["aspect_list"],
        },
      ],
      config
    );
  }

  protected async handleMessage(message: AgentMessage): Promise<any> {
    switch (message.type) {
      case "perceive":
        return this.perceive(message.payload);
      case "frame_switch":
        return this.switchFrame(message.payload);
      case "aspect_detect":
        return this.detectAspects(message.payload);
      default:
        return super.handleMessage(message);
    }
  }

  private async perceive(data: Record<string, any>) {
    this.updateCognitiveLoad(0.6);

    try {
      const result = await this.perspectival.perceive(data);

      this.updateCognitiveLoad(0.2);

      return {
        perceived: result.perceived,
        frame: result.frame,
        salience: result.salience,
        aspects: result.aspects,
      };
    } catch (error) {
      this.updateCognitiveLoad(0.1);
      throw error;
    }
  }

  private async switchFrame(request: { frame: string; context?: any }) {
    await this.perspectival.switchFrame(request.frame as any, request.context);
    return this.perspectival.getState();
  }

  private async detectAspects(data: Record<string, any>) {
    const result = await this.perspectival.perceive(data);
    return result.aspects;
  }
}

/**
 * ReasoningAgent
 *
 * Specialized agent for logical reasoning and inference.
 */
export class ReasoningAgent extends CognitiveAgent {
  private gripService = getCognitiveGripService();

  constructor(config: Record<string, any> = {}) {
    super(
      "reasoning_agent",
      "reasoner",
      [
        {
          name: "realize_relevance",
          description: "Apply relevance realization to possibilities",
          inputTypes: ["possibility_set"],
          outputTypes: ["ranked_possibilities"],
        },
        {
          name: "infer",
          description: "Perform logical inference",
          inputTypes: ["premises"],
          outputTypes: ["conclusions"],
        },
        {
          name: "evaluate",
          description: "Evaluate options against criteria",
          inputTypes: ["options", "criteria"],
          outputTypes: ["evaluation"],
        },
      ],
      config
    );
  }

  protected async handleMessage(message: AgentMessage): Promise<any> {
    switch (message.type) {
      case "realize_relevance":
        return this.realizeRelevance(message.payload);
      case "infer":
        return this.infer(message.payload);
      case "evaluate":
        return this.evaluate(message.payload);
      default:
        return super.handleMessage(message);
    }
  }

  private async realizeRelevance(payload: {
    possibilities: any[];
    goals?: any[];
    constraints?: any[];
  }) {
    this.updateCognitiveLoad(0.7);

    // Set goals if provided
    if (payload.goals) {
      for (const goal of payload.goals) {
        this.gripService.addGoal(goal);
      }
    }

    // Set constraints if provided
    if (payload.constraints) {
      for (const constraint of payload.constraints) {
        this.gripService.addConstraint(constraint);
      }
    }

    const result = await this.gripService.realizeRelevance(
      payload.possibilities.map((p, i) => ({
        id: p.id || `possibility_${i}`,
        content: p,
      }))
    );

    this.updateCognitiveLoad(0.2);

    return result;
  }

  private async infer(premises: any[]) {
    this.updateCognitiveLoad(0.6);

    // Simple forward chaining inference
    const conclusions: any[] = [];

    // Pattern matching on premises
    for (let i = 0; i < premises.length; i++) {
      for (let j = i + 1; j < premises.length; j++) {
        const combined = this.tryCombine(premises[i], premises[j]);
        if (combined) {
          conclusions.push(combined);
        }
      }
    }

    this.updateCognitiveLoad(0.1);

    return { premises, conclusions };
  }

  private tryCombine(p1: any, p2: any): any | null {
    // Simple pattern matching - extend with PLN
    if (typeof p1 === "object" && typeof p2 === "object") {
      const commonKeys = Object.keys(p1).filter(k => k in p2);
      if (commonKeys.length > 0) {
        return {
          type: "inference",
          from: [p1, p2],
          connection: commonKeys,
          confidence: 0.7,
        };
      }
    }
    return null;
  }

  private async evaluate(payload: { options: any[]; criteria: any[] }) {
    this.updateCognitiveLoad(0.5);

    const evaluations = payload.options.map(option => {
      let score = 0;
      const breakdown: Record<string, number> = {};

      for (const criterion of payload.criteria) {
        const criterionScore = this.evaluateCriterion(option, criterion);
        const weight = criterion.weight || 1;
        breakdown[criterion.name || criterion.id] = criterionScore;
        score += criterionScore * weight;
      }

      return {
        option,
        score,
        breakdown,
        normalized: score / payload.criteria.length,
      };
    });

    this.updateCognitiveLoad(0.1);

    return {
      evaluations: evaluations.sort((a, b) => b.score - a.score),
      best: evaluations[0],
    };
  }

  private evaluateCriterion(option: any, criterion: any): number {
    // Simple evaluation - extend with sophisticated metrics
    if (criterion.predicate) {
      return criterion.predicate(option) ? 1 : 0;
    }

    const optionStr = JSON.stringify(option).toLowerCase();
    const criterionStr = (
      criterion.target ||
      criterion.name ||
      ""
    ).toLowerCase();

    return optionStr.includes(criterionStr) ? 0.8 : 0.2;
  }
}

/**
 * ActionAgent
 *
 * Specialized agent for action planning and execution.
 */
export class ActionAgent extends CognitiveAgent {
  private actionHistory: Array<{ action: any; result: any; timestamp: Date }> =
    [];

  constructor(config: Record<string, any> = {}) {
    super(
      "action_agent",
      "actor",
      [
        {
          name: "plan",
          description: "Plan action sequence",
          inputTypes: ["goal", "context"],
          outputTypes: ["action_plan"],
        },
        {
          name: "execute",
          description: "Execute action",
          inputTypes: ["action"],
          outputTypes: ["action_result"],
        },
        {
          name: "motor_command",
          description: "Generate motor command",
          inputTypes: ["intention"],
          outputTypes: ["motor_output"],
        },
      ],
      config
    );
  }

  protected async handleMessage(message: AgentMessage): Promise<any> {
    switch (message.type) {
      case "plan":
        return this.planAction(message.payload);
      case "execute":
        return this.executeAction(message.payload);
      case "motor":
        return this.generateMotorCommand(message.payload);
      default:
        return super.handleMessage(message);
    }
  }

  private async planAction(payload: { goal: any; context?: any }) {
    this.updateCognitiveLoad(0.5);

    const plan = {
      goal: payload.goal,
      context: payload.context,
      steps: this.decompose(payload.goal),
      contingencies: [],
      estimatedDuration: 0,
      confidence: 0.8,
    };

    // Estimate duration
    plan.estimatedDuration = plan.steps.length * 100;

    this.updateCognitiveLoad(0.2);

    return plan;
  }

  private decompose(goal: any): any[] {
    // Simple goal decomposition
    if (typeof goal === "string") {
      return [{ type: "atomic", action: goal }];
    }

    if (goal.steps) {
      return goal.steps;
    }

    // Default decomposition
    return [
      { type: "prepare", action: "initialize" },
      { type: "execute", action: goal },
      { type: "verify", action: "check_result" },
    ];
  }

  private async executeAction(action: any) {
    this.updateCognitiveLoad(0.7);

    const startTime = Date.now();

    // Simulate action execution
    const result = {
      action,
      status: "completed",
      output: { executed: true },
      duration: 0,
      timestamp: new Date(),
    };

    // Simulate delay
    await new Promise(resolve => setTimeout(resolve, 10));

    result.duration = Date.now() - startTime;

    this.actionHistory.push({
      action,
      result,
      timestamp: new Date(),
    });

    this.updateCognitiveLoad(0.2);

    return result;
  }

  private async generateMotorCommand(intention: any) {
    this.updateCognitiveLoad(0.4);

    // Map intention to motor output
    const motorCommand = {
      intention,
      command: {
        type: "motor",
        vector: Array(128)
          .fill(0)
          .map(() => Math.random() * 2 - 1),
        magnitude: 0.5,
        direction: intention.direction || "forward",
      },
      timestamp: new Date(),
    };

    this.updateCognitiveLoad(0.1);

    return motorCommand;
  }

  getActionHistory() {
    return [...this.actionHistory];
  }
}

/**
 * MemoryAgent
 *
 * Specialized agent for memory storage and retrieval.
 */
export class MemoryAgent extends CognitiveAgent {
  private shortTermMemory: Map<string, any> = new Map();
  private workingMemory: Map<string, any> = new Map();
  private episodicBuffer: any[] = [];

  // Dynamic episodic buffer limit from evolutionary config
  private get maxEpisodes(): number {
    return Math.round(getConfig("episodicBufferLimit"));
  }

  // Dynamic consolidation threshold from evolutionary config
  private get consolidationAccessThreshold(): number {
    return Math.round(getConfig("consolidationAccessThreshold"));
  }

  constructor(config: Record<string, any> = {}) {
    super(
      "memory_agent",
      "memory",
      [
        {
          name: "store",
          description: "Store information in memory",
          inputTypes: ["memory_item"],
          outputTypes: ["store_confirmation"],
        },
        {
          name: "retrieve",
          description: "Retrieve information from memory",
          inputTypes: ["query"],
          outputTypes: ["memory_results"],
        },
        {
          name: "consolidate",
          description: "Consolidate short-term to long-term",
          inputTypes: ["consolidation_request"],
          outputTypes: ["consolidation_result"],
        },
      ],
      config
    );
  }

  protected async handleMessage(message: AgentMessage): Promise<any> {
    switch (message.type) {
      case "store":
        return this.store(message.payload);
      case "retrieve":
        return this.retrieve(message.payload);
      case "consolidate":
        return this.consolidate();
      case "episode":
        return this.recordEpisode(message.payload);
      default:
        return super.handleMessage(message);
    }
  }

  private store(payload: { key: string; value: any; type?: string }) {
    const memoryType = payload.type || "short_term";

    switch (memoryType) {
      case "short_term":
        this.shortTermMemory.set(payload.key, {
          value: payload.value,
          timestamp: Date.now(),
          accessCount: 0,
        });
        break;
      case "working":
        this.workingMemory.set(payload.key, {
          value: payload.value,
          timestamp: Date.now(),
          active: true,
        });
        break;
    }

    return { stored: true, key: payload.key, type: memoryType };
  }

  private retrieve(query: { key?: string; pattern?: string; type?: string }) {
    const results: any[] = [];

    // Search by key
    if (query.key) {
      const stm = this.shortTermMemory.get(query.key);
      if (stm) {
        stm.accessCount++;
        results.push({ key: query.key, ...stm, source: "short_term" });
      }

      const wm = this.workingMemory.get(query.key);
      if (wm) {
        results.push({ key: query.key, ...wm, source: "working" });
      }
    }

    // Search by pattern
    if (query.pattern) {
      const regex = new RegExp(query.pattern, "i");

      for (const [key, value] of this.shortTermMemory) {
        if (regex.test(key) || regex.test(JSON.stringify(value))) {
          results.push({ key, ...value, source: "short_term" });
        }
      }
    }

    return { results, count: results.length };
  }

  private consolidate() {
    let consolidated = 0;
    const threshold = this.consolidationAccessThreshold;

    // Move frequently accessed short-term to working memory
    for (const [key, value] of this.shortTermMemory) {
      if (value.accessCount >= threshold) {
        this.workingMemory.set(key, {
          ...value,
          consolidatedAt: Date.now(),
        });
        this.shortTermMemory.delete(key);
        consolidated++;
      }
    }

    return { consolidated, shortTermSize: this.shortTermMemory.size };
  }

  private recordEpisode(episode: any) {
    this.episodicBuffer.push({
      ...episode,
      timestamp: Date.now(),
      id: this.episodicBuffer.length,
    });

    // Maintain buffer size
    if (this.episodicBuffer.length > this.maxEpisodes) {
      this.episodicBuffer.shift();
    }

    return { recorded: true, episodeCount: this.episodicBuffer.length };
  }

  getMemoryStats() {
    return {
      shortTermSize: this.shortTermMemory.size,
      workingSize: this.workingMemory.size,
      episodeCount: this.episodicBuffer.length,
    };
  }
}

/**
 * AttentionAgent
 *
 * Specialized agent for attention allocation and management.
 */
export class AttentionAgent extends CognitiveAgent {
  private attentionFocus: Set<string> = new Set();
  private stiValues: Map<string, number> = new Map();
  private ltiValues: Map<string, number> = new Map();

  // Dynamic attention budget from evolutionary config
  private get attentionBudget(): number {
    return Math.round(getConfig("agentAttentionBudget"));
  }
  private _currentBudget: number;

  // Dynamic decay rate from evolutionary config
  private get decayRate(): number {
    return getConfig("agentDecayRate");
  }

  constructor(config: Record<string, any> = {}) {
    super(
      "attention_agent",
      "attention",
      [
        {
          name: "allocate",
          description: "Allocate attention to item",
          inputTypes: ["attention_request"],
          outputTypes: ["attention_result"],
        },
        {
          name: "spread",
          description: "Spread attention through network",
          inputTypes: ["spread_request"],
          outputTypes: ["spread_result"],
        },
        {
          name: "focus",
          description: "Set attention focus",
          inputTypes: ["focus_request"],
          outputTypes: ["focus_state"],
        },
      ],
      config
    );
    this._currentBudget = this.attentionBudget;
  }

  protected async handleMessage(message: AgentMessage): Promise<any> {
    switch (message.type) {
      case "allocate":
        return this.allocateAttention(message.payload);
      case "spread":
        return this.spreadAttention(message.payload);
      case "focus":
        return this.setFocus(message.payload);
      case "decay":
        return this.applyDecay();
      default:
        return super.handleMessage(message);
    }
  }

  private allocateAttention(payload: { target: string; amount: number }) {
    const available = Math.min(payload.amount, this._currentBudget);
    const current = this.stiValues.get(payload.target) || 0;

    this.stiValues.set(payload.target, current + available);
    this._currentBudget -= available;

    // Update focus
    if (current + available >= 100) {
      this.attentionFocus.add(payload.target);
    }

    return {
      target: payload.target,
      allocated: available,
      newSti: current + available,
      inFocus: this.attentionFocus.has(payload.target),
    };
  }

  private spreadAttention(payload: {
    from: string;
    to: string[];
    rate: number;
  }) {
    const sourceSti = this.stiValues.get(payload.from) || 0;
    const spreadAmount = sourceSti * payload.rate;
    const perTarget = spreadAmount / payload.to.length;

    for (const target of payload.to) {
      const current = this.stiValues.get(target) || 0;
      this.stiValues.set(target, current + perTarget);
    }

    this.stiValues.set(payload.from, sourceSti - spreadAmount);

    return {
      from: payload.from,
      spreadAmount,
      targets: payload.to.length,
    };
  }

  private setFocus(payload: { targets: string[] }) {
    this.attentionFocus.clear();

    for (const target of payload.targets) {
      this.attentionFocus.add(target);
      // Boost STI for focused items
      const current = this.stiValues.get(target) || 0;
      this.stiValues.set(target, Math.max(100, current));
    }

    return {
      focus: Array.from(this.attentionFocus),
      size: this.attentionFocus.size,
    };
  }

  private applyDecay() {
    const currentDecayRate = this.decayRate;
    let decayed = 0;

    for (const [key, value] of this.stiValues) {
      const newValue = value * currentDecayRate;
      this.stiValues.set(key, newValue);

      // Remove from focus if below threshold
      if (newValue < 100) {
        this.attentionFocus.delete(key);
      }

      decayed++;
    }

    // Replenish budget as 1% of max budget per decay cycle
    // Note: This scales with agentAttentionBudget from evolutionary config
    // For default budget=1000, this equals 10 units (same as original fixed value)
    // As budget evolves, replenishment rate adapts proportionally
    const maxBudget = this.attentionBudget;
    this._currentBudget = Math.min(
      maxBudget,
      this._currentBudget + maxBudget * 0.01
    );

    return { decayed, budget: this._currentBudget };
  }

  getAttentionState() {
    return {
      focus: Array.from(this.attentionFocus),
      budget: this._currentBudget,
      stiCount: this.stiValues.size,
    };
  }
}

/**
 * MetaAgent
 *
 * Meta-cognitive agent for self-monitoring and reflection.
 */
export class MetaAgent extends CognitiveAgent {
  private echogenesis = getEchogenesisService();
  private reflectionHistory: any[] = [];

  constructor(config: Record<string, any> = {}) {
    super(
      "meta_agent",
      "meta",
      [
        {
          name: "reflect",
          description: "Perform meta-cognitive reflection",
          inputTypes: ["reflection_request"],
          outputTypes: ["insight"],
        },
        {
          name: "monitor",
          description: "Monitor system state",
          inputTypes: ["monitor_request"],
          outputTypes: ["system_state"],
        },
        {
          name: "wisdom",
          description: "Cultivate wisdom",
          inputTypes: ["wisdom_request"],
          outputTypes: ["wisdom_result"],
        },
      ],
      config
    );
  }

  protected async handleMessage(message: AgentMessage): Promise<any> {
    switch (message.type) {
      case "reflect":
        return this.reflect(message.payload);
      case "monitor":
        return this.monitorSystem();
      case "wisdom":
        return this.cultivateWisdom();
      case "examine":
        return this.examineSelf();
      default:
        return super.handleMessage(message);
    }
  }

  private async reflect(payload: any) {
    this.updateCognitiveLoad(0.5);

    const reflection: {
      input: any;
      timestamp: Date;
      insights: Array<{
        type: string;
        observation: string;
        recommendation: string;
      }>;
      questions: string[];
      adjustments: string[];
    } = {
      input: payload,
      timestamp: new Date(),
      insights: [],
      questions: [],
      adjustments: [],
    };

    // Generate insights about processing
    if (payload.processingResult) {
      reflection.insights.push({
        type: "processing",
        observation: "Processing completed",
        recommendation: "Continue current approach",
      });
    }

    // Generate Socratic questions
    reflection.questions = [
      "What assumptions are being made?",
      "What perspectives are being missed?",
      "How might this be wrong?",
    ];

    this.reflectionHistory.push(reflection);
    this.updateCognitiveLoad(0.2);

    return reflection;
  }

  private async monitorSystem() {
    const state = await this.echogenesis.getState();

    return {
      cognitive: state,
      timestamp: new Date(),
      health: "operational",
      alerts: [],
    };
  }

  private async cultivateWisdom() {
    return this.echogenesis.cultivateWisdom();
  }

  private async examineSelf() {
    return this.echogenesis.examineSelf();
  }

  getReflectionHistory() {
    return [...this.reflectionHistory];
  }
}

/**
 * Create all specialized agents
 */
export function createSpecializedAgents(): Map<AgentRole, CognitiveAgent> {
  const agents = new Map<AgentRole, CognitiveAgent>();

  agents.set("perceiver", new PerceptionAgent());
  agents.set("reasoner", new ReasoningAgent());
  agents.set("actor", new ActionAgent());
  agents.set("memory", new MemoryAgent());
  agents.set("attention", new AttentionAgent());
  agents.set("meta", new MetaAgent());

  return agents;
}

// Module exports
export default {
  PerceptionAgent,
  ReasoningAgent,
  ActionAgent,
  MemoryAgent,
  AttentionAgent,
  MetaAgent,
  createSpecializedAgents,
};
