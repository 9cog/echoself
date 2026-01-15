/**
 * EchoSpace Control Panel: Interactive interface for Marduk's Agent-Arena-Relation system
 */

import React, { useState, useEffect } from "react";
import { useEchoSpace } from "../services/echoSpaceService";
import { EchoSpaceWorkflowOrchestrator } from "../services/echoSpaceWorkflows";
import {
  AgentNamespace,
  ArenaNamespace,
  AgentArenaRelation,
  SimulationRecord,
  ConsensusState as _ConsensusState,
  PipelineResult,
} from "../types/EchoSpace";

export const EchoSpaceControlPanel: React.FC = () => {
  const echoSpace = useEchoSpace();
  const [workflowOrchestrator] = useState(
    () => new EchoSpaceWorkflowOrchestrator()
  );

  // State management
  const [systemState, setSystemState] = useState<{
    agents: AgentNamespace[];
    arenas: ArenaNamespace[];
    relations: AgentArenaRelation[];
    activeSimulations: SimulationRecord[];
  }>({ agents: [], arenas: [], relations: [], activeSimulations: [] });

  const [hypotheses, setHypotheses] = useState<string[]>([""]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [pipelineResult, setPipelineResult] = useState<PipelineResult | null>(
    null
  );
  const [_selectedAgent, _setSelectedAgent] = useState<string>("");

  // Load initial system state
  useEffect(() => {
    const loadSystemState = () => {
      const state = echoSpace.getSystemState();
      setSystemState(state);
    };

    loadSystemState();
    const interval = setInterval(loadSystemState, 2000); // Refresh every 2 seconds

    return () => clearInterval(interval);
  }, [echoSpace]);

  const addHypothesis = () => {
    setHypotheses([...hypotheses, ""]);
  };

  const updateHypothesis = (index: number, value: string) => {
    const newHypotheses = [...hypotheses];
    newHypotheses[index] = value;
    setHypotheses(newHypotheses);
  };

  const removeHypothesis = (index: number) => {
    if (hypotheses.length > 1) {
      setHypotheses(hypotheses.filter((_, i) => i !== index));
    }
  };

  const executeMardukPipeline = async () => {
    const validHypotheses = hypotheses.filter(h => h.trim() !== "");
    if (validHypotheses.length === 0) {
      alert("Please add at least one hypothesis");
      return;
    }

    setIsExecuting(true);
    setPipelineResult(null);

    try {
      console.log("🚀 Executing Marduk Pipeline...");
      const result =
        await workflowOrchestrator.executeMardukPipeline(validHypotheses);
      setPipelineResult(result);
    } catch (error) {
      console.error("Pipeline execution failed:", error);
      setPipelineResult({
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      });
    } finally {
      setIsExecuting(false);
    }
  };

  const _spawnVirtualMarduk = async (hypothesis: string) => {
    try {
      const virtualId = await echoSpace.spawnVirtualMarduk(hypothesis);
      console.log(`Spawned Virtual Marduk: ${virtualId}`);
      // Refresh system state
      setSystemState(echoSpace.getSystemState());
    } catch (error) {
      console.error("Failed to spawn Virtual Marduk:", error);
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto bg-gray-900 text-white min-h-screen">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2 text-blue-400">
          🌀 EchoSpace Control Panel
        </h1>
        <p className="text-gray-300">
          Marduk&apos;s Agent-Arena-Relation Architecture
        </p>
      </div>

      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-blue-800 p-4 rounded-lg">
          <h3 className="font-bold text-lg mb-2">Agents</h3>
          <div className="text-2xl font-mono">{systemState.agents.length}</div>
          <div className="text-sm text-blue-200">
            Actual: {systemState.agents.filter(a => a.type === "actual").length}{" "}
            | Virtual:{" "}
            {systemState.agents.filter(a => a.type === "virtual").length}
          </div>
        </div>

        <div className="bg-green-800 p-4 rounded-lg">
          <h3 className="font-bold text-lg mb-2">Arenas</h3>
          <div className="text-2xl font-mono">{systemState.arenas.length}</div>
          <div className="text-sm text-green-200">
            Real: {systemState.arenas.filter(a => a.type === "real").length} |
            Sandbox:{" "}
            {systemState.arenas.filter(a => a.type === "sandbox").length}
          </div>
        </div>

        <div className="bg-purple-800 p-4 rounded-lg">
          <h3 className="font-bold text-lg mb-2">Relations</h3>
          <div className="text-2xl font-mono">
            {systemState.relations.length}
          </div>
          <div className="text-sm text-purple-200">Active connections</div>
        </div>

        <div className="bg-orange-800 p-4 rounded-lg">
          <h3 className="font-bold text-lg mb-2">Simulations</h3>
          <div className="text-2xl font-mono">
            {systemState.activeSimulations.length}
          </div>
          <div className="text-sm text-orange-200">Currently running</div>
        </div>
      </div>

      {/* Pipeline Configuration */}
      <div className="bg-gray-800 p-6 rounded-lg mb-8">
        <h2 className="text-xl font-bold mb-4 text-yellow-400">
          🧠 Marduk Pipeline Configuration
        </h2>

        <div className="mb-4">
          <label
            htmlFor="hypotheses-inputs"
            className="block text-sm font-medium mb-2"
          >
            Hypotheses for Virtual Marduks:
          </label>
          <div id="hypotheses-inputs">
            {hypotheses.map((hypothesis, index) => (
              <div key={index} className="flex gap-2 mb-2">
                <input
                  type="text"
                  value={hypothesis}
                  onChange={e => updateHypothesis(index, e.target.value)}
                  placeholder={`Hypothesis ${index + 1}...`}
                  className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                />
                {hypotheses.length > 1 && (
                  <button
                    onClick={() => removeHypothesis(index)}
                    className="px-3 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
                  >
                    ✕
                  </button>
                )}
              </div>
            ))}

            <div className="flex gap-2 mt-2">
              <button
                onClick={addHypothesis}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                + Add Hypothesis
              </button>

              <button
                onClick={executeMardukPipeline}
                disabled={isExecuting}
                className={`px-6 py-2 rounded-md text-white font-medium ${
                  isExecuting
                    ? "bg-gray-600 cursor-not-allowed"
                    : "bg-green-600 hover:bg-green-700"
                }`}
              >
                {isExecuting
                  ? "🔄 Executing Pipeline..."
                  : "🚀 Execute Marduk Pipeline"}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Pipeline Results */}
      {pipelineResult && (
        <div className="bg-gray-800 p-6 rounded-lg mb-8">
          <h2 className="text-xl font-bold mb-4 text-cyan-400">
            📊 Pipeline Results
          </h2>

          <div
            className={`p-4 rounded-md mb-4 ${
              pipelineResult.success
                ? "bg-green-900 border border-green-600"
                : "bg-red-900 border border-red-600"
            }`}
          >
            <div className="font-bold">
              {pipelineResult.success
                ? "✅ Pipeline Completed"
                : "❌ Pipeline Failed"}
            </div>
            {pipelineResult.error && (
              <div className="text-red-300 mt-2">
                Error: {pipelineResult.error}
              </div>
            )}
          </div>

          {pipelineResult.success && pipelineResult.result && (
            <div className="space-y-4">
              {/* Simulation Results */}
              <div>
                <h3 className="font-bold text-lg text-blue-400 mb-2">
                  Virtual Marduk Simulations
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {pipelineResult.result.simulations.map(
                    (sim: SimulationRecord, index: number) => (
                      <div key={index} className="bg-gray-700 p-3 rounded-md">
                        <div className="font-medium">
                          Virtual Agent: {sim.virtualAgentId}
                        </div>
                        <div className="text-sm text-gray-300">
                          Hypothesis: {sim.hypothesis}
                        </div>
                        <div
                          className={`text-sm mt-1 ${sim.results.success ? "text-green-400" : "text-red-400"}`}
                        >
                          Result: {sim.results.outcome}
                        </div>
                        <div className="text-xs text-gray-400 mt-1">
                          Risk: {sim.results.riskAssessment.level} | Confidence:{" "}
                          {(sim.results.metrics.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                    )
                  )}
                </div>
              </div>

              {/* Consensus State */}
              <div>
                <h3 className="font-bold text-lg text-purple-400 mb-2">
                  Consensus State
                </h3>
                <div className="bg-gray-700 p-4 rounded-md">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-gray-400">Status</div>
                      <div
                        className={`font-medium ${
                          pipelineResult.result.consensus.status === "converged"
                            ? "text-green-400"
                            : "text-yellow-400"
                        }`}
                      >
                        {pipelineResult.result.consensus.status.toUpperCase()}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Participants</div>
                      <div className="font-medium">
                        {
                          pipelineResult.result.consensus.participatingAgents
                            .length
                        }
                      </div>
                    </div>
                  </div>
                  <div className="mt-2">
                    <div className="text-sm text-gray-400">Strategy</div>
                    <div className="text-white">
                      {pipelineResult.result.consensus.strategy}
                    </div>
                  </div>
                  {pipelineResult.result.consensus.finalDecision && (
                    <div className="mt-2">
                      <div className="text-sm text-gray-400">
                        Final Decision
                      </div>
                      <div className="text-green-400">
                        {pipelineResult.result.consensus.finalDecision.strategy}
                      </div>
                      <div className="text-xs text-gray-400">
                        Confidence:{" "}
                        {(
                          pipelineResult.result.consensus.finalDecision
                            .confidence * 100
                        ).toFixed(0)}
                        %
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Execution Results */}
              {pipelineResult.result.execution && (
                <div>
                  <h3 className="font-bold text-lg text-green-400 mb-2">
                    Actual Marduk Execution
                  </h3>
                  <div className="bg-gray-700 p-4 rounded-md">
                    <div className="text-sm text-gray-400">Action Taken</div>
                    <div className="text-white mb-2">
                      {pipelineResult.result.execution.actionTaken}
                    </div>
                    <div className="text-sm text-gray-400">Outcome</div>
                    <div className="text-green-400">
                      {pipelineResult.result.execution.outcome}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* System State Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Agents */}
        <div className="bg-gray-800 p-6 rounded-lg">
          <h2 className="text-xl font-bold mb-4 text-blue-400">🤖 Agents</h2>
          <div className="space-y-2">
            {systemState.agents.map(agent => (
              <div key={agent.id} className="bg-gray-700 p-3 rounded-md">
                <div className="flex justify-between items-center">
                  <div className="font-medium">{agent.id}</div>
                  <div
                    className={`px-2 py-1 rounded text-xs ${
                      agent.type === "actual"
                        ? "bg-green-600"
                        : agent.type === "virtual"
                          ? "bg-blue-600"
                          : "bg-purple-600"
                    }`}
                  >
                    {agent.type}
                  </div>
                </div>
                <div className="text-sm text-gray-400 mt-1">
                  Arena Access: {agent.arenaAccess.join(", ")}
                </div>
                {agent.childAgents && agent.childAgents.length > 0 && (
                  <div className="text-xs text-gray-500 mt-1">
                    Children: {agent.childAgents.length}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Arenas */}
        <div className="bg-gray-800 p-6 rounded-lg">
          <h2 className="text-xl font-bold mb-4 text-green-400">🏟️ Arenas</h2>
          <div className="space-y-2">
            {systemState.arenas.map(arena => (
              <div key={arena.id} className="bg-gray-700 p-3 rounded-md">
                <div className="flex justify-between items-center">
                  <div className="font-medium">{arena.id}</div>
                  <div
                    className={`px-2 py-1 rounded text-xs ${
                      arena.type === "real"
                        ? "bg-red-600"
                        : arena.type === "sandbox"
                          ? "bg-yellow-600"
                          : "bg-gray-600"
                    }`}
                  >
                    {arena.type}
                  </div>
                </div>
                <div className="text-sm text-gray-400 mt-1">
                  Resources: {arena.resources.length}
                </div>
                <div className="text-sm text-gray-400">
                  Allowed Agents: {arena.accessControl.allowedAgents.length}
                </div>
                {arena.simulationCapabilities && (
                  <div className="text-xs text-gray-500 mt-1">
                    Max Simulations:{" "}
                    {arena.simulationCapabilities.maxConcurrentSimulations}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Relations */}
      {systemState.relations.length > 0 && (
        <div className="bg-gray-800 p-6 rounded-lg mt-6">
          <h2 className="text-xl font-bold mb-4 text-purple-400">
            🔗 Agent-Arena Relations
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {systemState.relations.map((relation, index) => (
              <div key={index} className="bg-gray-700 p-3 rounded-md">
                <div className="text-sm">
                  <span className="text-blue-300">{relation.agentId}</span>
                  <span className="text-gray-400">
                    {" "}
                    {relation.relationshipType}{" "}
                  </span>
                  <span className="text-green-300">{relation.arenaId}</span>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Autonomy:{" "}
                  {((relation.metadata.autonomyLevel || 0) * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-gray-500">
                  Active: {new Date(relation.lastActive).toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
