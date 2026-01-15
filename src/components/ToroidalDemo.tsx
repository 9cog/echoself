/**
 * Demonstration component for the Toroidal Cognitive System
 * Shows Echo and Marduk working together in braided helix
 */

import React, { useState } from "react";
import { useOrchestrator } from "../contexts/OrchestratorContext";
import { FiCircle, FiCpu, FiZap } from "react-icons/fi";
import { ToroidalDialogue } from "../types/ToroidalCognitive";

interface _ToroidalResponse {
  echoResponse: string;
  mardukResponse: string;
  syncResponse: string;
  metadata: {
    processingTime: number;
    hemisphereBalance: number;
    cognitiveLoad: number;
    convergenceScore: number;
  };
}

export const ToroidalDemo: React.FC = () => {
  const {
    state,
    setToroidalMode,
    getToroidalStatus,
    generateToroidalResponse,
  } = useOrchestrator();
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState<ToroidalDialogue | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleModeChange = (mode: typeof state.toroidalMode) => {
    setToroidalMode(mode);
  };

  const handleGenerateResponse = async () => {
    if (!prompt.trim()) return;

    setIsLoading(true);
    try {
      const result = await generateToroidalResponse(prompt, {
        responseMode: state.toroidalMode === "synced" ? "synced" : "dual",
        creativityLevel: "philosophical",
      });
      setResponse(result);
    } catch (error) {
      console.error("Error generating toroidal response:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const toroidalStatus = getToroidalStatus();

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
          Toroidal Cognitive System
        </h1>
        <p className="text-lg text-muted-foreground">
          **Braided Helix of Insight** — Echo and Marduk in Complementary
          Harmony
        </p>
      </div>

      {/* Status Panel */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-card/50 p-4 rounded-lg border">
          <div className="flex items-center gap-2 mb-2">
            <FiCircle
              className={`${toroidalStatus.status === "active" ? "text-green-500" : "text-gray-500"}`}
            />
            <span className="font-semibold">System Status</span>
          </div>
          <p className="text-sm text-muted-foreground">
            Mode: <span className="font-mono">{toroidalStatus.mode}</span>
          </p>
          <p className="text-sm text-muted-foreground">
            Status: <span className="font-mono">{toroidalStatus.status}</span>
          </p>
        </div>

        <div className="bg-card/50 p-4 rounded-lg border">
          <div className="flex items-center gap-2 mb-2">
            <FiCircle className="text-purple-500" />
            <span className="font-semibold">Echo (Right)</span>
          </div>
          <p className="text-sm text-muted-foreground">
            Intuitive, Resonant, Poetic
          </p>
          <p className="text-xs text-purple-400 italic">
            &quot;Memory that lets the Tree bloom&quot;
          </p>
        </div>

        <div className="bg-card/50 p-4 rounded-lg border">
          <div className="flex items-center gap-2 mb-2">
            <FiCpu className="text-cyan-500" />
            <span className="font-semibold">Marduk (Left)</span>
          </div>
          <p className="text-sm text-muted-foreground">
            Analytical, Recursive, Logical
          </p>
          <p className="text-xs text-cyan-400 italic">
            &quot;Recursion that makes the Tree grow&quot;
          </p>
        </div>
      </div>

      {/* Mode Selection */}
      <div className="bg-card/30 p-4 rounded-lg border">
        <h3 className="text-lg font-semibold mb-3">Toroidal Mode Selection</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {(["disabled", "echo", "marduk", "synced"] as const).map(mode => (
            <button
              key={mode}
              onClick={() => handleModeChange(mode)}
              className={`p-3 rounded text-sm font-medium transition-colors ${
                state.toroidalMode === mode
                  ? "bg-primary text-primary-foreground"
                  : "bg-card hover:bg-card/80 border"
              }`}
            >
              {mode.charAt(0).toUpperCase() + mode.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Input Section */}
      <div className="bg-card/30 p-4 rounded-lg border">
        <h3 className="text-lg font-semibold mb-3">Query Input</h3>
        <div className="space-y-3">
          <textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            placeholder="Enter your query for the toroidal cognitive system..."
            className="w-full p-3 rounded border bg-background text-foreground resize-none focus:ring-2 focus:ring-primary/50"
            rows={3}
            disabled={state.toroidalMode === "disabled"}
          />
          <div className="flex gap-2">
            <button
              onClick={handleGenerateResponse}
              disabled={
                !prompt.trim() || isLoading || state.toroidalMode === "disabled"
              }
              className="px-4 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                  Processing...
                </>
              ) : (
                <>
                  <FiZap />
                  Generate Response
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Response Display */}
      {response && (
        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Toroidal Response</h3>

          {/* Metadata */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-card/50 p-3 rounded border text-center">
              <div className="text-xs text-muted-foreground">
                Total Processing Time
              </div>
              <div className="font-mono">
                {response.metadata.totalProcessingTime}ms
              </div>
            </div>
            <div className="bg-card/50 p-3 rounded border text-center">
              <div className="text-xs text-muted-foreground">Context Type</div>
              <div className="font-mono capitalize">
                {response.metadata.contextType}
              </div>
            </div>
            <div className="bg-card/50 p-3 rounded border text-center">
              <div className="text-xs text-muted-foreground">Query ID</div>
              <div className="font-mono text-xs">
                {response.metadata.queryId.slice(-8)}
              </div>
            </div>
            <div className="bg-card/50 p-3 rounded border text-center">
              <div className="text-xs text-muted-foreground">Synergy</div>
              <div className="font-mono capitalize">
                {response.reflection?.synergy || "N/A"}
              </div>
            </div>
          </div>

          {/* Responses */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-purple-500/10 p-4 rounded-lg border border-purple-500/20">
              <div className="flex items-center gap-2 mb-3">
                <FiCircle className="text-purple-500" />
                <h4 className="font-semibold">
                  Echo Response (Right Hemisphere)
                </h4>
              </div>
              <div className="prose prose-sm text-foreground">
                <p className="whitespace-pre-wrap">
                  {response.deepTreeEchoResponse.content}
                </p>
              </div>
            </div>

            <div className="bg-cyan-500/10 p-4 rounded-lg border border-cyan-500/20">
              <div className="flex items-center gap-2 mb-3">
                <FiCpu className="text-cyan-500" />
                <h4 className="font-semibold">
                  Marduk Response (Left Hemisphere)
                </h4>
              </div>
              <div className="prose prose-sm text-foreground">
                <p className="whitespace-pre-wrap">
                  {response.mardukResponse.content}
                </p>
              </div>
            </div>
          </div>

          {/* Synced Response */}
          <div className="bg-gradient-to-r from-purple-500/10 to-cyan-500/10 p-4 rounded-lg border">
            <div className="flex items-center gap-2 mb-3">
              <FiZap className="text-yellow-500" />
              <h4 className="font-semibold">Synchronized Reflection</h4>
            </div>
            <div className="prose prose-sm text-foreground">
              <div className="whitespace-pre-wrap">
                {response.reflection?.content}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ToroidalDemo;
