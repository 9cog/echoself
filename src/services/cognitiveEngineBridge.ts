/**
 * Cognitive Engine Bridge - Python-TypeScript interop for cognitive engines
 *
 * Bridges the TypeScript frontend with Python cognitive engines
 * (RelevanceRealizationEngine and VirtualEmbodiment) via child_process.
 * Provides graceful fallback when Python is unavailable.
 */

import { execFile } from "child_process";
import { promisify } from "util";
import * as path from "path";

const execFileAsync = promisify(execFile);

// ---------- Types ----------

export interface RelevanceEvaluation {
  conceptId: string;
  relevanceScore: number;
  criteria: {
    goalAlignment: number;
    predictivePower: number;
    cognitiveEconomy: number;
    noveltyValue: number;
    contextualFit: number;
  };
  opponentProcessStates: {
    explorationExploitation: number;
    breadthDepth: number;
    speedAccuracy: number;
    certaintyOpenness: number;
  };
}

export interface EmbodiedGrounding {
  originalLanguage: string;
  groundedMetaphors: string[];
  sensoriMotorMappings: Array<{
    concept: string;
    modality: string;
    affordance: string;
  }>;
  bodySchemaRelevance: number;
  predictionError: number;
}

export interface BridgeStatus {
  pythonAvailable: boolean;
  relevanceEngineReady: boolean;
  embodimentEngineReady: boolean;
  pythonVersion: string | null;
  lastCheck: string;
}

// ---------- Service ----------

export class CognitiveEngineBridge {
  private static instance: CognitiveEngineBridge;
  private pythonPath: string;
  private projectRoot: string;
  private pythonAvailable: boolean | null = null;
  private pythonVersion: string | null = null;

  private constructor() {
    this.pythonPath = process.env.PYTHON_PATH || "python3";
    // Resolve project root relative to this file's location (src/services/)
    this.projectRoot =
      process.env.ECHOSELF_ROOT ||
      path.resolve(__dirname, "..", "..");
  }

  public static getInstance(): CognitiveEngineBridge {
    if (!CognitiveEngineBridge.instance) {
      CognitiveEngineBridge.instance = new CognitiveEngineBridge();
    }
    return CognitiveEngineBridge.instance;
  }

  // ---------- Python availability ----------

  /**
   * Check whether Python 3 and required dependencies are available.
   */
  public async checkPythonAvailability(): Promise<boolean> {
    try {
      const { stdout } = await execFileAsync(this.pythonPath, [
        "-c",
        "import sys, numpy; print(sys.version.split()[0])",
      ]);
      this.pythonVersion = stdout.trim();
      this.pythonAvailable = true;
      return true;
    } catch {
      this.pythonAvailable = false;
      this.pythonVersion = null;
      console.warn(
        "CognitiveEngineBridge: Python 3 or numpy not available. Using TypeScript fallbacks."
      );
      return false;
    }
  }

  /**
   * Return current bridge health status.
   */
  public async getStatus(): Promise<BridgeStatus> {
    if (this.pythonAvailable === null) {
      await this.checkPythonAvailability();
    }
    return {
      pythonAvailable: this.pythonAvailable ?? false,
      relevanceEngineReady: this.pythonAvailable ?? false,
      embodimentEngineReady: this.pythonAvailable ?? false,
      pythonVersion: this.pythonVersion,
      lastCheck: new Date().toISOString(),
    };
  }

  // ---------- Relevance Realization ----------

  /**
   * Evaluate the relevance of a concept within a given context by calling
   * the Python RelevanceRealizationEngine.
   *
   * Falls back to a heuristic TypeScript implementation when Python is
   * unavailable.
   */
  public async evaluateRelevance(
    conceptId: string,
    conceptData: Record<string, unknown>,
    context: {
      goals?: string[];
      noveltyNeeded?: boolean;
      precisionNeeded?: boolean;
    } = {}
  ): Promise<RelevanceEvaluation> {
    if (this.pythonAvailable === null) {
      await this.checkPythonAvailability();
    }

    if (this.pythonAvailable) {
      return this.evaluateRelevancePython(conceptId, conceptData, context);
    }
    return this.evaluateRelevanceFallback(conceptId, conceptData, context);
  }

  private async evaluateRelevancePython(
    conceptId: string,
    conceptData: Record<string, unknown>,
    context: Record<string, unknown>
  ): Promise<RelevanceEvaluation> {
    const script = `
import sys, json
sys.path.insert(0, '${this.projectRoot}')
from relevance_realization_engine import RelevanceRealizationEngine, Possibility

engine = RelevanceRealizationEngine()
ctx = json.loads('''${JSON.stringify(context)}''')
concept = json.loads('''${JSON.stringify(conceptData)}''')

p = Possibility(id='${conceptId}', data=concept)
results = engine.realize_relevance([p], ctx)

if results:
    r = results[0]
    out = {
        "conceptId": r.id,
        "relevanceScore": round(r.criteria.score(), 4),
        "criteria": {
            "goalAlignment": round(r.criteria.goal_alignment, 4),
            "predictivePower": round(r.criteria.predictive_power, 4),
            "cognitiveEconomy": round(r.criteria.cognitive_economy, 4),
            "noveltyValue": round(r.criteria.novelty_value, 4),
            "contextualFit": round(r.criteria.contextual_fit, 4),
        },
        "opponentProcessStates": {
            "explorationExploitation": round(engine.exploration_exploitation.balance, 4),
            "breadthDepth": round(engine.breadth_depth.balance, 4),
            "speedAccuracy": round(engine.speed_accuracy.balance, 4),
            "certaintyOpenness": round(engine.certainty_openness.balance, 4),
        }
    }
else:
    out = {
        "conceptId": '${conceptId}',
        "relevanceScore": 0,
        "criteria": {"goalAlignment":0,"predictivePower":0,"cognitiveEconomy":0,"noveltyValue":0,"contextualFit":0},
        "opponentProcessStates": {"explorationExploitation":0.5,"breadthDepth":0.5,"speedAccuracy":0.5,"certaintyOpenness":0.6}
    }

print(json.dumps(out))
`;

    try {
      const { stdout } = await execFileAsync(this.pythonPath, ["-c", script], {
        timeout: 15000,
      });
      return JSON.parse(stdout.trim()) as RelevanceEvaluation;
    } catch (error) {
      console.error("Python relevance evaluation failed, using fallback:", error);
      return this.evaluateRelevanceFallback(
        conceptId,
        conceptData,
        context
      );
    }
  }

  private evaluateRelevanceFallback(
    conceptId: string,
    conceptData: Record<string, unknown>,
    context: {
      goals?: string[];
      noveltyNeeded?: boolean;
      precisionNeeded?: boolean;
    }
  ): RelevanceEvaluation {
    // Heuristic fallback when Python is not available
    const complexity = (conceptData.complexity as number) ?? 0.5;
    const cognitiveEconomy = 1.0 / (1.0 + complexity);

    const goalAlignment = context.goals && context.goals.length > 0 ? 0.5 : 0.3;
    const noveltyValue = context.noveltyNeeded ? 0.7 : 0.4;
    const predictivePower = 0.5;
    const contextualFit = 0.7;

    const relevanceScore =
      goalAlignment * 0.3 +
      predictivePower * 0.25 +
      cognitiveEconomy * 0.2 +
      noveltyValue * 0.15 +
      contextualFit * 0.1;

    return {
      conceptId,
      relevanceScore: Math.round(relevanceScore * 10000) / 10000,
      criteria: {
        goalAlignment,
        predictivePower,
        cognitiveEconomy: Math.round(cognitiveEconomy * 10000) / 10000,
        noveltyValue,
        contextualFit,
      },
      opponentProcessStates: {
        explorationExploitation: 0.5,
        breadthDepth: 0.5,
        speedAccuracy: 0.5,
        certaintyOpenness: 0.6,
      },
    };
  }

  // ---------- Virtual Embodiment ----------

  /**
   * Ground language in embodied metaphors by calling the Python
   * VirtualEmbodiment engine.
   *
   * Falls back to a template-based TypeScript implementation when Python
   * is unavailable.
   */
  public async groundLanguageInEmbodiment(
    language: string,
    environmentContext: {
      objects?: Array<{ id: string; position?: number[]; value?: number }>;
      goal?: { type: string; valueThreshold?: number };
    } = {}
  ): Promise<EmbodiedGrounding> {
    if (this.pythonAvailable === null) {
      await this.checkPythonAvailability();
    }

    if (this.pythonAvailable) {
      return this.groundLanguagePython(language, environmentContext);
    }
    return this.groundLanguageFallback(language, environmentContext);
  }

  private async groundLanguagePython(
    language: string,
    environmentContext: Record<string, unknown>
  ): Promise<EmbodiedGrounding> {
    // Build a safe environment state for Python (numpy arrays constructed in-script)
    const objectsJson = JSON.stringify(
      ((environmentContext.objects as any[]) || []).map((o: any) => ({
        id: o.id,
        position: o.position || [0, 0, 0],
        value: o.value || 0.5,
      }))
    );

    const script = `
import sys, json, numpy as np
sys.path.insert(0, '${this.projectRoot}')
from virtual_embodiment import VirtualEmbodiment

embodiment = VirtualEmbodiment()

objects_raw = json.loads('''${objectsJson}''')
environment = {
    'objects': [
        {'id': o['id'], 'position': np.array(o['position']), 'value': o['value']}
        for o in objects_raw
    ],
    'terrain': {'walkable': [{'center': np.array([1.0, 0.0, 0.0])}]}
}

goal = ${environmentContext.goal ? `json.loads('''${JSON.stringify(environmentContext.goal)}''')` : "None"}

result = embodiment.perceive_act_cycle(environment, goal)
proprio = embodiment.get_proprioception()

# Map concepts in the language to sensorimotor modalities
language = '''${language.replace(/'/g, "\\'")}'''
words = language.lower().split()

modality_map = {
    'see': 'vision', 'look': 'vision', 'observe': 'vision', 'watch': 'vision',
    'hear': 'audio', 'listen': 'audio', 'sound': 'audio',
    'touch': 'touch', 'feel': 'touch', 'grasp': 'touch', 'hold': 'touch',
    'move': 'proprioception', 'walk': 'proprioception', 'run': 'proprioception',
    'sense': 'interoception', 'aware': 'interoception',
}

mappings = []
for word in words:
    if word in modality_map:
        mappings.append({
            'concept': word,
            'modality': modality_map[word],
            'affordance': f'{word}_action'
        })

metaphors = [
    f"The concept grounds through {m['modality']} as {m['concept']}"
    for m in mappings
] or [f"Language '{language[:50]}' maps to embodied spatial reasoning"]

prediction_error = float(result.get('prediction_error', 0.0))

out = {
    'originalLanguage': language,
    'groundedMetaphors': metaphors,
    'sensoriMotorMappings': mappings or [{'concept': 'abstract', 'modality': 'proprioception', 'affordance': 'spatial_reasoning'}],
    'bodySchemaRelevance': float(proprio['energy']),
    'predictionError': round(prediction_error, 4)
}

print(json.dumps(out))
`;

    try {
      const { stdout } = await execFileAsync(this.pythonPath, ["-c", script], {
        timeout: 15000,
      });
      return JSON.parse(stdout.trim()) as EmbodiedGrounding;
    } catch (error) {
      console.error(
        "Python embodiment grounding failed, using fallback:",
        error
      );
      return this.groundLanguageFallback(language, environmentContext);
    }
  }

  private groundLanguageFallback(
    language: string,
    _environmentContext: Record<string, unknown>
  ): EmbodiedGrounding {
    // Template-based fallback when Python is unavailable
    const words = language.toLowerCase().split(/\s+/);

    const modalityKeywords: Record<string, string> = {
      see: "vision",
      look: "vision",
      observe: "vision",
      hear: "audio",
      listen: "audio",
      touch: "touch",
      feel: "touch",
      grasp: "touch",
      move: "proprioception",
      walk: "proprioception",
      sense: "interoception",
    };

    const mappings: Array<{
      concept: string;
      modality: string;
      affordance: string;
    }> = [];

    for (const word of words) {
      if (modalityKeywords[word]) {
        mappings.push({
          concept: word,
          modality: modalityKeywords[word],
          affordance: `${word}_action`,
        });
      }
    }

    const metaphors =
      mappings.length > 0
        ? mappings.map(
            m =>
              `The concept "${m.concept}" grounds through ${m.modality} modality`
          )
        : [
            `Language "${language.substring(0, 50)}" maps to abstract spatial reasoning`,
            "Embodied grounding via proprioceptive self-model",
          ];

    return {
      originalLanguage: language,
      groundedMetaphors: metaphors,
      sensoriMotorMappings:
        mappings.length > 0
          ? mappings
          : [
              {
                concept: "abstract",
                modality: "proprioception",
                affordance: "spatial_reasoning",
              },
            ],
      bodySchemaRelevance: 1.0,
      predictionError: 0.0,
    };
  }
}

export default CognitiveEngineBridge;
