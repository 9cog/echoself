/**
 * EchoZeroDomain — training, persona, memory, and NanEcho surface unions.
 * Invalid combinations are unrepresentable. Do not add parallel booleans.
 */

export type NonEmpty<T> = readonly [T, ...T[]];

export type ExplicitConfirmation = string & {
  readonly ExplicitConfirmation: unique symbol;
};

export type PreparedCorpus = {
  readonly kind: "prepared";
  readonly source: string;
};

export type PrepFailure = {
  readonly kind: "failed";
  readonly reason: string;
  readonly fallbackCorpus: never;
};

export type TrainingMode =
  | {
      readonly kind: "ci";
      readonly documented: { readonly layers: 4; readonly iterations: 200 };
    }
  | { readonly kind: "full"; readonly layers: 12; readonly iterations: 50000 }
  | { readonly kind: "relentless"; readonly schedule: "every 4 hours" };

export type CheckpointSource =
  | {
      readonly priority: 1;
      readonly kind: "latest_checkpoint";
      readonly path: ".training-progress/checkpoints/latest_checkpoint.pt";
    }
  | { readonly priority: 2; readonly kind: "downloaded_artifact" }
  | { readonly priority: 3; readonly kind: "github_actions_cache" }
  | { readonly priority: 4; readonly kind: "backup" };

export const RESTORE_ORDER = [
  "latest_checkpoint",
  "downloaded_artifact",
  "github_actions_cache",
  "backup",
] as const satisfies readonly CheckpointSource["kind"][];

export type Fingerprint = { readonly config: string; readonly data: string };

export type CheckpointRef = {
  readonly id: string;
  readonly fingerprint: Fingerprint;
  readonly iteration: number;
  readonly valLoss: number;
  readonly source: CheckpointSource;
  readonly evidenceFile: string;
};

export type Discovery =
  | {
      readonly kind: "checkpoints_found";
      readonly checkpoints: NonEmpty<CheckpointRef>;
    }
  | { readonly kind: "no_checkpoint"; readonly guardianVerified: true };

export type LineageVerdict =
  | { readonly kind: "coherent"; readonly head: CheckpointRef }
  | {
      readonly kind: "divergent";
      readonly fingerprint: Fingerprint;
      readonly heads: NonEmpty<CheckpointRef>;
    }
  | { readonly kind: "forked"; readonly fingerprints: NonEmpty<Fingerprint> }
  | { readonly kind: "uninitialized" }
  | { readonly kind: "metadata_only"; readonly head: CheckpointRef };

export const LINEAGE_KINDS = [
  "coherent",
  "divergent",
  "forked",
  "uninitialized",
  "metadata_only",
] as const satisfies readonly LineageVerdict["kind"][];

export const RESTORE_LINEAGE = [
  "divergent",
  "forked",
  "uninitialized",
  "metadata_only",
] as const satisfies readonly Exclude<LineageVerdict["kind"], "coherent">[];

export type TrainingOrigin =
  | { readonly kind: "restored"; readonly checkpoint: CheckpointRef }
  | {
      readonly kind: "confirmed_fresh_start";
      readonly discovery: Extract<Discovery, { kind: "no_checkpoint" }>;
      readonly confirmation: ExplicitConfirmation;
    };

export type TrainingReadiness =
  | {
      readonly kind: "runnable";
      readonly origin: TrainingOrigin;
      readonly corpus: PreparedCorpus;
    }
  | {
      readonly kind: "prep_blocked";
      readonly origin: TrainingOrigin;
      readonly prep: PrepFailure;
    }
  | {
      readonly kind: "restore_required";
      readonly discovery: Extract<Discovery, { kind: "checkpoints_found" }>;
    };

export type PersonaDimension =
  | "cognitive"
  | "introspective"
  | "adaptive"
  | "recursive"
  | "synergistic"
  | "holographic"
  | "neural-symbolic"
  | "dynamic";

export const PERSONA: Readonly<
  Record<PersonaDimension, { readonly weight: 0.15 | 0.1 }>
> = {
  cognitive: { weight: 0.15 },
  introspective: { weight: 0.15 },
  adaptive: { weight: 0.15 },
  recursive: { weight: 0.15 },
  synergistic: { weight: 0.1 },
  holographic: { weight: 0.1 },
  "neural-symbolic": { weight: 0.1 },
  dynamic: { weight: 0.1 },
};

export const attentionThreshold = (
  cognitiveLoad: number,
  recentActivity: number
): number => 0.5 + cognitiveLoad * 0.3 - recentActivity * 0.2;

export type MemoryOperation =
  | {
      readonly kind: "continual_learn";
      readonly transcriptDelta: string;
      readonly target: "AGENTS.md";
    }
  | {
      readonly kind: "dream";
      readonly scope: "mem0";
      readonly action: "decide_only";
      readonly evidence: string;
    }
  | {
      readonly kind: "remember";
      readonly scope: "mem0";
      readonly infer: false;
      readonly facts: readonly string[];
    };

export type AgentZeroSurface =
  | {
      readonly kind: "tool";
      readonly local:
        | "nanecho"
        | "checkpoint_guardian"
        | "training_progress_inspector";
    }
  | {
      readonly kind: "extension";
      readonly local: "continual_learning_hook" | "mem0_memory_point";
    }
  | {
      readonly kind: "profile";
      readonly local: "deep_tree_echo";
      readonly mode: TrainingMode;
    }
  | {
      readonly kind: "instrument";
      readonly local:
        | "checkpoint_guardian"
        | "prepare_nanecho"
        | "evaluation"
        | "autognosis";
    }
  | {
      readonly kind: "subordinate";
      readonly local: "nanecho" | "checkpoint_guardian" | "mem0_dream";
    }
  | {
      readonly kind: "nanecho_surface";
      readonly local: "echoself/data/nanecho";
    }
  | {
      readonly kind: "autognosis";
      readonly local: "echoself.autognosis";
      readonly layer: AutognosisLayer;
    }
  | { readonly kind: "harmonic_esn"; readonly local: "harmonic_resonance" }
  | {
      readonly kind: "tiny_inference";
      readonly local: "nanecho_runtime" | "llama_cpp_spec_client";
    }
  | { readonly kind: "vorticog_map"; readonly local: "vorticog_adapter" };

export type SituationTag =
  | "resume_train"
  | "fresh_train"
  | "restore"
  | "evaluate"
  | "hook_due"
  | "dream_due"
  | "remember_fact"
  | "respond"
  | "compose_surface";

export type CommandKind =
  | "restore"
  | "train"
  | "evaluate"
  | "continual_learn"
  | "dream"
  | "remember"
  | "respond"
  | "compose";

export const DISPATCH = {
  resume_train: "train",
  fresh_train: "train",
  restore: "restore",
  evaluate: "evaluate",
  hook_due: "continual_learn",
  dream_due: "dream",
  remember_fact: "remember",
  respond: "respond",
  compose_surface: "compose",
} as const satisfies Record<SituationTag, CommandKind>;

/** Autognosis folds into the existing reducer; it never adds a train path. */
export type AutognosisCommandKind = Exclude<CommandKind, "compose">;

export type AutognosisLayer = "l0_observation" | "l1_pattern" | "l2_meta";

export const AUTOGNOSIS_LAYERS = [
  "l0_observation",
  "l1_pattern",
  "l2_meta",
] as const satisfies readonly AutognosisLayer[];

export type AutognosisObservation = {
  readonly layer: AutognosisLayer;
  readonly id: string;
  readonly kind: string;
  readonly path: string;
  readonly present: boolean;
  readonly required_for?: string | null;
};

export type AutognosisReport = {
  readonly lineage: { readonly kind: LineageVerdict["kind"] };
  readonly train_ready: boolean;
  readonly next_command: AutognosisCommandKind;
};

const AUTOGNOSIS_TO_SITUATION = {
  restore: "restore",
  evaluate: "evaluate",
  continual_learn: "hook_due",
  dream: "dream_due",
  remember: "remember_fact",
  respond: "respond",
  train: "respond",
} as const satisfies Record<AutognosisCommandKind, SituationTag>;

export function situationFromAutognosis(
  report: AutognosisReport
): SituationTag {
  const verdict = report.lineage.kind;
  if (
    verdict === "divergent" ||
    verdict === "forked" ||
    verdict === "metadata_only" ||
    !report.train_ready
  ) {
    return "restore";
  }
  return AUTOGNOSIS_TO_SITUATION[report.next_command];
}

export function commandFromAutognosis(
  report: AutognosisReport
): AutognosisCommandKind {
  const dispatched = DISPATCH[situationFromAutognosis(report)];
  if (dispatched === "compose" || dispatched === "train") {
    return "respond";
  }
  return dispatched;
}

export type NanEchoDataRef =
  | { readonly kind: "runtime_surface"; readonly path: string }
  | PreparedCorpus
  | PrepFailure
  | { readonly kind: "missing"; readonly tried: readonly string[] };

export type OscillatorState = {
  readonly phases: NonEmpty<number>;
  readonly amplitudes: NonEmpty<number>;
};

export type VorticogAgentType = "persona" | "need" | "dreamcog" | "erebus";

export type NeedKind =
  | "energy"
  | "social"
  | "safety"
  | "curiosity"
  | "coherence";

export type VorticogAgent =
  | {
      readonly type: "persona";
      readonly dimension: PersonaDimension;
    }
  | {
      readonly type: "need";
      readonly need: NeedKind;
    }
  | {
      readonly type: "dreamcog";
      readonly memoryOp: Extract<MemoryOperation, { kind: "dream" }>;
    }
  | {
      readonly type: "erebus";
      readonly source: "sidecar_event";
    };

export type TinyInferenceResult =
  | {
      readonly kind: "generated";
      readonly text: string;
      readonly backend: "nanecho_runtime";
    }
  | { readonly kind: "unavailable"; readonly reason: string };

export type Mech0MemoryType =
  | "semantic"
  | "episodic"
  | "procedural"
  | "autognosic";
