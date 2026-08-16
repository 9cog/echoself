export const MEMORY_TYPES = [
  "semantic",
  "episodic",
  "procedural",
  "autognosic",
] as const;

export type MemoryType = (typeof MEMORY_TYPES)[number];

export type SelfAspect =
  | "identity"
  | "capability"
  | "checkout"
  | "belief"
  | "self_model";

export type SemanticSpec = {
  kind: "semantic";
  concepts: string[];
  weights: Record<string, number>;
};

export type EpisodicSpec = {
  kind: "episodic";
  occurred_at: string;
  event: string;
};

export type ProceduralSpec = {
  kind: "procedural";
  instrument: string;
  signature: string | null;
  steps: string[];
};

export type AutognosicSpec = {
  kind: "autognosic";
  about: SelfAspect;
};

export type TypeSpec =
  | SemanticSpec
  | EpisodicSpec
  | ProceduralSpec
  | AutognosicSpec;

export type MemoryRecord = {
  id: string;
  type: MemoryType;
  content: string;
  created_at: string;
  confidence: number;
  pinned: boolean;
  source: string;
  metadata: Record<string, unknown>;
  spec: TypeSpec;
  dual_write_id: string | null;
  score?: number;
};

export type MemorySaveInput = {
  type: MemoryType;
  content: string;
  source?: string;
  confidence?: number;
  pinned?: boolean;
  metadata?: Record<string, unknown>;
} & Partial<
  | { concepts: string[]; weights: Record<string, number> }
  | { occurred_at: string; event: string }
  | { instrument: string; signature?: string; steps?: string[] }
  | { about: SelfAspect; subject?: string }
>;
