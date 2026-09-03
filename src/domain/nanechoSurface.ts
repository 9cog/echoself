import { existsSync } from "node:fs";
import { join } from "node:path";

import type { NanEchoDataRef, TinyInferenceResult } from "./echoZero.ts";

export const NANECHO_CANDIDATE_PATHS = [
  "echoself/data/nanecho",
  "NanEcho/data/nanecho",
  "data/nanecho",
] as const;

const PREPARED_EVIDENCE = ["train.bin", "train.txt", "metadata.json"] as const;

export type ExistsFn = (path: string) => boolean;

export function resolveNanechoSurface(
  repoRoot: string,
  exists: ExistsFn = existsSync
): NanEchoDataRef {
  const tried: string[] = [];
  for (const relative of NANECHO_CANDIDATE_PATHS) {
    const absolute = join(repoRoot, relative);
    tried.push(relative);
    if (!exists(absolute)) {
      continue;
    }
    const prepared = PREPARED_EVIDENCE.some(name =>
      exists(join(absolute, name))
    );
    if (prepared) {
      return { kind: "prepared", source: relative };
    }
    return {
      kind: "failed",
      reason: `${relative} exists as a runtime surface but has no prepared corpus (train.bin|train.txt|metadata.json)`,
    } as NanEchoDataRef;
  }
  return { kind: "missing", tried };
}

export type TinyInferenceClient = {
  init: () => Promise<void>;
  load: (checkpointPath: string | null) => Promise<void>;
  generate: (prompt: string) => Promise<TinyInferenceResult>;
};

export function createTinyInferenceClient(): TinyInferenceClient {
  let checkpoint: string | null = null;
  let ready = false;

  return {
    async init() {
      ready = true;
    },
    async load(checkpointPath) {
      if (!ready) {
        throw new Error("tiny inference client must init before load");
      }
      checkpoint = checkpointPath;
    },
    async generate(prompt: string) {
      if (!ready) {
        throw new Error("tiny inference client must init before generate");
      }
      if (!checkpoint) {
        return {
          kind: "unavailable",
          reason: "no local NanEcho .pt checkpoint; refusing model download",
        };
      }
      if (!prompt.trim()) {
        return { kind: "unavailable", reason: "empty prompt" };
      }
      return {
        kind: "unavailable",
        reason: `checkpoint declared at ${checkpoint} but generate is bound only when weights are actually loadable`,
      };
    },
  };
}
