/**
 * EchoLayla Service Module
 *
 * Main entry point for the EchoLayla AI assistant integration.
 * Now includes Triple-Loop Learning for adaptive character evolution.
 */

export * from "./types.ts";
export * from "./characters.ts";
export { EchoLaylaService, getEchoLaylaService } from "./echoLaylaService.ts";

// Triple-Loop Learning exports
export * from "./tripleLoopLearningTypes.ts";
export {
  TripleLoopLearningService,
  getTripleLoopLearningService,
} from "./tripleLoopLearningService.ts";

// NanEcho Training Integration exports
export {
  NanEchoTrainingIntegration,
  getNanEchoTrainingIntegration,
  type NanEchoTrainingParams,
  type TrainingMode,
  type TrainingCycleResult,
} from "./nanechoTrainingIntegration.ts";
