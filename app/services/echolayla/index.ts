/**
 * EchoLayla Service Module
 *
 * Main entry point for the EchoLayla AI assistant integration.
 * Now includes Triple-Loop Learning for adaptive character evolution.
 */

export * from "./types";
export * from "./characters";
export { EchoLaylaService, getEchoLaylaService } from "./echoLaylaService";

// Triple-Loop Learning exports
export * from "./tripleLoopLearningTypes";
export {
  TripleLoopLearningService,
  getTripleLoopLearningService,
} from "./tripleLoopLearningService";

// NanEcho Training Integration exports
export {
  NanEchoTrainingIntegration,
  getNanEchoTrainingIntegration,
  type NanEchoTrainingParams,
  type TrainingMode,
  type TrainingCycleResult,
} from "./nanechoTrainingIntegration";
