# EchoSelf → u9n Integration Guide

**Date:** 2026-03-25
**Branch:** `claude/integrate-echoself-angelclaw-H04RL`

## Overview

This document describes how echoself (9cog/echoself) cognitive services are integrated into the u9n (orgitcog/u9n) unified cognitive framework. The integration bridges echoself's TypeScript/Python web-based architecture into u9n's C++ Unreal Engine cognitive core.

## Service Mapping

| echoself Service            | u9n Component               | Location                                          |
| --------------------------- | --------------------------- | ------------------------------------------------- |
| `DeepTreeEchoService`       | `UEchoSelfIntegration`      | `Source/EchoSelf/EchoSelfIntegration.h/.cpp`      |
| `ToroidalCognitiveService`  | `UToroidalCognitiveAdapter` | `Source/EchoSelf/ToroidalCognitiveAdapter.h/.cpp` |
| `EchoSpaceService`          | `UEchoSpaceMemoryBridge`    | `Source/EchoSelf/EchoSpaceMemoryBridge.h/.cpp`    |
| `HypergraphBridge` (Python) | `UHypergraphBridgeAdapter`  | `Source/EchoSelf/HypergraphBridgeAdapter.h/.cpp`  |
| NanEcho pipeline            | `FNanEchoTrainingConfig`    | `Training/NanEchoConfig.h`                        |

## Detailed Mapping

### DeepTreeEchoService → UEchoSelfIntegration

| TypeScript                      | C++                                 |
| ------------------------------- | ----------------------------------- |
| `DTEOptions.creativityLevel`    | `EEchoSelfCreativityLevel` enum     |
| `DTEOptions.toroidalMode`       | `SetToroidalMode(bool)`             |
| `EchoResonancePattern`          | `FEchoResonancePattern` struct      |
| `CycleMetrics`                  | `FEchoSelfCycleMetrics` struct      |
| `initializeResonancePatterns()` | `InitializeCoreResonancePatterns()` |
| `recordCycleMetrics()`          | `RecordCycleMetrics()`              |
| `getImprovementTrend()`         | `GetImprovementTrend()`             |
| `adaptResonancePatterns()`      | `AdaptResonancePatterns()`          |

### ToroidalCognitiveService → UToroidalCognitiveAdapter

The toroidal model treats cognition as flowing on a torus surface where Deep Tree Echo (right/creative) and Marduk (left/analytical) hemispheres continuously exchange information.

- Phase-locked oscillation: DTE peaks at phase 0, Marduk at phase PI
- Coherence tracking for inter-hemisphere synchronization
- Insight generation at phase transition boundaries
- Connects to u9n's `StrategicCognitionBridge` for hemisphere coordination

### HypergraphBridge → UHypergraphBridgeAdapter

| Python (hypergraph_bridge.py)    | C++                                       |
| -------------------------------- | ----------------------------------------- |
| `calculate_salience()`           | `CalculateSalience()`                     |
| `adaptive_attention_threshold()` | `GetAdaptiveAttentionThreshold()`         |
| `get_repository_files()`         | N/A (UE5 equivalent via asset management) |
| `create_cognitive_prompt()`      | `CreateCognitivePrompt()`                 |
| AtomSpace nodes                  | `FHypergraphNode` struct                  |
| Hyperedges                       | `FHypergraphEdge` struct                  |

### EchoSpaceService → UEchoSpaceMemoryBridge

| TypeScript                              | C++                              |
| --------------------------------------- | -------------------------------- |
| pgvector cosine similarity              | `CosineSimilarity()` (in-memory) |
| Memory types (episodic, semantic, etc.) | `EEchoMemoryType` enum           |
| Memory fragments                        | `FMemoryFragment` struct         |
| Vector retrieval                        | `RetrieveSimilar()`              |
| Consolidation                           | `ConsolidateMemories()`          |

### NanEcho Pipeline → FNanEchoTrainingConfig

The NanEcho training pipeline continues to run as an external Python process. The `FNanEchoTrainingConfig` struct in u9n holds the configuration that parameterizes training runs, including:

- Base model (GPT-2), sequence length, vocabulary size
- ESN augmentation parameters (reservoir size, spectral radius, leak rate)
- Training hyperparameters (learning rate, batch size, epochs)
- Persona enforcement settings
- CI/CD integration (HuggingFace repo, cycle interval)

## Build Integration

The EchoSelf module is defined in `Source/EchoSelf/EchoSelf.Build.cs` with dependencies on:

- `Core`, `CoreUObject`, `Engine` (UE5 fundamentals)
- `DeepTreeEcho` (u9n native cognitive module)
- `AngelClaw`, `ReservoirEcho` (companion integration modules)

## Architecture Diagram

```
echoself (TypeScript/Python)          u9n (C++/UE5)
┌─────────────────────────┐          ┌──────────────────────────┐
│ DeepTreeEchoService     │ ──────── │ UEchoSelfIntegration     │
│ ToroidalCognitiveService│ ──────── │ UToroidalCognitiveAdapter│
│ EchoSpaceService        │ ──────── │ UEchoSpaceMemoryBridge   │
│ HypergraphBridge (Py)   │ ──────── │ UHypergraphBridgeAdapter │
│ NanEcho Pipeline (Py)   │ ──────── │ FNanEchoTrainingConfig   │
└─────────────────────────┘          └──────────────────────────┘
                                              │
                                              ▼
                                     StrategicCognitionBridge
                                     NestedTensorPartitionSystem
                                     NeurochemicalSystems
```
