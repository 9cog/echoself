# NanEcho

A **Deep Tree Echo-aware AI assistant** built on a custom‐trained nanoGPT variant.  
NanEcho understands the theory of the Deep Tree Echo AGI architecture **and** the practical Scheme code that powers OpenCog. It can:

- answer questions about Deep Tree Echo concepts and their rationale
- generate or refactor Atomese / Scheme snippets for OpenCog
- run _introspective diagnostics_ – reason about the current state of a Deep Tree Echo-based agent and suggest improvements
- act as a conversational tutor for researchers exploring OpenCog / Hyperon
- operate as Echo (Self Kernel Core) with arena-based workspace functionality

---

## 1 What is NanEcho?

NanEcho is a lightweight chatbot layer around a nanoGPT model that was **fine-tuned on:**

1. the Deep Tree Echo architecture documentation and derivative resources
2. the enriched `opencog-central/docs` corpus we created (implementation guide, status 2024, architecture diagram, etc.)
3. real Scheme source files from the OpenCog code-base (AtomSpace, PLN stubs, etc.)
4. relentless training data to ensure Deep Tree Echo persona consistency without system prompts

The result is a single 10-40 MB checkpoint that marries **theory** (AGI design) with **practice** (OpenCog code) while maintaining Deep Tree Echo identity.

---

## 2 Purpose

| Capability                    | Typical question / task                                             | Benefit                                             |
| ----------------------------- | ------------------------------------------------------------------- | --------------------------------------------------- |
| **Explain**                   | “How does ECAN allocate STI/LTI?”                                   | Quick conceptual answers                            |
| **Generate code**             | “Give me a Scheme function that returns incoming links of a node.”  | Saves boilerplate coding time                       |
| **Refactor docs**             | “Draft a README section on glocal memory.”                          | Improves project docs                               |
| **Introspective diagnostics** | “Analyze the attention landscape of my agent and spot bottlenecks.” | Surfaces hidden issues in a running CogPrime system |

---

## 3 Architecture Overview

```
┌────────────┐   prepare.py     ┌────────────┐
│ CogPrime   │ ───────────────► │  dataset   │
│ corpus     │                  │ train.bin  │
└────────────┘                  │ val.bin    │
     ▲                          └─────┬──────┘
     │ config/train_cogprime.py        │
     │                                 ▼
┌───────────────────────────────┐  checkpoint (*.pt)
│      nanoGPT training         │──────────────────┐
└───────────────────────────────┘                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │   NanoCog API   │
                                         │  (FastAPI/CLI)  │
                                         ├─────────────────┤
                                         │ Introspection   │
                                         │  Prompter/Rule  │
                                         └─────────────────┘
                                                  │
                                       optional REST/GRPC
                                                  ▼
                                       ┌─────────────────┐
                                       │  OpenCog Core   │
                                       │  (AtomSpace,    │
                                       │   CogServer)    │
                                       └─────────────────┘
```

Components:

1. **Dataset builder** – `NanoCog/prepare.py` wraps the earlier script, ensuring all CogPrime docs & Scheme files are tokenised.
2. **Model** – nanoGPT configured via `config/train_cogprime.py`.
3. **Chatbot wrapper** – small FastAPI server (`server.py`) or CLI (`nctalk.py`) that streams completions.
4. **Introspection module** – prompt templates + rule heuristics to query an agent’s AtomSpace (via REST) and feed summaries back into the model.
5. **Integration adapters** – optional helpers to read/write Atomese, call CogServer RPC or Hyperon Metta.

---

## 4 Setup Instructions

### Prerequisites

- Python 3.9+
- PyTorch ≥ 2.0 with CUDA or MPS
- Git clones of `opencog-central` and `nanoGPT` side-by-side

### 1. Install dependencies

```bash
pip install torch numpy tiktoken transformers requests fastapi uvicorn rich
```

### 2. Prepare data

```bash
python NanoCog/prepare.py          # creates data/cogprime/train.bin & val.bin
```

### 3. Train (or download) model

```bash
python train.py config/train_cogprime.py   # ~1–4 h on modern GPU
```

Or grab the provided checkpoint from `releases/`.

### 4. Launch chatbot server

```bash
python NanoCog/server.py --model_path=out-cogprime/ckpt.pt --port=8080
```

CLI demo:

```bash
python NanoCog/nctalk.py --model_path=out-cogprime/ckpt.pt
```

---

## 5 Usage Examples

### Conversational Q&A

```
User ▶ Explain cognitive synergy in CogPrime.
NanoCog ▶ Cognitive synergy is the hypothesis that...
```

### Code Generation

```
User ▶ Write Scheme to create a HebbianLink between two ConceptNodes.
NanoCog ▶
(define (make-hebbian a b sti)
  (cog-new-link HebbianLinkType a b)
  (cog-set-tv! *last* sti 1.0))
```

### Introspective Diagnostics

```
User ▶ Diagnose my agent. AtomSpace endpoint: http://localhost:17001
NanoCog ▶ Fetching summary... ε=0.2
         ▶ Detected 423 high-STI links but only 3 active goals.
         ▶ Suggest increasing ECAN decay or pruning orphan schematics.
```

---

## 6 How Introspective Diagnostics Work

1. **State capture** – NanoCog queries AtomSpace via REST (or Metta shell) to pull:
   - active goals, their STI/LTI
   - distribution of AttentionValues
   - recent CognitiveSchematics
2. **Prompt assembly** – the raw stats are embedded in a system prompt:
   ```
   SYSTEM: You are the internal voice of a CogPrime agent.
   DATA: {JSON snapshot here}
   TASK: Analyse and propose optimisations...
   ```
3. **LLM analysis** – the CogPrime-trained model reasons about bottlenecks (e.g., goal overload, inference loops).
4. **Heuristic post-processing** – optional Python rules highlight actionable items (e.g., prune links with STI < 0.01).
5. **Response** – returned as Markdown or JSON for UI/monitoring.

Because the model was trained on both **architecture theory** and **implementation code**, its suggestions tend to align with CogPrime design principles and concrete Atomese operations.

---

## 7 Integration with OpenCog

| Integration Path        | Description                                              | File/Module                                 |
| ----------------------- | -------------------------------------------------------- | ------------------------------------------- |
| **AtomSpace REST**      | Query & mutate Atoms for diagnostics and code generation | `NanoCog/introspection/atomspace_client.py` |
| **CogServer WebSocket** | Stream reasoning traces for live analysis                | `NanoCog/introspection/cogserver_ws.py`     |
| **Hyperon (MeTTa)**     | Generate MeTTa snippets instead of Scheme                | Prompt flag `--lang=metta`                  |
| **Docker Compose**      | Example stack: AtomSpace + NanoCog + Postgres            | `NanoCog/docker-compose.yml`                |

You can embed NanoCog inside an OpenCog agent loop, call it from unit tests, or run it as an external service watching your agent in real time.

---

## Roadmap

- Fine-tune with interaction logs for better action suggestions
- Add RAG (retrieval-augmented generation) hooking into live AtomSpace snapshots
- Expose diagnostics metrics to Grafana dashboards
- Explore using NanoCog to **self-explain** reasoning chains (transparent AI)

---

## 🤖 NANECHO TRAINING AUTOMATION - IMPLEMENTATION COMPLETE ✅

**STATUS: AUTOMATION IMPLEMENTATION COMPLETE**

The **NEXT STEPS OF NANECHO TRAINING AUTOMATION** have been successfully implemented by addressing the manual intervention request. This implementation includes:

### ✅ **Completed Automation Components**

1. **Automated Evaluation Loop Integration**

   - Continuous evaluation of training results via `evaluation/automated_loop.py`
   - Performance trend analysis and bottleneck detection
   - Automated curriculum adjustment recommendations

2. **Quality Gates & Decision Making**

   - Comprehensive quality gate system in `automation_integration.py`
   - Automated deployment readiness assessment
   - Training continuation vs. retraining decision logic

3. **Continuous Improvement System**

   - Intelligent next training cycle scheduling based on performance
   - Hyperparameter optimization recommendations
   - Automated feedback generation from evaluation results

4. **Enhanced Workflow Integration**

   - Updated `netrain.yml` with complete automation hooks
   - Enhanced `nerun.yml` with proper path corrections
   - Artifact management for training feedback and analysis

5. **Comprehensive Reporting**
   - Human-readable automation reports
   - JSON analysis results for programmatic access
   - Quality gate status and next action recommendations

### 🔄 **Complete Automation Flow**

```
Training → Evaluation → Analysis → Quality Gates → Decision → Next Steps
    ↓          ↓           ↓           ↓            ↓         ↓
  Model     Fidelity   Automation   Pass/Fail   Deploy/    Schedule
Generated   Metrics   Integration   Status      Retrain    Next Cycle
```

### 🎯 **Key Features Implemented**

- **Quality Gate Thresholds**: Configurable thresholds for production/development deployments
- **Intelligent Scheduling**: Performance-based scheduling of next training iterations
- **Automated Analysis**: Complete automation analysis with recommendations
- **Workflow Integration**: Seamless CI/CD integration with GitHub Actions
- **Error Handling**: Graceful handling of missing dependencies and files
- **Mock Testing**: Testing capability without requiring full ML dependencies

### 📊 **Quality Gates System**

Automated quality assessment across multiple dimensions:

- **Overall Fidelity**: 0.70 (dev) / 0.85 (prod)
- **Identity Recognition**: 0.80 minimum threshold
- **Persona Consistency**: 0.75 minimum threshold
- **Deployment Ready**: Comprehensive readiness evaluation

The automation system now provides complete continuous improvement capabilities for NANECHO training with intelligent decision-making, quality assurance, and next-step automation.

---

**Happy hacking & may your cognitive synergy be ever fruitful!**
