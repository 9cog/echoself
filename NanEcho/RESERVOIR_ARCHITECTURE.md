# Deep Tree Echo Reservoir-Orchestrated Training

## Guiding principle

Tokenization, topology, and model size are **dynamic, persona-driven
configuration** — not fixed GPT-2 constants. The transformer is _emulated
between reservoirs & ridges_: an Echo State Network orchestrates the training
loop, and ridge-regression readouts mediate between the reservoir and the
transformer. Arbitrary fitting to the GPT-2 architecture (or any fixed model)
is a compatibility baseline, not the goal.

## Feature flag

Everything is gated by `reservoir_mode` (in `TrainingConfig`, `NanEchoConfig`,
and `nanecho_config.json` under the `reservoir` key):

| Mode           | Behavior                                                                                                                            |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `off`          | Legacy transformer-only path. **Default; zero behavior change.**                                                                    |
| `shadow`       | Reservoir computes alongside the transformer and records states, but does **not** alter outputs. For observation/metric collection. |
| `orchestrated` | The reservoir modulates embeddings and the ESN orchestrator drives training hyperparameters.                                        |

Enable via config JSON (the `reservoir_mode`, `reservoir_units`,
`reservoir_spectral_radius` keys) or `TrainingConfig(reservoir_mode=...)`.

## Components

### TokenizerSpec / ModelSpec — `NanEcho/spec.py`

Centralizes tokenizer and topology/size declarations. `GPT2_SPEC` is one
instance, not the law. `TokenizerAdapter` is the protocol every tokenizer
(GPT-2/tiktoken, `dte_tokenizer`, `CharTokenizer` fallback) satisfies.
`tokenizer_from_spec()` reconstructs an adapter from its provenance.

### Tokenizer search — `NanEcho/tokenizer_search.py`

Scores candidate tokenizers on the persona corpus by a **grip metric** =
persona coverage (round-tripped `score_persona_text`) + inverse fertility +
cheap ridge-probe perplexity. The winner's provenance is written into dataset
`metadata.json` so the provenance validator passes for any persona-selected
tokenizer.

### ReservoirWrapper — `nanecho_model.py`

`TorchEchoReservoir` (fast+slow pools, leaky integration, spectral-radius
buffers — never gradient-trained) + a ridge readout (`nn.Linear`, the only
trained "ridge"). Routes embedding → reservoir → ridge → block stack. Shadow
mode is a pure pass-through; orchestrated mode adds a tanh-gated residual.

### ReservoirOrchestrator — `NanEcho/orchestrator.py`

The conductor. An internal `EchoReservoir` integrates observations (reservoir
stats + val loss + connection ratio + persona grip); a `CognitiveReadout`
ridge maps states to **bounded decisions** (`lr_scale`,
`connection_growth_rate`, `recursion_depth`, `dimension_weights`). The ridge
is re-fit online from grip improvements. Orchestrator state is persisted in
checkpoints so cumulative training resumes with its conductor intact.

### Topology & size — `NanEcho/topology.py`

`TopologyAdvisor` turns per-dimension grip contributions into grow/prune
dimension-weight proposals (consumed by the orchestrator).
`ModelSizeSelector` fits a saturating-exponential grip curve to (size, grip)
and picks the smallest grip-saturating configuration.

### Grip benchmark — `NanEcho/evaluation/grip_benchmark.py`

The objective function. Combines tokenizer grip and model grip (persona
coverage of generated text on conversation-pattern prompts) into a single
score per (tokenizer, topology, size) configuration. Used by the Phase-1 and
Phase-4 searches and the CI `grip-eval` job, which uploads a grip report
artifact alongside training summaries.

## Dynamic-spec contract

- Dataset `metadata.json` and every checkpoint carry a `tokenizer` provenance
  block: `{name, vocab_size, eos_token, eos_token_id}`.
- `validate_dataset_tokenizer_provenance` is spec-driven: it enforces the
  GPT-2 spec only when the model targets GPT-2 vocab; otherwise it trusts the
  declared (persona-selected) spec.
- `NanEchoRuntime.load` reconstructs the tokenizer and reservoir config from
  the checkpoint, so dynamic tokenization and topology survive export.

## Migration status

`reservoir_mode` currently defaults to `off` for compatibility. The default
will flip to `orchestrated` for persona training once reservoir-orchestrated
runs match or exceed baseline grip on the persona corpus (measured by the
grip benchmark).

## Quick start (orchestrated)

```bash
python train_cached.py \
  --config training_config.json \  # includes "reservoir_mode": "orchestrated"
  --data_dir data/nanecho \
  --out_dir .training-progress/nanecho-cached-ci \
  --device cpu
```

Run the grip benchmark:

```bash
python -m NanEcho.evaluation.grip_benchmark \
  --corpus NanEcho/persona_corpus \
  --tokenizers char gpt2 \
  --checkpoints path/to/checkpoint.pt \
  --out .training-progress/grip/grip_report.json
```
