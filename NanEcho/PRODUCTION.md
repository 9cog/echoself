# NanEcho production runtime

NanEcho inference now uses `NanEcho/runtime.py` everywhere. CLI, API, evaluation,
and export all load real `NanEchoModel` weights and share the GPT-2 `tiktoken`
encoding. There is no canned-response or character-token inference fallback.

## Checkpoints

Supported checkpoints contain:

- `model_state_dict`
- complete architecture metadata in `config` (standard trainer) or
  `model_config` (cached trainer)
- a vocabulary at least as large as GPT-2's 50,257 tokens
- explicit `tokenizer` provenance with `name: gpt2`, `vocab_size: 50257`,
  `eos_token: <|endoftext|>`, and `eos_token_id: 50256`

Tokenizer provenance is never inferred from `vocab_size`: a legacy
character-tokenized checkpoint may also claim 50,257 entries. Checkpoints with
missing, incomplete, or incompatible tokenizer metadata are rejected. The
standard trainer, training cache, and native exporter write this declaration.
nanoGPT checkpoints are also rejected.

Only load trusted local checkpoint files. Compatibility with older trainer
artifacts requires PyTorch pickle loading when restricted loading rejects
NumPy metric scalars.

## Data preparation and training

```bash
python NanEcho/prepare_nanecho.py \
  --source_root . \
  --output_dir data/nanecho \
  --min_total_tokens 4096 \
  --min_split_tokens 256

python train_nanecho.py --data_dir data/nanecho --out_dir out-nanecho
```

The builder excludes tests, evaluation code, generated artifacts, caches, and
output trees. It assigns whole authored files and synthetic document families to
deterministic `train`, `val`, or held-out `test` groups _before_ chunking, so
chunks from one source cannot leak across splits. Persona derivatives retain the
original sample's document group, so an original and every transformed variant
always remain in the same split. Persona, reinforcement, and Deep Tree Echo
weights accept `0.0` through `1.0` at one decimal place. Each accepted step maps
to a distinct exact integer occurrence count; greater precision is rejected
rather than reported as an effective control that produces identical training
data. A nonzero Deep Tree Echo weight with its mode disabled is also rejected.
Metadata records group membership, exact occurrence controls, and token counts.
Configured corpus minimums are enforced and training never creates fallback data.
Datasets and newly written checkpoints use the same complete tokenizer provenance
object. Older corpus metadata with a tokenizer name but no vocabulary/EOS fields
must be regenerated before training.

New models instantiate and consume all eight configured persona dimensions.
Training progressively changes their weights by phase. Measured underperformance produces JSON feedback in
`eval-nanecho/persona_feedback/`. Completion of configured iterations is not
reported as convergence or persona mastery.

## Inference and evaluation

```bash
python NanEcho/netalk.py --model_path out-nanecho/best_model.pt
python NanEcho/neserver.py --model_path out-nanecho/best_model.pt --port 8081

python NanEcho/evaluation/echo_fidelity.py \
  --model_path out-nanecho/best_model.pt \
  --heldout_path data/nanecho/test.txt \
  --output_path fidelity_report.json
```

The API is same-origin only by default. Set `NANECHO_CORS_ORIGINS` to a
comma-separated list of trusted browser origins when cross-origin access is
required.

Evaluation reports actual generated-output scores, behavioral consistency,
held-out perplexity, zero-system-prompt results, and a prompted comparison.
Strided perplexity masks overlap so every held-out target is scored exactly once
with available prior context. Generation without a seed advances the process RNG
normally; an explicitly seeded request is deterministic and restores global RNG
state afterward. Seeded requests pass a request-local `torch.Generator` through
both model-side stochastic pattern injection and token sampling, so concurrent
seeded server requests do not race over process-global RNG state.
The server exposes rolling lexical persona coverage as drift monitoring. These
metrics are diagnostics, not proof of cognition or convergence.

Historical checkpoints that actually contain only the original first four
persona modules remain loadable; runtime metadata reports those four as active.
They cannot gain the other four dimensions without retraining or migration.

## Exports

```bash
python NanEcho/export_model.py \
  --model_path out-nanecho/best_model.pt \
  --output_dir exports/nanecho

# Attempt ONNX logits export when the installed PyTorch/ONNX stack supports it:
python NanEcho/export_model.py \
  --model_path out-nanecho/best_model.pt \
  --output_dir exports/nanecho --onnx
```

Native PyTorch is the canonical deployment format. ONNX export fails explicitly
when unsupported. NanEcho does not emit GGUF because this custom architecture
has no verified llama.cpp-compatible conversion.
