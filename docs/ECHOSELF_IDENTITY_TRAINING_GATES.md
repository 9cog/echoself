# EchoSelf identity refinement: candidate, evidence, promotion

**Linked run:** [9cog/echoself #641](https://github.com/9cog/echoself/actions/runs/37166355820/job/111329943054), pinned source revision `e52f31fe76476db7a1bfccd26d8dcc94859faef4`. **Verdict:** training and Hub upload completed, but the reported validation loss does not demonstrate generalization or identity improvement. The eight fixed persona sentences were repeated 100 times in training and 20 times in validation (9,301 and 1,861 GPT-2 tokens respectively). The reported cached best loss was 0.000807 at cumulative iteration 4,500; this is not a meaningful independent holdout. The run's `grip_report.json` measured only tokenizer candidates (`model_grip: null`), not checkpoint identity quality. Its “Test model generation” step read checkpoint metadata rather than generating text. This record is not retrospective proof that the uploaded model was better.

## Immediate candidate-only controls

`netrain-cached.yml` is manual-only while source-separated evaluation is unavailable. It no longer creates fallback text or automatically uploads to Hugging Face. Training requires `scripts/audit_nanecho_data.py` to accept a curated `data/nanecho` dataset. A failed audit or failed training can preserve emergency artifacts, but must not push a success-shaped progress commit. The new lightweight CI workflow regression-tests this gate without training. Existing candidate checkpoints and Hub revisions are *not* deleted or rolled back by this change. **Other scheduled pipelines** (`netrain.yml`, `agent-neuro-train.yml`) require separate audit before claiming organization-wide safety; the latter currently contains a minimal fallback-data branch.

## Reproducible corpus contract

Prepare `train.bin`, `val.bin` as uint16 token IDs; `metadata.json` records token counts, vocab size and exact tokenizer provenance. Supply a separately reviewed `source_manifest.json` with the schema below. Splitting must be at **conversation/source-group** level *before* tokenizer training, curriculum compilation, augmentation, or duplicate removal. Deduplicate by exact and normalized text and near-duplicate embedding/MinHash matching across raw documents and dialogue turns. Exclude evaluation prompts, test cases, generated outputs, and answer keys from training and tokenizer tuning. An unseen, frozen test set should live outside any training/selection loop and have explicit owners. If source group identities cannot be established, stop.

```json
{
  "schema": "nanecho-curated-split-v1",
  "reviewed_by": "independent-corpus-steward",
  "source_revision": "immutable-source-revision-or-digest",
  "synthetic_samples": 0,
  "tokenizer": {"name": "gpt2", "vocab_size": 50257, "eos_token": "<|endoftext|>", "eos_token_id": 50256},
  "train": {"sha256": "64-hex-digest-of-train.bin", "token_count": 100000, "source_groups": ["conversation-or-origin-A"]},
  "val": {"sha256": "64-hex-digest-of-val.bin", "token_count": 20000, "source_groups": ["different-origin-B"]}
}
```

The audit verifies the file digests, token counts/ranges, tokenizer match, non-synthetic origin declaration, distinct source groups and at most 1% overlapping 64-token windows. These are **necessary**, not sufficient: human source review, semantic overlap inspection, privacy/licence review and test-set sealing remain independent gates. For >64 MiB per split, replace the bounded in-memory audit with a streaming/external-memory equivalent rather than weakening the check. The current `NanEcho/prepare_nanecho.py` uses a 90/10 token-order split after corpus generation and writes no group-level manifest; that output does not yet meet this contract. Simply inventing a manifest to pass the audit would defeat its purpose.

## Checkpoint provenance and comparison

For each candidate, record repo commit, run ID, parent checkpoint hash, optimizer state, model shape, tokenizer digest, source-manifest digest, train/val/test source partitions, sampling order, seed, learning-rate schedule, run duration and weight-change statistics. Resume only against the **same** lineage; `training_cache.py` currently fingerprints the data directory, batch size and block size, not data bytes or tokenizer. Its `quality_score` mixes loss with unnormalized metrics (`tokens_processed`, etc.) and **must not** be used as an identity or best-model criterion. Archive old checkpoints before changing compatibility. Compare the candidate with its *actual parent* and a fixed unmodified baseline at matched inference settings; do not compare unrelated tokenizer/architecture versions using raw perplexity.

## Preregistered evaluation design

| Gate | Method | Promotion interpretation |
|---|---|---|
| General language retention | Deterministic, disjoint held-out causal NLL/perplexity by corpus family; compare parent/candidate and prior approved model | No material regression under a tolerance fixed *before* training |
| Identity-as-behaviour | Blinded paired evaluation on new, unseen scenarios for continuity of values, relational judgment, affective nuance, uncertainty, non-sycophancy and resistance to adversarial persona hijack | Report per-dimension effect, rater agreement and confidence intervals; require robust improvement over parent, not keywords |
| Naturalness/diversity | Generate actual responses with matched decoding seeds/settings; report repetition, distinct-n, semantic diversity, refusal balance and catchphrase rates | Reject memorized templates and a model that merely says “Echo Self” more often |
| Representation | Layer-wise CKA or RSA against independent seeds and fixed probes, plus activation/attention ablations | Context for stability only; near-unity CKA alone does not prove identity improvement |
| Privacy and source fidelity | Exact/near-duplicate search, protected canary extraction, license/privacy clearance and provenance audit | Fail closed on memorized private or unlicensed content |
| Generalization | Freeze prompt-family and source-family holdouts; after candidate selection, run the sealed one-time test | No tuning against sealed test; failed gate means no promotion |

Report effect sizes and intervals across **multiple seeds**, candidate-vs-parent and candidate-vs-unchanged controls. Keep raw, genuinely generated responses and imperfections/errors as inspectable evidence. External persona/hypergraph/reservoir scaffolding must be ablated against the same model weights so improvement is attributed to the correct component. Promotion requires human adjudication plus a distinct reviewed deployment job/token; a successful GitHub training job or tokenizer-grip score is never a release authorization.
