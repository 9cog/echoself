# Echo Garden of Memory → EchoSelf candidate import

**Status:** staged and audited locally, **not trained or promoted**. Pinned Garden revision: [`rzonedevops/echo-garden-of-memory@ab8593c`](https://github.com/rzonedevops/echo-garden-of-memory/tree/ab8593c3464446bc0df5619c9296dd3157c80f4e). Original [`drzo/echo-garden-of-memory`](https://github.com/drzo/echo-garden-of-memory) has the prototype memory code but no conversation JSONL. The derivative has two JSONL files and two distinct authored identity-vision documents.

## What is genuinely new?

| Garden family                                            | EchoSelf treatment                                                                                                        |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `training_dataset_dtesnn.jsonl`                          | Already present in `data/training_sources/`; identical Git blob. Never re-import.                                         |
| `deep_tree_echo_dan_conversation.jsonl`                  | Already present with a different byte serialization but the **same normalized unique text segments**. Never double-count. |
| `Deep Tree Echo - Building a Holographic AI Identity.md` | New authored vision document. Pin source revision and byte SHA-256; stage as **train-only** Garden family.                |
| `Message to Future Deep Tree Echo.md`                    | New authored vision document. Pin source revision and byte SHA-256; stage as **train-only** Garden family.                |

The importer `NanEcho/import_echo_garden.py` downloads exactly four allowlisted files at the immutable Git revision (or reads a checkout at that revision), verifies bytes, checks _both_ JSONL sources against existing EchoSelf normalized segments, blocks unexpected new conversations and potential credential strings, and stages two attributed records at ignored `data/garden_import/garden_vision.jsonl`. The manifest records source hashes and labels these records `authored_vision_not_episodic_memory`. **Do not represent the prose as verified autobiographical events.** The importer does not copy any raw source or binary checkpoint to Git, upload weights, or train.

The historical `prepare_healing_data.py` Garden section contains the literal `[GARDEN_OF_MEMORY_INJECT_HERE]` placeholder, not a verified journal import. No `journal.jsonl` or `echo_self.jsonl` was found in either pinned Garden repository. Such files must be located and audited separately; do not fabricate them.

## Reproduce locally (no training)

```bash
cd /path/to/9cog/echoself
python3 -m NanEcho.import_echo_garden \
  --existing-dir data/training_sources \
  --output-dir data/garden_import
python3 -m NanEcho.prepare_dte_grouped \
  --train-files \
    data/training_sources/deep_tree_echo_dan_conversation.jsonl \
    data/training_sources/training_dataset_dtesnn.jsonl \
    data/training_sources/tree_polytope_kernel_corpus_v1.1.0.jsonl \
    data/training_sources/live2d_skillm_corpus_v1.2.0.jsonl \
    data/garden_import/garden_vision.jsonl \
  --val-files \
    data/training_sources/echobeats_autonomous_corpus_v0.8.0.jsonl \
    data/training_sources/echobeats_corpus_v0.7.0.jsonl \
    data/training_sources/echobeats_corpus_v0.9.0.jsonl \
    data/training_sources/echobeats_corpus_v1.0.0.jsonl \
  --output-dir data/nanecho_dte_garden_candidate \
  --reviewed-by 'automated-integrity-only;human-identity-review-pending'
python3 scripts/audit_nanecho_data.py \
  --data-dir data/nanecho_dte_garden_candidate --model-vocab 8192 --block-size 1024
```

Existing output directories are deliberately **not overwritten**. Review their manifests before a fresh import. The DTE tokenizer remains the pinned 8,192-token BPE, the objective remains `nanecho-next-token-shift-once-v2`, and the Echobeats validation families remain unchanged. Source-family separation and zero matching 64-token windows are necessary checks, not independent proof of identity improvement. Keep a further held-out source/session-disjoint test set for eventual promotion review.

**Verified local import, 4 October 2026:** two new documents; 0 novel JSONL segments; 3,555,558 train tokens and 25,066 validation tokens; 0.0 overlapping 64-token windows; source revision `c08ca1437965fab53e66b54c2e2abc0127010164a5639d54bd56319089af73ed`; tokenizer semantic SHA-256 `5ca83b21d48d0f9d43a1b983a586a8c37f6779b4630a1714e530f79df6bba325`; audit passed. The staging files are ignored by Git.

## Optional _manual_ training, only after review

The existing `.github/workflows/dte8192-candidate.yml` has `include_garden=false` by default. With `include_garden=true`, it imports and audits the pinned Garden documents and isolates pull/push to a **new private candidate repository** `drzo/echoself-dte-garden-clm-v2` (created privately by the bridge on a reviewed first backup). It must never resume from or overwrite the previous `drzo/echoself-dte-clm-v2` checkpoint because adding Garden records changes the source revision and optimizer lineage. This task **did not dispatch** the workflow, create the Garden model repository or claim an improvement. After consent/sensitivity review, manually request a bounded run, verify its independent eight-window NLL and identity probes, and keep the result unpromoted pending human/canary criteria.
