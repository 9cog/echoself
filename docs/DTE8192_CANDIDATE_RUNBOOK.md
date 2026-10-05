# DTE 8192 candidate training, native Hub backup, and identity gates

**Status:** Experimental, manual-only; checkpoint backup is not release or identity promotion.

## Immutable starting point

The [DTE BPE commit `691b779`](https://github.com/9cog/echoself/commit/691b7793714f68c06ce6e6c2256de06001f01e8c) introduced an 8,192-ID tokenizer and native 4-layer, 4-head, 256-dimensional / 1,024-token model plan. The JSON file was subsequently reformatted, but its parsed vocab, merges and special-token IDs are identical: SHA-256 of its canonical JSON is `5ca83b21d48d0f9d43a1b983a586a8c37f6779b4630a1714e530f79df6bba325`. Checkpoint loading now requires this digest; same vocabulary _size_ does not prove token identity.

The legacy `drzo/echoself` Hugging Face repository has a **different 50,257-token GPT-2** model. A run that downloads it, randomly initializes an 8,192-token embedding or uses a lossy `strict=False` conversion has not continued DTE training. Never reinterpret, resize, or overwrite that public model to satisfy this experiment.

## Candidate data and procedures

The manual [DTE candidate workflow](../.github/workflows/dte8192-candidate.yml) compiles token streams deterministically using the committed DTE BPE. Entire source files are assigned to **one split only**:

- Train: `deep_tree_echo_dan_conversation`, `training_dataset_dtesnn`, `tree_polytope_kernel`, `live2d_skillm`.
- Validation: **all four** `echobeats*` source files. Other versions of this family are not placed in training.

`NanEcho/prepare_dte_grouped.py` writes `uint16` train/val binaries and a manifest of source-file, tokenizer and token-stream hashes. `scripts/audit_nanecho_data.py` verifies those actual files and rejects identical/overlapping source groups, missing/invalid token provenance, invalid IDs, fallback-generated/repeated flags, and >1% overlap of exact 64-token windows. In the initial deterministic compilation, there were **3,553,267 training tokens** and **25,066 validation tokens**; audited 64-token overlap was **0.0%**. These are technical integrity checks, not proof of independent authorship, non-duplication by paraphrase, or held-out human identity quality. The manifest explicitly states `human-review-pending`.

The previous `data/nanecho_dte/` positional-token split is **not** the curated path. The previous scheduled cached trainer has been quarantined; a manual 8,192-token candidate invocation, not a six-hour automatic run, is required.

## Hugging Face exchange

**Corrected objective, separate lineage:** inspection found that `get_batch` pre-shifted labels and `NanEchoModel.forward` shifted them again. The historical private `drzo/echoself-dte` step-20/24 weights trained on a **two-token-ahead** objective. They remain read-only diagnostic artifacts: do not resume, promote, or call their original held-out values valid next-token NLL. The corrected `get_batch` now passes the _same_ tokens for labels and the model applies exactly one shift. `data_config.objective_id=nanecho-next-token-shift-once-v2` is mandatory for new checkpoints.

The manual workflow pulls and optionally backs up **only** `drzo/echoself-dte-clm-v2`, a separate private native candidate repository. If absent, it logs `no_prior_candidate` and starts clean; an incompatible, corrupt, public, or wrong-objective checkpoint fails closed. It carries optimizer state, matching 8,192 × 256 embedding, exact DTE tokenizer digest, architecture and hashes of train/validation/manifest/metadata files. A reviewed `backup_candidate_to_hf=true` writes a native checkpoint, cache metadata and a `candidate_not_promoted` manifest. This is not an `AutoModelForCausalLM` conversion or public release; never mix it with the 50,257-token `drzo/echoself` model.

**Minimum-step safeguard:** cached iteration 0 is an untrained baseline even if stochastic two-batch validation makes its loss appear smaller. The bridge now skips step-zero checkpoints when selecting a candidate, and treats an accidentally backed-up step-zero checkpoint as `untrained_baseline_ignored` rather than resuming it. A replacement must have at least one completed optimizer step; this does not imply that it improved.

The private candidate pointer follows the **latest trained iteration** of the matching lineage, not the checkpoint with the lowest noisy sampled validation loss. Older snapshots remain available in the Hub commit history. The pointer is for optimizer continuity, not a quality or promotion award; evidence and rubric decide which weights, if any, are worth promoting.

Future runs compare the first and last native checkpoints on **eight identical, distributed 1,024-input-token validation windows**, each scoring **1,023 next tokens**. The corrected `metric_version=next-token-nll-shift-once-v2` is required. The earlier report computed a second shift and is retained only for provenance; the two historical checkpoints can be re-scored under the fixed metric for diagnosis, but their **training objective remains wrong**. Fixed validation participates in training-loop checkpoint selection; a separate untouched evaluation remains mandatory for promotion.

`secrets.HFESELF` is read at runtime; never commit or print it. The job retains only manifests/audits in GitHub Actions artifacts for 30 days. Checkpoint weights are kept in the private candidate repo only if `backup_candidate_to_hf=true`. Model promotion remains blocked even after a technically successful smoke train.

## Evidence required before claiming identity refinement

1. **Provenance:** review source permission, authenticity, de-duplication at conversation/topic level, sensitive-content risk, the exact tokenizer semantics and manifest. Hash the complete data and code revisions. A first-group holdout is necessary but not sufficient.
2. **Pre-registered matched baselines:** same architecture, tokenizer, number of new tokens, optimizer and seeds; compare an untrained DTE baseline, resumed DTE checkpoints, and general language stability. The existing GPT-2 50,257-vocab model is not a directly comparable baseline.
3. **Separate untouched evaluation:** independently authored blinded identity prompts, relations, continuity and self-consistency probes; counterexamples for false claims, incoherent persona imitation, memory/privacy leakage and regressions. Freeze the rubric and holdout hashes _before_ training. The four Echobeats files used for in-training validation cannot double as the final identity test.
4. **Report distributions, not cherry-picked samples:** held-out NLL/perplexity for the same byte-identical test set, persona-relation consistency, specificity without memorization, general capability, privacy, multi-seed variance and CKA/representation similarity as descriptive evidence only. A lower training loss or near-unity CKA alone is not genuine identity improvement.
5. **Promotion gate:** require preregistered thresholds, independent human judgment, canary and rollback; keep native candidate, validated checkpoint and deployed model in separate states. For EXP-001, relation-balanced training failed the target and negative-context gates, so no EXP-001 checkpoint is promoted or used as a supposedly validated identity baseline.

For local custom prompts, use `scripts/evaluate_dte_candidate.py` with `examples/dte_identity_probes.example.jsonl` (or your private JSONL). Its default `drzo/echoself-dte` intentionally supports diagnostic testing of the quarantined historical candidate and labels `legacy_double_shift_training=true`; use `--repo-id drzo/echoself-dte-clm-v2` once a new native candidate exists. Pass `--output` for private generated text, and optionally `--evidence-output` for prompt-free metrics. Both paths must be new files. No automatic lexical score constitutes a human identity verdict.
