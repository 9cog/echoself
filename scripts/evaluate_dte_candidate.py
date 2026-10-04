#!/usr/bin/env python3
"""Evaluate the PRIVATE native NanEcho DTE candidate on operator-supplied JSONL probes.

Usage (from repo root):
  HF_TOKEN="$HFMAN" python scripts/evaluate_dte_candidate.py \
    --prompts /private/probes.jsonl --output /private/dte-results.json \
    --evidence-output /private/dte-evidence.json --max-new-tokens 24

Probe rows: {"id":"precision-01", "family":"semantic_precision",
             "prompt":"User: ...\\nEcho:", "reference":"An optional original held-out reply."}
Prompt/reference strings and generated outputs stay in --output ONLY; evidence output
has hashes and numerical diagnostics, not probe text. These diagnostics are not
identity/factuality or release grades. No model upload and no training occur here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def load_probes(path: Path) -> list[dict]:
    probes, seen = [], set()
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', item['id']):
            raise ValueError(f'Probe line {line_no}: id must be a short stable identifier')
        if item['id'] in seen:
            raise ValueError(f'Duplicate probe id {item["id"]!r}')
        seen.add(item['id'])
        if not isinstance(item.get('prompt'), str) or not item['prompt'].strip():
            raise ValueError(f'Probe {item["id"]}: nonempty prompt required')
        if not isinstance(item.get('family'), str) or not item['family'].strip():
            raise ValueError(f'Probe {item["id"]}: family required')
        if 'reference' in item and (not isinstance(item['reference'], str) or not item['reference'].strip()):
            raise ValueError(f'Probe {item["id"]}: reference must be nonempty text')
        probes.append({k: item[k] for k in ('id', 'family', 'prompt', 'reference') if k in item})
    if not probes:
        raise ValueError('At least one probe is required')
    return probes


def verified_candidate(repo_id: str, revision: str | None, token: str):
    """Pin a private Hub revision and verify bytes before restricted torch loading."""
    from huggingface_hub import HfApi, hf_hub_download
    from NanEcho.hf_checkpoint_bridge import MANIFEST_PATH, METADATA_PATH, MODEL, PATH, sha256

    info = HfApi(token=token).model_info(repo_id, revision=revision)
    if not info.private:
        raise ValueError('Candidate repository must be private')
    revision = info.sha
    manifest_path = Path(hf_hub_download(repo_id, MANIFEST_PATH, revision=revision, token=token))
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('status') != 'candidate_not_promoted' or
            manifest.get('format') != 'nanecho-dte-native-candidate-v1' or
            manifest.get('source_repo') != '9cog/echoself' or int(manifest.get('iteration', 0)) < 1):
        raise ValueError('Unexpected candidate status/lineage or untrained checkpoint')
    weights = Path(hf_hub_download(repo_id, PATH, revision=revision, token=token))
    if sha256(weights) != manifest.get('sha256'):
        raise ValueError('Hub checkpoint SHA-256 does not match pinned manifest')
    meta = json.loads(Path(hf_hub_download(repo_id, METADATA_PATH, revision=revision, token=token)).read_text())
    if (meta.get('checkpoint_id') != manifest.get('checkpoint_id') or
            int(meta.get('iteration', 0)) != manifest['iteration'] or
            any(meta.get('model_config', {}).get(key) != expected for key, expected in MODEL.items())):
        raise ValueError('Candidate cache metadata/architecture disagree with manifest')
    # Restricted load: never run arbitrary Hub pickle even if a checksum matches.
    import torch
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    config = checkpoint.get('model_config') or checkpoint.get('config') or {}
    tokenizer = checkpoint.get('tokenizer') or {}
    for key, expected in MODEL.items():
        if config.get(key) != expected:
            raise ValueError(f'Native DTE architecture mismatch: {key}')
    embedding = checkpoint.get('model_state_dict', {}).get('token_embedding.weight')
    if embedding is None or tuple(embedding.shape) != (8192, 256):
        raise ValueError('Unexpected native token embedding')
    if (checkpoint.get('checkpoint_id') != manifest['checkpoint_id'] or
            int(checkpoint.get('iteration', 0)) != manifest['iteration'] or
            tokenizer.get('name') != 'dte_bpe' or
            tokenizer.get('tokenizer_sha256') != manifest.get('tokenizer_sha256') or
            meta.get('data_config') != checkpoint.get('data_config')):
        raise ValueError('Checkpoint tokenizer/dataset metadata or iteration disagree with manifest')
    manifest['verified_objective_id'] = checkpoint['data_config'].get('objective_id', 'legacy-double-shift-invalid-clm')
    return weights, manifest, revision


def teacher_forced_nll(runtime, prefix: str, suffix: str | None = None, *, seed: int = 42, stride: int = 256) -> tuple[float, int]:
    """Score each target token once, with prior context, under the native model.

    For suffix scoring tokenizes prefix and suffix separately (a clear conditional
    token-sequence convention); no target appears in its own input logit. Position
    zero has no preceding context, so prefix-only scoring starts at token one.
    """
    import torch
    import torch.nn.functional as F

    before = runtime.encode(prefix)
    target = runtime.encode(suffix) if suffix is not None else []
    ids = before + target
    first = len(before) if suffix is not None else 1
    if not before or not ids or first >= len(ids):
        raise ValueError('Need at least one context token and one predicted token')
    if stride < 1 or stride >= runtime.config.block_size:
        raise ValueError('stride must lie between 1 and block_size - 1')
    loss_sum, count = 0.0, 0
    # Restore RNG state: stochastic hypergraph injection, if enabled in the
    # model config, must be reproducible and paired across checkpoint comparisons.
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        for target_start in range(first, len(ids), stride):
            target_end = min(target_start + stride, len(ids))
            context_start = max(0, target_end - runtime.config.block_size)
            if context_start >= target_start:
                raise ValueError('Scoring window lost preceding context')
            x = torch.tensor([ids[context_start:target_end]], device=runtime.device)
            out = runtime.model(x)['logits'][0, :-1].float()
            labels = x[0, 1:]
            offset = target_start - context_start - 1
            part = F.cross_entropy(out[offset:], labels[offset:], reduction='sum').item()
            predicted = target_end - target_start
            if not math.isfinite(part):
                raise ValueError('Non-finite token loss')
            loss_sum += part
            count += predicted
    return loss_sum / count, count


def diagnostics(text: str) -> dict:
    words = re.findall(r"\b\w+\b", text.casefold())
    pairs = list(zip(words, words[1:]))
    return {'words': len(words), 'distinct_2': len(set(pairs)) / len(pairs) if pairs else None,
            'empty': not text.strip(), 'replacement_characters': text.count('\ufffd')}


def aggregate_diagnostics(results: list[dict]) -> dict:
    distinct = len({item['generation_sha256'] for item in results})
    zero_words = sum(item['diagnostics']['words'] == 0 for item in results)
    return {'probe_count': len(results), 'distinct_generation_count': distinct,
            'zero_word_generations': zero_words,
            'collapse_warning': distinct == 1 or zero_words == len(results),
            'human_identity_verdict': None}


def private_write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write('\n')


def evaluate(runtime, probes: list[dict], *, max_new_tokens: int, seed: int,
             temperature: float, top_k: int, top_p: float, do_sample: bool) -> list[dict]:
    results = []
    for row in probes:
        prompt = row['prompt']
        prompt_nll = None
        try:
            prompt_nll, prompt_count = teacher_forced_nll(runtime, prompt, seed=seed)
        except ValueError as exc:
            if 'Need at least one context token' not in str(exc):
                raise
            prompt_count = 0
        response = runtime.generate(prompt, max_new_tokens=max_new_tokens, seed=seed,
                                    temperature=temperature, top_k=top_k, top_p=top_p,
                                    do_sample=do_sample)
        result = {'id': row['id'], 'family': row['family'], 'prompt': prompt,
                  'generation': response, 'prompt_tokens_scored': prompt_count,
                  'prompt_nll': prompt_nll, 'prompt_perplexity': math.exp(min(80, prompt_nll)) if prompt_nll is not None else None,
                  'diagnostics': diagnostics(response), 'generation_sha256': hashlib.sha256(response.encode()).hexdigest(),
                  'human_scores': None}
        if 'reference' in row:
            nll, count = teacher_forced_nll(runtime, prompt, row['reference'], seed=seed)
            result.update(reference=row['reference'], reference_tokens_scored=count,
                          reference_conditional_nll=nll, reference_conditional_perplexity=math.exp(min(80, nll)))
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', required=True, type=Path, help='Private JSONL, one original probe per line')
    parser.add_argument('--output', required=True, type=Path, help='Private detailed generations and references (new file only)')
    parser.add_argument('--evidence-output', type=Path, help='Optional non-text summary for dtechorg echo-eval (new file only)')
    parser.add_argument('--repo-id', default='drzo/echoself-dte')
    parser.add_argument('--revision', help='Optional pinned Hub revision; by default pin current HEAD at download time')
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--greedy', action='store_true', help='Deterministic decoding for comparison; still a real model response')
    args = parser.parse_args()
    if args.output.resolve() == args.prompts.resolve() or (args.evidence_output and
        args.evidence_output.resolve() in (args.prompts.resolve(), args.output.resolve())):
        parser.error('Input, private report and evidence report must be distinct files')
    if args.max_new_tokens < 1 or args.max_new_tokens > 256:
        parser.error('max-new-tokens must be 1..256')
    if args.temperature <= 0 or not 0 <= args.top_p <= 1 or args.top_k < 0:
        parser.error('invalid sampling parameters')
    token = os.getenv('HF_TOKEN') or os.getenv('HFMAN')
    if not token:
        parser.error('Set HF_TOKEN for the private native checkpoint')
    probes = load_probes(args.prompts)
    weights, manifest, revision = verified_candidate(args.repo_id, args.revision, token)
    from NanEcho.runtime import NanEchoRuntime
    runtime = NanEchoRuntime.load(weights, device=args.device)
    if runtime.metadata['iteration'] != manifest['iteration'] or runtime.tokenizer.name != 'dte_bpe':
        raise ValueError('Wrong runtime checkpoint or tokenizer')
    scored = evaluate(runtime, probes, max_new_tokens=args.max_new_tokens, seed=args.seed,
                      temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                      do_sample=not args.greedy)
    provenance = {'schema': 'dte-native-identity-candidate-eval-v1', 'repo_id': args.repo_id,
                  'revision': revision, 'checkpoint_sha256': manifest['sha256'],
                  'iteration': manifest['iteration'], 'tokenizer_sha256': manifest['tokenizer_sha256'],
                  'source_revision': manifest['source_revision'], 'probe_set_sha256': canonical_hash(probes),
                  'seed': args.seed, 'decoding': {'max_new_tokens': args.max_new_tokens,
                  'temperature': args.temperature, 'top_k': args.top_k, 'top_p': args.top_p,
                  'do_sample': not args.greedy}, 'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
                  'status': 'candidate_not_promoted', 'identity_improvement_proven': False,
                  'training_objective_id': manifest['verified_objective_id'],
                  'legacy_double_shift_training': manifest['verified_objective_id'] == 'legacy-double-shift-invalid-clm'}
    summary = aggregate_diagnostics(scored)
    private_write(args.output, {'provenance': provenance, 'results': scored, 'summary': summary})
    if args.evidence_output:
        private_write(args.evidence_output, {'provenance': provenance,
                      'results': [{k: v for k, v in result.items() if k not in ('prompt', 'reference', 'generation')}
                                  for result in scored], 'summary': summary,
                      'note': 'Automatic diagnostics only; no independent identity or human quality verdict.'})
    print(json.dumps({'status': provenance['status'], 'iteration': provenance['iteration'],
                      'probe_count': len(scored), 'private_report': str(args.output),
                      'evidence_report': str(args.evidence_output) if args.evidence_output else None}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
