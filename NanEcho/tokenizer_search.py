"""Persona-gripped tokenizer search (Phase 1).

Trains/scores candidate tokenizers against the persona corpus and selects
the one with the best *grip* on the echoself persona, rather than assuming
GPT-2 (or any fixed architecture) is optimal.

Grip metric (higher is better) combines, per candidate tokenizer:

1. ``persona_coverage`` — mean ``score_persona_text`` coverage over the
   persona corpus after an encode→decode round-trip. A tokenizer that
   fragments persona-bearing tokens loses coverage; one that keeps them
   intact preserves the signal.
2. ``fertility_score`` — inverse of average tokens-per-word on the persona
   corpus (subword fertility). Lower fertility = fewer fragments = tighter
   persona representation. Normalized so 1.0 = one token per word.
3. ``probe_score`` — perplexity of a tiny reservoir-probe ridge readout
   predicting next-token identity from running char statistics, mapped to
   (0, 1]. A cheap stand-in for downstream model perplexity that keeps the
   search fast enough to run in CI.

The winning tokenizer's provenance is written into the dataset
``metadata.json`` so the runtime/training provenance validator passes for
*any* persona-selected tokenizer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from NanEcho.drift import score_persona_text
from NanEcho.spec import (
    TokenizerAdapter,
    TokenizerSpec,
    CharTokenizer,
    GPT2_SPEC,
)


@dataclass
class CandidateScore:
    """Grip breakdown for one candidate tokenizer."""

    name: str
    vocab_size: int
    persona_coverage: float
    fertility: float  # tokens per word (lower is better)
    fertility_score: float
    probe_perplexity: float
    probe_score: float
    grip: float  # weighted combination (higher is better)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Default weights for the grip components.
DEFAULT_GRIP_WEIGHTS = {
    "persona_coverage": 0.5,
    "fertility_score": 0.2,
    "probe_score": 0.3,
}


def _corpus_texts(corpus_dir: Path) -> List[str]:
    """Load persona conversation texts from a directory of markdown/jsonl."""
    texts: List[str] = []
    for path in sorted(corpus_dir.glob("**/*")):
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "messages" in record:
                    for msg in record["messages"]:
                        content = msg.get("content", "")
                        if content:
                            texts.append(content)
                elif "text" in record:
                    texts.append(record["text"])
        elif path.suffix in (".md", ".txt"):
            texts.append(path.read_text(encoding="utf-8"))
    return texts


def persona_coverage(tokenizer: TokenizerAdapter, texts: Sequence[str]) -> float:
    """Mean persona-dimension coverage after an encode/decode round-trip."""
    if not texts:
        return 0.0
    scores: List[float] = []
    for text in texts:
        try:
            round_tripped = tokenizer.decode(tokenizer.encode(text))
        except Exception:
            # A tokenizer that cannot round-trip the corpus gets no grip.
            return 0.0
        dimension_scores = score_persona_text(round_tripped)
        if dimension_scores:
            scores.append(sum(dimension_scores.values()) / len(dimension_scores))
    return float(np.mean(scores)) if scores else 0.0


def fertility(tokenizer: TokenizerAdapter, texts: Sequence[str]) -> float:
    """Average tokens per word on the corpus (subword fertility)."""
    total_tokens = 0
    total_words = 0
    for text in texts:
        words = text.split()
        if not words:
            continue
        total_words += len(words)
        total_tokens += len(tokenizer.encode(text))
    if total_words == 0:
        return float("inf")
    return total_tokens / total_words


def _char_probe_perplexity(
    tokenizer: TokenizerAdapter, texts: Sequence[str], train_fraction: float = 0.8
) -> float:
    """Cheap next-token probe perplexity via ridge readout on char statistics.

    Builds a tiny linear probe from running character histogram features to
    next-token one-hot targets, fits by ridge regression, and returns the
    mean per-token perplexity on a held-out split. This is a stand-in for
    downstream model perplexity that runs in milliseconds.
    """
    # Flatten corpus into token id stream per text.
    streams = [tokenizer.encode(t) for t in texts if t.strip()]
    streams = [s for s in streams if len(s) > 8]
    if not streams:
        return float("inf")

    vocab = tokenizer.vocab_size
    # Feature dim: 256 char-class histogram bins (of previous token) + bias.
    feat_dim = 32

    def features(prev_id: int) -> np.ndarray:
        f = np.zeros(feat_dim)
        f[prev_id % feat_dim] = 1.0
        f[-1] = 1.0  # bias
        return f

    X_rows: List[np.ndarray] = []
    Y_rows: List[int] = []
    for s in streams:
        for i in range(1, len(s)):
            X_rows.append(features(s[i - 1]))
            Y_rows.append(min(s[i], vocab - 1))

    X = np.stack(X_rows)
    Y = np.array(Y_rows)
    n = len(Y)
    split = max(1, int(n * train_fraction))
    Xtr, Xte = X[:split], X[split:]
    Ytr, Yte = Y[:split], Y[split:]
    if len(Yte) == 0:
        Xte, Yte = Xtr, Ytr

    # One-hot targets over a capped label space for tractability.
    label_space = min(vocab, 512)
    Ytr_oh = np.zeros((len(Ytr), label_space))
    Ytr_oh[np.arange(len(Ytr)), Ytr % label_space] = 1.0

    ridge = 1e-3
    XtX = Xtr.T @ Xtr + ridge * np.eye(feat_dim)
    XtY = Xtr.T @ Ytr_oh
    W = np.linalg.solve(XtX, XtY)

    logits = Xte @ W  # (N, label_space)
    logits -= logits.max(axis=1, keepdims=True)
    logZ = np.log(np.exp(logits).sum(axis=1))
    nll = -(logits[np.arange(len(Yte)), Yte % label_space] - logZ)
    return float(np.exp(np.mean(nll)))


def probe_score(perplexity: float) -> float:
    """Map perplexity to (0, 1]; 1.0 = perfect prediction."""
    if not math.isfinite(perplexity) or perplexity <= 0:
        return 0.0
    return 1.0 / perplexity


def score_candidate(
    tokenizer: TokenizerAdapter,
    texts: Sequence[str],
    weights: Optional[Dict[str, float]] = None,
) -> CandidateScore:
    """Compute the full grip breakdown for one candidate."""
    weights = weights or DEFAULT_GRIP_WEIGHTS
    cov = persona_coverage(tokenizer, texts)
    fert = fertility(tokenizer, texts)
    fert_score = 1.0 / fert if math.isfinite(fert) and fert > 0 else 0.0
    ppl = _char_probe_perplexity(tokenizer, texts)
    p_score = probe_score(ppl)
    grip = (
        weights["persona_coverage"] * cov
        + weights["fertility_score"] * fert_score
        + weights["probe_score"] * p_score
    )
    return CandidateScore(
        name=tokenizer.name,
        vocab_size=tokenizer.vocab_size,
        persona_coverage=cov,
        fertility=fert,
        fertility_score=fert_score,
        probe_perplexity=ppl,
        probe_score=p_score,
        grip=grip,
    )


def search(
    corpus_dir: str | Path,
    candidates: Optional[Sequence[Callable[[], TokenizerAdapter]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Score all candidates on the persona corpus and pick the winner.

    Parameters
    ----------
    corpus_dir:
        Directory of persona conversation files (markdown/jsonl).
    candidates:
        Factories producing tokenizer adapters. Defaults to the char
        fallback plus GPT-2 (if tiktoken is importable).
    weights:
        Grip component weights; see ``DEFAULT_GRIP_WEIGHTS``.

    Returns
    -------
    dict with ``winner`` (TokenizerSpec provenance), ``scores`` (all
    candidates), and ``corpus_size``.
    """
    corpus_dir = Path(corpus_dir)
    texts = _corpus_texts(corpus_dir)

    if candidates is None:
        candidates = [CharTokenizer]
        try:
            from NanEcho.runtime import NanEchoTokenizer

            candidates.append(NanEchoTokenizer)
        except Exception:
            pass

    scores: List[CandidateScore] = []
    for factory in candidates:
        try:
            tokenizer = factory()
        except Exception:
            continue
        scores.append(score_candidate(tokenizer, texts, weights))

    if not scores:
        raise RuntimeError("No tokenizer candidates could be evaluated")

    winner = max(scores, key=lambda s: s.grip)
    # Map winner name back to a spec via a fresh adapter instance.
    winner_spec = None
    for factory in candidates:
        try:
            tok = factory()
        except Exception:
            continue
        if tok.name == winner.name:
            winner_spec = TokenizerSpec.from_provenance(tok.provenance())
            break
    if winner_spec is None:
        raise RuntimeError("Winning tokenizer has no provenance")

    return {
        "winner": winner_spec.provenance(),
        "scores": [s.to_dict() for s in sorted(scores, key=lambda s: -s.grip)],
        "corpus_size": len(texts),
    }


def write_provenance_to_metadata(
    metadata_path: str | Path, spec: TokenizerSpec
) -> Dict[str, Any]:
    """Write the winning tokenizer's provenance into dataset metadata.json.

    Extends the schema so the runtime/training provenance validator passes
    for any persona-selected tokenizer, not just GPT-2.
    """
    metadata_path = Path(metadata_path)
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tokenizer"] = spec.provenance()
    metadata["vocab_size"] = spec.vocab_size
    metadata["tokenizer_selected_by"] = "NanEcho/tokenizer_search.py"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":  # pragma: no cover - CLI entry
    import argparse

    parser = argparse.ArgumentParser(description="Persona-gripped tokenizer search")
    parser.add_argument(
        "--corpus",
        default=str(Path(__file__).parent / "persona_corpus"),
        help="Persona corpus directory",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional dataset metadata.json to update with the winner",
    )
    args = parser.parse_args()

    result = search(args.corpus)
    print(json.dumps(result, indent=2))
    if args.metadata:
        spec = TokenizerSpec.from_provenance(result["winner"])
        write_provenance_to_metadata(args.metadata, spec)
        print(f"\n✅ Wrote winning tokenizer provenance to {args.metadata}")
