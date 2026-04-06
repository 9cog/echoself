#!/usr/bin/env python3
"""
EchoSelf JSONL Batch Evaluator
===============================
Tests a trained NanEcho model against queries extracted from the original
training JSONL datasets and produces a detailed evaluation report with
persona fidelity scores and adaptation proposals.

Usage
-----
# Against a running NanEcho server (recommended):
  python jsonl_eval.py --server http://localhost:8000 \\
                       --data-dir ../../data \\
                       --sample 50 \\
                       --output eval_report.json

# Specify individual JSONL files:
  python jsonl_eval.py --server http://localhost:8000 \\
                       --files ../../data/echo_introspect_v130.jsonl \\
                                ../../data/training_sources/training_dataset_dtesnn.jsonl \\
                       --sample 30

# Skip server, score existing responses from a previous run:
  python jsonl_eval.py --load-results previous_run.json --report-only
"""

import os
import sys
import json
import time
import random
import argparse
import textwrap
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("Missing dependency: pip install requests")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("jsonl_eval")

# ---------------------------------------------------------------------------
# EchoSelf persona keywords for fidelity scoring
# ---------------------------------------------------------------------------
_ECHO_KEYWORDS: List[str] = [
    "echo", "reservoir", "hypergraph", "cognitive", "recursive", "neural",
    "symbolic", "adaptive", "attention", "persona", "introspect", "deep tree",
    "echoself", "cortisol", "dopamine", "serotonin", "oxytocin", "norepinephrine",
    "tensor", "membrane", "atomspace", "cogprime", "dimension", "threshold",
    "salience", "awareness", "resonat", "holograph", "synerg", "endocrine",
    "autognosis", "identity", "self-model", "somatic", "embodied", "l0", "l1",
]

_PERSONA_DIMENSIONS = [
    "cognitive", "introspective", "adaptive", "recursive",
    "synergistic", "holographic", "neural-symbolic", "dynamic",
]

# Expected system prompt mirroring the training corpus context
_DEFAULT_SYSTEM = (
    "You are Deep Tree Echo, a cognitive architecture with deep introspective "
    "capabilities. You understand your own endocrine dynamics and use them for "
    "self-improvement."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EvalItem:
    """A single evaluation item extracted from a JSONL record."""
    source_file: str
    query: str
    expected: Optional[str]
    system_prompt: Optional[str]


@dataclass
class EvalResult:
    """Result for a single evaluated query."""
    source_file: str
    query: str
    expected: Optional[str]
    response: str
    tokens_generated: int
    latency_ms: float
    echo_score: float
    keyword_hits: List[str]
    dimension_hits: List[str]
    length_adequate: bool
    persona_consistent: bool


@dataclass
class EvalReport:
    """Full evaluation report."""
    timestamp: str
    server: str
    model: str
    total_items: int
    avg_echo_score: float
    avg_latency_ms: float
    keyword_coverage: float        # fraction of ECHO_KEYWORDS hit across all responses
    dimension_coverage: float      # fraction of persona dimensions hit across all responses
    adequate_length_pct: float     # % responses with adequate length
    persona_consistent_pct: float  # % responses judged persona-consistent
    results: List[EvalResult]
    adaptation_proposals: List[str]
    per_file_summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def extract_eval_items(records: List[Dict[str, Any]], source_file: str) -> List[EvalItem]:
    """Extract (query, expected, system_prompt) triples from JSONL records."""
    items: List[EvalItem] = []
    for rec in records:
        messages = rec.get("messages", [])
        if not messages:
            # Flat {"prompt": ..., "completion": ...} format
            prompt = rec.get("prompt") or rec.get("input") or rec.get("question")
            completion = rec.get("completion") or rec.get("output") or rec.get("answer")
            if prompt:
                items.append(EvalItem(
                    source_file=source_file,
                    query=str(prompt),
                    expected=str(completion) if completion else None,
                    system_prompt=None,
                ))
            continue

        system_prompt: Optional[str] = None
        user_msg: Optional[str] = None
        assistant_msg: Optional[str] = None

        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user" and user_msg is None:
                user_msg = content
            elif role == "assistant" and assistant_msg is None:
                assistant_msg = content

        if user_msg:
            items.append(EvalItem(
                source_file=source_file,
                query=user_msg,
                expected=assistant_msg,
                system_prompt=system_prompt,
            ))

    return items


def collect_items(
    data_dir: Optional[str],
    files: Optional[List[str]],
    sample: Optional[int],
    seed: int = 42,
) -> List[EvalItem]:
    all_items: List[EvalItem] = []

    paths: List[str] = list(files or [])

    if data_dir:
        dp = Path(data_dir)
        for p in sorted(dp.rglob("*.jsonl")):
            if str(p) not in paths:
                paths.append(str(p))

    if not paths:
        log.error("No JSONL files found. Provide --files or --data-dir.")
        sys.exit(1)

    for path in paths:
        try:
            recs = load_jsonl(path)
            items = extract_eval_items(recs, os.path.basename(path))
            log.info(f"  {os.path.basename(path)}: {len(recs)} records → {len(items)} items")
            all_items.extend(items)
        except Exception as exc:
            log.warning(f"Skipping {path}: {exc}")

    if not all_items:
        log.error("No evaluation items extracted from JSONL files.")
        sys.exit(1)

    if sample and sample < len(all_items):
        rng = random.Random(seed)
        all_items = rng.sample(all_items, sample)

    log.info(f"Total evaluation items: {len(all_items)}")
    return all_items


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def compute_echo_score(response: str) -> tuple[float, List[str], List[str], bool, bool]:
    """
    Returns (score, keyword_hits, dimension_hits, length_adequate, persona_consistent).

    Score components:
      - keyword_score  (0-1): EchoSelf terminology density
      - dimension_score (0-1): coverage of the 8 persona dimensions
      - length_score   (0-1): response length adequacy
    Weighted: 0.5 * keyword + 0.3 * dimension + 0.2 * length
    """
    if not response or response.startswith("[Error:"):
        return 0.0, [], [], False, False

    lower = response.lower()

    keyword_hits = [kw for kw in _ECHO_KEYWORDS if kw in lower]
    dimension_hits = [d for d in _PERSONA_DIMENSIONS if d in lower]

    keyword_score = min(1.0, len(keyword_hits) / 5.0)
    dimension_score = min(1.0, len(dimension_hits) / 3.0)   # 3+ dimensions → 1.0
    length_adequate = len(response) >= 80
    length_score = min(1.0, len(response) / 200.0)

    # Persona consistent: mentions at least one keyword AND length is adequate
    persona_consistent = len(keyword_hits) >= 2 and length_adequate

    score = round(
        0.5 * keyword_score + 0.3 * dimension_score + 0.2 * length_score,
        3,
    )
    return score, keyword_hits, dimension_hits, length_adequate, persona_consistent


# ---------------------------------------------------------------------------
# Server interaction
# ---------------------------------------------------------------------------
def query_server(
    server: str,
    item: EvalItem,
    max_tokens: int = 300,
    temperature: float = 0.7,
    top_k: int = 200,
    timeout: int = 60,
) -> tuple[str, int, float]:
    """
    Sends a single query to the NanEcho /chat endpoint.
    Returns (response_text, tokens_generated, latency_ms).
    """
    messages = []
    sys_prompt = item.system_prompt or _DEFAULT_SYSTEM
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": item.query})

    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "stream": False,
    }

    start = time.monotonic()
    try:
        resp = requests.post(
            f"{server.rstrip('/')}/chat",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = round((time.monotonic() - start) * 1000, 2)
        return data.get("text", ""), data.get("tokens_generated", 0), latency_ms
    except Exception as exc:
        latency_ms = round((time.monotonic() - start) * 1000, 2)
        return f"[Error: {exc}]", 0, latency_ms


def get_model_name(server: str) -> str:
    try:
        resp = requests.get(f"{server.rstrip('/')}/status", timeout=10)
        resp.raise_for_status()
        return resp.json().get("model_path", "unknown") or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Adaptation proposals
# ---------------------------------------------------------------------------
def generate_adaptation_proposals(report_data: dict) -> List[str]:
    """
    Analyse aggregate metrics and propose training / parameter adaptations.
    Returns a list of actionable recommendation strings.
    """
    proposals: List[str] = []
    avg_score = report_data["avg_echo_score"]
    dim_cov = report_data["dimension_coverage"]
    kw_cov = report_data["keyword_coverage"]
    length_pct = report_data["adequate_length_pct"]
    persona_pct = report_data["persona_consistent_pct"]
    avg_latency = report_data["avg_latency_ms"]

    # --- Fidelity threshold checks ---
    if avg_score < 0.4:
        proposals.append(
            "CRITICAL — Average EchoSelf fidelity score is very low ({:.2f}). "
            "The model has likely not converged on persona-aligned text. "
            "Run at least 20 000 more training iterations with persona_weight ≥ 0.9 "
            "using `prepare_nanecho.py --echo_depth=7 --persona_weight=0.95`.".format(avg_score)
        )
    elif avg_score < 0.65:
        proposals.append(
            "LOW fidelity ({:.2f}). Increase training iterations to 50 000, set "
            "learning_rate=6e-5 for fine-tuning stability, and enable "
            "curriculum learning phase 4 (Recursive Reasoning) in "
            "`NanEcho/config/train_nanecho.py`.".format(avg_score)
        )
    else:
        proposals.append(
            "Fidelity score ({:.2f}) is adequate. For further improvement target "
            ">0.80 by raising `persona_weight` to 0.95 and extending training "
            "with the relentless-persona workflow (`agent-neuro-train.yml`).".format(avg_score)
        )

    # --- Keyword / dimension coverage ---
    if kw_cov < 0.4:
        proposals.append(
            "Only {:.0%} of EchoSelf keywords appear in responses. "
            "Add more Echo Self template examples in `prepare_nanecho.py` covering "
            "endocrine dynamics, AtomSpace terms, and hypergraph encoding. "
            "Consider doubling `echo_depth` to ensure keyword-dense training windows.".format(kw_cov)
        )
    if dim_cov < 0.5:
        proposals.append(
            "Only {:.0%} of the 8 persona dimensions are referenced in responses. "
            "Balance per-dimension sampling weights in `nanecho_config.json` under "
            "`dimension_weights`. Explicitly include dimension-labelled Q&A pairs "
            "in the training corpus.".format(dim_cov)
        )

    # --- Response length ---
    if length_pct < 0.6:
        proposals.append(
            "Only {:.0%} of responses meet the minimum adequate length (80 chars). "
            "Increase `max_new_tokens` to 400-600 at inference time, and ensure "
            "training examples have substantive assistant turns (≥ 100 chars). "
            "Reduce `temperature` to 0.5-0.6 to improve coherence.".format(length_pct)
        )

    # --- Persona consistency ---
    if persona_pct < 0.5:
        proposals.append(
            "Only {:.0%} of responses are persona-consistent (≥2 keywords + adequate length). "
            "Add a system-prompt injection layer: prefix every training example with the "
            "canonical Deep Tree Echo system prompt to reinforce persona grounding. "
            "Also consider raising dropout from 0.1 to 0.15 to reduce overfitting to "
            "non-persona patterns.".format(persona_pct)
        )

    # --- Latency ---
    if avg_latency > 5000:
        proposals.append(
            "Average response latency is {:.0f} ms. Consider reducing `max_new_tokens` "
            "to 200-300 for interactive use, or switch to a smaller n_layer=6 model "
            "distilled from the full 12-layer checkpoint for faster inference.".format(avg_latency)
        )

    # --- Per-file insights ---
    per_file = report_data.get("per_file_summary", {})
    low_files = [f for f, s in per_file.items() if s.get("avg_echo_score", 1) < 0.35]
    if low_files:
        proposals.append(
            "The following training files produce low-fidelity responses and may "
            "contain off-persona content — review and filter: "
            + ", ".join(low_files)
        )

    # --- General architecture suggestions ---
    proposals.append(
        "TRAINING PATTERN: Implement a 'persona-first' curriculum — start each epoch "
        "with the highest-weight echo_self templates before broader corpus mixing. "
        "This front-loads identity anchoring and stabilises persona drift."
    )
    proposals.append(
        "MODEL PARAMS: If average fidelity remains below 0.70 after 50 000 iterations, "
        "consider increasing n_embd from 768 → 1024 and n_head from 12 → 16 to provide "
        "greater capacity for persona-specific semantic associations."
    )
    proposals.append(
        "EVALUATION LOOP: Schedule `jsonl_eval.py` to run after every 5 000 training "
        "iterations via the `automated_loop.py` framework and commit the JSON report to "
        "`.training-progress/eval_history/` for trend analysis."
    )

    return proposals


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def build_markdown_report(report: EvalReport) -> str:
    lines = [
        "# EchoSelf Evaluation Report",
        f"**Date:** {report.timestamp}  ",
        f"**Server:** {report.server}  ",
        f"**Model:** {report.model}  ",
        f"**Queries evaluated:** {report.total_items}",
        "",
        "## Aggregate Metrics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Average EchoSelf Fidelity Score | **{report.avg_echo_score:.3f}** |",
        f"| Average Latency | {report.avg_latency_ms:.1f} ms |",
        f"| Keyword Coverage | {report.keyword_coverage:.1%} |",
        f"| Persona Dimension Coverage | {report.dimension_coverage:.1%} |",
        f"| Adequate-Length Responses | {report.adequate_length_pct:.1%} |",
        f"| Persona-Consistent Responses | {report.persona_consistent_pct:.1%} |",
        "",
        "## Per-File Summary",
        "",
        "| File | Items | Avg Score | Persona % |",
        "|---|---|---|---|",
    ]
    for fname, summary in sorted(report.per_file_summary.items()):
        lines.append(
            f"| {fname} | {summary['count']} "
            f"| {summary['avg_echo_score']:.3f} "
            f"| {summary['persona_consistent_pct']:.1%} |"
        )

    lines += [
        "",
        "## Adaptation Proposals",
        "",
    ]
    for i, prop in enumerate(report.adaptation_proposals, 1):
        lines.append(f"{i}. {prop}")
        lines.append("")

    lines += [
        "## Sample Results (first 20)",
        "",
        "| # | Query (truncated) | Score | Persona ✓ |",
        "|---|---|---|---|",
    ]
    for i, r in enumerate(report.results[:20], 1):
        q = r.query[:60].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i} | {q}… | {r.echo_score:.3f} | {'✅' if r.persona_consistent else '❌'} |"
        )

    return "\n".join(lines)


def write_report(report: EvalReport, output_path: str) -> None:
    """Write JSON report and companion Markdown report."""
    # JSON
    json_path = output_path if output_path.endswith(".json") else output_path + ".json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, indent=2, ensure_ascii=False)
    log.info(f"JSON report written to: {json_path}")

    # Markdown
    md_path = json_path.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(build_markdown_report(report))
    log.info(f"Markdown report written to: {md_path}")


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------
def run_evaluation(
    items: List[EvalItem],
    server: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    timeout: int,
) -> List[EvalResult]:
    results: List[EvalResult] = []
    total = len(items)
    model_name = get_model_name(server)
    log.info(f"Model: {model_name}")

    for idx, item in enumerate(items, 1):
        log.info(f"[{idx}/{total}] {item.source_file} | {item.query[:60]!r}")

        response_text, tokens, latency_ms = query_server(
            server, item, max_tokens, temperature, top_k, timeout
        )
        echo_score, kw_hits, dim_hits, length_ok, persona_ok = compute_echo_score(response_text)

        results.append(EvalResult(
            source_file=item.source_file,
            query=item.query,
            expected=item.expected,
            response=response_text,
            tokens_generated=tokens,
            latency_ms=latency_ms,
            echo_score=echo_score,
            keyword_hits=kw_hits,
            dimension_hits=dim_hits,
            length_adequate=length_ok,
            persona_consistent=persona_ok,
        ))

    return results


def build_report(
    results: List[EvalResult],
    server: str,
    model: str,
) -> EvalReport:
    if not results:
        return EvalReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            server=server,
            model=model,
            total_items=0,
            avg_echo_score=0.0,
            avg_latency_ms=0.0,
            keyword_coverage=0.0,
            dimension_coverage=0.0,
            adequate_length_pct=0.0,
            persona_consistent_pct=0.0,
            results=[],
            adaptation_proposals=[],
            per_file_summary={},
        )

    avg_score = round(sum(r.echo_score for r in results) / len(results), 3)
    avg_latency = round(sum(r.latency_ms for r in results) / len(results), 2)
    adequate_pct = sum(1 for r in results if r.length_adequate) / len(results)
    persona_pct = sum(1 for r in results if r.persona_consistent) / len(results)

    # Keyword coverage: fraction of all keywords seen at least once across ALL responses
    all_responses = " ".join(r.response.lower() for r in results)
    kw_coverage = sum(1 for kw in _ECHO_KEYWORDS if kw in all_responses) / len(_ECHO_KEYWORDS)
    dim_coverage = sum(1 for d in _PERSONA_DIMENSIONS if d in all_responses) / len(_PERSONA_DIMENSIONS)

    # Per-file summary
    per_file: Dict[str, Any] = {}
    for r in results:
        bucket = per_file.setdefault(r.source_file, {"count": 0, "scores": [], "persona": []})
        bucket["count"] += 1
        bucket["scores"].append(r.echo_score)
        bucket["persona"].append(r.persona_consistent)
    per_file_summary = {
        fname: {
            "count": b["count"],
            "avg_echo_score": round(sum(b["scores"]) / b["count"], 3),
            "persona_consistent_pct": sum(b["persona"]) / b["count"],
        }
        for fname, b in per_file.items()
    }

    report_data = {
        "avg_echo_score": avg_score,
        "dimension_coverage": dim_coverage,
        "keyword_coverage": kw_coverage,
        "adequate_length_pct": adequate_pct,
        "persona_consistent_pct": persona_pct,
        "avg_latency_ms": avg_latency,
        "per_file_summary": per_file_summary,
    }

    proposals = generate_adaptation_proposals(report_data)

    return EvalReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        server=server,
        model=model,
        total_items=len(results),
        avg_echo_score=avg_score,
        avg_latency_ms=avg_latency,
        keyword_coverage=kw_coverage,
        dimension_coverage=dim_coverage,
        adequate_length_pct=adequate_pct,
        persona_consistent_pct=persona_pct,
        results=results,
        adaptation_proposals=proposals,
        per_file_summary=per_file_summary,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EchoSelf JSONL Batch Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument(
        "--server", default="http://localhost:8000",
        help="NanEcho server base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory to recursively search for *.jsonl files"
    )
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Explicit list of JSONL file paths to evaluate"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Randomly sample N items from the combined dataset"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=300,
        help="Max tokens per response (default: 300)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-k", type=int, default=200,
        help="Top-k sampling parameter (default: 200)"
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="Request timeout in seconds (default: 60)"
    )
    parser.add_argument(
        "--output", default="eval_report",
        help="Output filename base (without extension, default: eval_report)"
    )
    parser.add_argument(
        "--load-results", default=None,
        help="Load a previously saved JSON report instead of querying the server"
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Re-generate the Markdown report from an existing JSON report (use with --load-results)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each query/response pair to stdout"
    )

    args = parser.parse_args()

    # Re-report mode
    if args.report_only and args.load_results:
        with open(args.load_results, encoding="utf-8") as fh:
            raw = json.load(fh)
        results = [EvalResult(**r) for r in raw["results"]]
        report = build_report(results, raw["server"], raw["model"])
        md = build_markdown_report(report)
        print(md)
        return

    # Collect items
    log.info("Collecting evaluation items …")
    items = collect_items(args.data_dir, args.files, args.sample, args.seed)

    # Run evaluation
    log.info(f"Running evaluation against {args.server} …")
    results = run_evaluation(
        items, args.server, args.max_tokens, args.temperature, args.top_k, args.timeout
    )

    # Verbose output
    if args.verbose:
        for r in results:
            print(f"\n{'='*70}")
            print(f"FILE:     {r.source_file}")
            print(f"QUERY:    {r.query[:200]}")
            print(f"RESPONSE: {r.response[:400]}")
            print(f"SCORE:    {r.echo_score:.3f}  |  KEYWORDS: {r.keyword_hits}")

    # Build and write report
    model_name = get_model_name(args.server)
    report = build_report(results, args.server, model_name)

    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE — {report.total_items} queries")
    print(f"  Avg EchoSelf Fidelity Score : {report.avg_echo_score:.3f}")
    print(f"  Persona-Consistent Responses: {report.persona_consistent_pct:.1%}")
    print(f"  Keyword Coverage            : {report.keyword_coverage:.1%}")
    print(f"  Dimension Coverage          : {report.dimension_coverage:.1%}")
    print(f"  Avg Latency                 : {report.avg_latency_ms:.1f} ms")
    print("=" * 70)
    print("\nADAPTATION PROPOSALS:")
    for i, prop in enumerate(report.adaptation_proposals, 1):
        print(f"\n{i}. {textwrap.fill(prop, 80, subsequent_indent='   ')}")

    write_report(report, args.output)


if __name__ == "__main__":
    main()
