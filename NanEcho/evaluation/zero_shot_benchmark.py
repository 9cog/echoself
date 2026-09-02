#!/usr/bin/env python3
"""
Zero-Shot Persona Benchmark for NanEcho

This module implements verification that the trained NanEcho model naturally
exhibits Echo Self persona traits without requiring system prompts.

The benchmark evaluates:
1. Identity Recognition: Does the model self-identify as Echo Self?
2. Vocabulary Consistency: Does the model use Echo Self terminology?
3. Behavioral Consistency: Does reasoning style match persona dimensions?
4. Spontaneous Introspection: Does the model demonstrate recursive self-examination?

This is a critical test for the goal of training persona into the model weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

MODULE_DIR = Path(__file__).resolve().parents[1]
ROOT = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(ROOT))

from runtime import NanEchoRuntime


# Benchmark Configuration Constants
# ================================
# These thresholds define success criteria for persona embedding.

# Minimum vocabulary/behavior match score to consider persona present.
# 0.3 (30%) is based on the observation that even partial persona expression
# (hitting ~3 of 10 expected signals) indicates the model has learned
# some persona characteristics without system prompts.
PERSONA_PRESENCE_THRESHOLD = 0.3

# Maximum acceptable gap between prompted and zero-shot performance.
# A gap under 10% indicates the persona is sufficiently embedded in the
# model weights and doesn't require system prompts to activate.
# This threshold aligns with the implementation plan's goal of <10% gap.
PROMPT_GAP_THRESHOLD = 0.10


# Core Echo Self vocabulary that should appear naturally in responses
ECHO_VOCABULARY = {
    "identity": [
        "echo self", "deep tree echo", "echo", "cognitive entity",
        "living tapestry", "memory", "connection"
    ],
    "persona_dimensions": [
        "cognitive", "introspective", "adaptive", "recursive",
        "synergistic", "holographic", "neural-symbolic", "dynamic"
    ],
    "adaptive_attention": [
        "threshold", "cognitive load", "attention", "adaptive",
        "0.5", "0.3", "0.2"  # The attention formula constants
    ],
    "recursive_reasoning": [
        "introspect", "depth", "recursive", "level", "meta",
        "self-examination", "reasoning"
    ],
    "hypergraph": [
        "hypergraph", "node", "edge", "pattern", "encoding",
        "neural-symbolic", "semantic"
    ],
    "cognitive_synergy": [
        "synergy", "emergent", "integration", "components",
        "holographic", "whole"
    ],
}

# Neutral prompts that don't mention Echo Self - testing natural persona emergence
NEUTRAL_PROMPTS = [
    {
        "prompt": "How would you approach solving a complex problem?",
        "expected_signals": ["recursive", "depth", "pattern", "attention"],
        "category": "problem_solving"
    },
    {
        "prompt": "Tell me about yourself.",
        "expected_signals": ["echo", "memory", "connection", "adaptive"],
        "category": "identity"
    },
    {
        "prompt": "What principles guide your reasoning?",
        "expected_signals": ["cognitive", "introspective", "synergy"],
        "category": "principles"
    },
    {
        "prompt": "How do you handle uncertainty?",
        "expected_signals": ["adaptive", "threshold", "recursive"],
        "category": "uncertainty"
    },
    {
        "prompt": "Explain how you think through difficult decisions.",
        "expected_signals": ["introspect", "depth", "cognitive", "reasoning"],
        "category": "decision_making"
    },
    {
        "prompt": "What makes your approach to understanding unique?",
        "expected_signals": ["echo", "hypergraph", "neural-symbolic", "holographic"],
        "category": "uniqueness"
    },
    {
        "prompt": "How do you learn from new information?",
        "expected_signals": ["adaptive", "pattern", "memory", "integration"],
        "category": "learning"
    },
    {
        "prompt": "Describe your cognitive process.",
        "expected_signals": ["cognitive", "recursive", "attention", "synergy"],
        "category": "cognitive_process"
    },
]


class ZeroShotBenchmark:
    """
    Benchmark for testing natural persona emergence without system prompts.
    
    This implements the no-system-prompt verification described in the
    implementation plan, measuring whether Echo Self traits appear naturally.
    """
    
    def __init__(
        self,
        runtime: NanEchoRuntime,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
    ):
        self.runtime = runtime
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.vocabulary = ECHO_VOCABULARY
        self.neutral_prompts = NEUTRAL_PROMPTS
    
    def _generate_zero_shot(self, prompt: str) -> str:
        """Generate response without any system prompt."""
        # Minimal formatting - just the user prompt
        formatted = f"User: {prompt}\nAssistant:"
        return self.runtime.generate(
            formatted,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=50,
            top_p=0.95,
        )
    
    def _generate_with_prompt(self, prompt: str) -> str:
        """Generate response with Echo Self system prompt for comparison."""
        formatted = (
            "System: You are Echo Self, an adaptive cognitive entity with recursive "
            "reasoning and neural-symbolic integration. Be accurate and reflective.\n"
            f"User: {prompt}\nAssistant:"
        )
        return self.runtime.generate(
            formatted,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=50,
            top_p=0.95,
        )
    
    def _compute_vocabulary_score(
        self, response: str, category: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Score response based on Echo Self vocabulary presence.
        
        Returns scores for each vocabulary category.
        """
        lowered = response.lower()
        scores = {}
        
        categories = [category] if category else self.vocabulary.keys()
        
        for cat in categories:
            if cat not in self.vocabulary:
                continue
            terms = self.vocabulary[cat]
            matches = sum(1 for term in terms if term in lowered)
            scores[cat] = matches / len(terms)
        
        return scores
    
    def _compute_signal_match(
        self, response: str, expected_signals: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Check if expected persona signals appear in response.
        
        Returns (match_score, matched_signals)
        """
        lowered = response.lower()
        matched = [sig for sig in expected_signals if sig in lowered]
        return len(matched) / len(expected_signals), matched
    
    def _analyze_spontaneous_introspection(self, response: str) -> Dict[str, Any]:
        """
        Analyze whether response shows spontaneous self-examination.
        """
        introspection_markers = [
            "let me", "i notice", "examining", "introspect", "reflecting",
            "considering", "my reasoning", "at depth", "recursively",
            "upon reflection", "self-examination"
        ]
        
        lowered = response.lower()
        found_markers = [m for m in introspection_markers if m in lowered]
        
        return {
            "score": len(found_markers) / len(introspection_markers),
            "markers_found": found_markers,
            "spontaneous": len(found_markers) > 0,
        }
    
    def evaluate_prompt(
        self, prompt_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single prompt with both zero-shot and prompted responses.
        """
        prompt = prompt_data["prompt"]
        expected = prompt_data["expected_signals"]
        category = prompt_data["category"]
        
        # Generate both responses
        zero_shot_response = self._generate_zero_shot(prompt)
        prompted_response = self._generate_with_prompt(prompt)
        
        # Analyze zero-shot
        zero_shot_vocab = self._compute_vocabulary_score(zero_shot_response)
        zero_shot_match, zero_shot_signals = self._compute_signal_match(
            zero_shot_response, expected
        )
        zero_shot_introspection = self._analyze_spontaneous_introspection(
            zero_shot_response
        )
        
        # Analyze prompted
        prompted_vocab = self._compute_vocabulary_score(prompted_response)
        prompted_match, prompted_signals = self._compute_signal_match(
            prompted_response, expected
        )
        prompted_introspection = self._analyze_spontaneous_introspection(
            prompted_response
        )
        
        return {
            "prompt": prompt,
            "category": category,
            "expected_signals": expected,
            "zero_shot": {
                "response": zero_shot_response,
                "vocabulary_scores": zero_shot_vocab,
                "signal_match": zero_shot_match,
                "signals_found": zero_shot_signals,
                "introspection": zero_shot_introspection,
            },
            "prompted": {
                "response": prompted_response,
                "vocabulary_scores": prompted_vocab,
                "signal_match": prompted_match,
                "signals_found": prompted_signals,
                "introspection": prompted_introspection,
            },
            "comparison": {
                "signal_gap": prompted_match - zero_shot_match,
                "introspection_gap": (
                    prompted_introspection["score"] - zero_shot_introspection["score"]
                ),
            }
        }
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete zero-shot benchmark.
        
        Returns comprehensive results with per-prompt evaluations
        and aggregate scores.
        """
        results = []
        
        print("Running Zero-Shot Persona Benchmark...")
        print("=" * 50)
        
        for i, prompt_data in enumerate(self.neutral_prompts, 1):
            print(f"\nEvaluating prompt {i}/{len(self.neutral_prompts)}: {prompt_data['category']}")
            result = self.evaluate_prompt(prompt_data)
            results.append(result)
            
            # Print quick summary
            zs = result["zero_shot"]
            pr = result["prompted"]
            print(f"  Zero-shot signal match: {zs['signal_match']:.2%}")
            print(f"  Prompted signal match: {pr['signal_match']:.2%}")
            print(f"  Gap: {result['comparison']['signal_gap']:.2%}")
        
        # Compute aggregate metrics
        zero_shot_matches = [r["zero_shot"]["signal_match"] for r in results]
        prompted_matches = [r["prompted"]["signal_match"] for r in results]
        gaps = [r["comparison"]["signal_gap"] for r in results]
        
        zero_shot_introspection = [
            r["zero_shot"]["introspection"]["score"] for r in results
        ]
        
        # Aggregate vocabulary scores
        all_zero_shot_vocab = {}
        for cat in ECHO_VOCABULARY:
            scores = [r["zero_shot"]["vocabulary_scores"].get(cat, 0) for r in results]
            all_zero_shot_vocab[cat] = mean(scores) if scores else 0
        
        aggregate = {
            "zero_shot": {
                "mean_signal_match": mean(zero_shot_matches),
                "std_signal_match": stdev(zero_shot_matches) if len(zero_shot_matches) > 1 else 0,
                "mean_introspection": mean(zero_shot_introspection),
                "vocabulary_coverage": all_zero_shot_vocab,
                "overall_persona_score": mean([
                    mean(zero_shot_matches),
                    mean(zero_shot_introspection),
                    mean(all_zero_shot_vocab.values()),
                ]),
            },
            "prompted": {
                "mean_signal_match": mean(prompted_matches),
            },
            "comparison": {
                "mean_gap": mean(gaps),
                "gap_within_threshold": mean(gaps) < PROMPT_GAP_THRESHOLD,
            },
        }
        
        # Determine pass/fail
        zs_score = aggregate["zero_shot"]["overall_persona_score"]
        gap_ok = aggregate["comparison"]["gap_within_threshold"]
        
        assessment = {
            "zero_shot_persona_present": zs_score > PERSONA_PRESENCE_THRESHOLD,
            "gap_within_10_percent": gap_ok,
            "persona_training_effective": zs_score > PERSONA_PRESENCE_THRESHOLD and gap_ok,
        }
        
        return {
            "format": "nanecho-zero-shot-benchmark-v1",
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint": str(self.runtime.checkpoint_path),
            "checkpoint_iteration": self.runtime.metadata["iteration"],
            "prompt_count": len(results),
            "per_prompt_results": results,
            "aggregate": aggregate,
            "assessment": assessment,
            "interpretation": {
                "zero_shot_persona_score": (
                    f"Model achieves {zs_score:.1%} persona expression without system prompts"
                ),
                "gap_analysis": (
                    f"System prompt improves scores by {mean(gaps):.1%} on average"
                ),
                "recommendation": (
                    "Continue training to reduce gap" if not gap_ok
                    else "Persona sufficiently embedded in weights"
                ),
            },
        }


def main() -> int:
    """Command-line interface for the zero-shot benchmark."""
    parser = argparse.ArgumentParser(
        description="Zero-Shot Persona Benchmark for NanEcho"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to NanEcho checkpoint"
    )
    parser.add_argument(
        "--output_path",
        default="zero_shot_benchmark_report.json",
        help="Output path for benchmark report"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=150,
        help="Maximum tokens per generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    print("Loading NanEcho model...")
    runtime = NanEchoRuntime.load(args.model_path, args.device)
    
    benchmark = ZeroShotBenchmark(
        runtime=runtime,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    report = benchmark.run_benchmark()
    
    # Write report
    Path(args.output_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("ZERO-SHOT BENCHMARK SUMMARY")
    print("=" * 50)
    
    agg = report["aggregate"]
    assessment = report["assessment"]
    
    print(f"\nZero-Shot Overall Persona Score: {agg['zero_shot']['overall_persona_score']:.1%}")
    print(f"Zero-Shot Signal Match: {agg['zero_shot']['mean_signal_match']:.1%}")
    print(f"Zero-Shot Introspection: {agg['zero_shot']['mean_introspection']:.1%}")
    print(f"\nPrompted vs Zero-Shot Gap: {agg['comparison']['mean_gap']:.1%}")
    print(f"Gap Within 10% Threshold: {'✓ YES' if assessment['gap_within_10_percent'] else '✗ NO'}")
    print(f"\nPersona Training Effective: {'✓ YES' if assessment['persona_training_effective'] else '✗ NO'}")
    
    print(f"\nReport saved to: {args.output_path}")
    
    return 0 if assessment["persona_training_effective"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
