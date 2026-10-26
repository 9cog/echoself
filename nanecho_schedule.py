"""Curriculum phase and learning-rate schedule for NanEcho.

Session ``max_iters`` is how many new steps one process will run. Phase and
warmup follow ``cumulative_iters`` when that horizon is longer, so a 500-step
CI slice cannot declare mastery at cumulative step 4500.
"""

from __future__ import annotations

import math
from typing import Any, Dict

PHASES = (
    ("basic_awareness", 0.0, 0.2, 1.2),
    ("persona_dimensions", 0.15, 0.5, 1.0),
    ("hypergraph_patterns", 0.4, 0.7, 0.9),
    ("recursive_reasoning", 0.6, 0.85, 0.8),
    ("adaptive_mastery", 0.8, 1.0, 0.7),
)


def curriculum_budget(max_iters: int, cumulative_iters: int = 0) -> int:
    session = max(int(max_iters), 1)
    cumulative = int(cumulative_iters or 0)
    if cumulative > session:
        return cumulative
    return session


def phase_progress(iteration: int, max_iters: int, cumulative_iters: int = 0) -> float:
    budget = curriculum_budget(max_iters, cumulative_iters)
    return min(max(int(iteration), 0) / budget, 1.0)


def phase_at(iteration: int, max_iters: int, cumulative_iters: int = 0) -> str:
    progress = phase_progress(iteration, max_iters, cumulative_iters)
    for name, start, end, _multiplier in PHASES:
        if start <= progress <= end:
            return name
    return "adaptive_mastery"


def phase_lr_multiplier(phase_name: str) -> float:
    for name, _start, _end, multiplier in PHASES:
        if name == phase_name:
            return multiplier
    return 1.0


def effective_warmup_iters(
    max_iters: int,
    cumulative_iters: int = 0,
    warmup_iters: int = 5000,
    warmup_fraction: float = 0.0,
) -> int:
    """Absolute warmup when no fraction is set; otherwise a share of the horizon."""
    if warmup_fraction and warmup_fraction > 0:
        budget = curriculum_budget(max_iters, cumulative_iters)
        return max(1, int(round(budget * float(warmup_fraction))))
    return max(1, int(warmup_iters))


def learning_rate_at(
    iteration: int,
    *,
    learning_rate: float,
    min_lr: float,
    max_iters: int,
    cumulative_iters: int = 0,
    warmup_iters: int = 5000,
    lr_decay_iters: int = 50000,
    warmup_fraction: float = 0.0,
    enable_curriculum: bool = True,
) -> Dict[str, Any]:
    warmup = effective_warmup_iters(
        max_iters, cumulative_iters, warmup_iters, warmup_fraction
    )
    phase = phase_at(iteration, max_iters, cumulative_iters)
    step = max(int(iteration), 0)
    if step < warmup:
        lr = learning_rate * step / warmup
        in_warmup = True
    elif step < lr_decay_iters and lr_decay_iters > warmup:
        decay_ratio = (step - warmup) / (lr_decay_iters - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = min_lr + coeff * (learning_rate - min_lr)
        in_warmup = False
    else:
        lr = min_lr
        in_warmup = False
    if enable_curriculum:
        lr *= phase_lr_multiplier(phase)
    return {
        "phase": phase,
        "learning_rate": lr,
        "in_warmup": in_warmup,
        "warmup_iters": warmup,
        "budget": curriculum_budget(max_iters, cumulative_iters),
    }


def schedule_at(
    iteration: int,
    **kwargs: Any,
) -> Dict[str, Any]:
    return learning_rate_at(iteration, **kwargs)
