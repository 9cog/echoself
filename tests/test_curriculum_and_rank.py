"""Curriculum progress follows the cumulative budget, and checkpoints rank by loss."""

import json
from pathlib import Path

import pytest

from checkpoint_rank import rank_checkpoint_ids, select_best_checkpoint_id
from nanecho_schedule import learning_rate_at, phase_at, schedule_at

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / ".training-progress" / "nanecho-cached-ci" / "cache" / "metadata.json"


def test_resume_at_4500_is_early_curriculum_not_mastery():
    """A 500-iter CI session must not treat cumulative step 4500 as the last phase."""
    before = phase_at(4500, max_iters=500, cumulative_iters=0)
    after = phase_at(4500, max_iters=500, cumulative_iters=200_000)
    assert before == "adaptive_mastery"
    assert after == "basic_awareness"


def test_warmup_is_a_fraction_of_the_cumulative_budget():
    """Iteration 4500 is still inside a fixed 5000-step warmup, and past a 2% horizon."""
    learning_rate = 2e-4
    before = learning_rate_at(
        4500,
        learning_rate=learning_rate,
        min_lr=1e-5,
        max_iters=500,
        cumulative_iters=0,
        warmup_iters=5000,
        lr_decay_iters=50_000,
        warmup_fraction=0.0,
    )
    after = schedule_at(
        4500,
        learning_rate=learning_rate,
        min_lr=1e-5,
        max_iters=500,
        cumulative_iters=200_000,
        warmup_iters=5000,
        lr_decay_iters=50_000,
        warmup_fraction=0.02,
    )
    assert before["phase"] == "adaptive_mastery"
    assert before["in_warmup"] is True
    assert after["phase"] == "basic_awareness"
    assert after["in_warmup"] is False
    assert after["warmup_iters"] == 4000
    assert after["learning_rate"] > before["learning_rate"]


def test_cached_ci_best_id_and_cleanup_order():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    legacy_order = rank_checkpoint_ids(metadata, metric="quality_score")
    loss_order = rank_checkpoint_ids(metadata, metric="val_loss")
    legacy_best = legacy_order[0]
    loss_best = select_best_checkpoint_id(metadata, metric="val_loss")

    assert legacy_best == "ckpt_20261003_202331_4500_46032ab3_9470fbb7"
    assert loss_best == legacy_best
    assert [metadata[item]["iteration"] for item in legacy_order] == [
        4500, 4300, 4000, 3700, 3600, 3500, 3400, 3300,
    ]
    assert [metadata[item]["iteration"] for item in loss_order] == [
        4500, 4300, 3700, 3600, 3500, 4000, 3400, 3300,
    ]
    assert select_best_checkpoint_id(metadata) == loss_best
    with pytest.raises(ValueError, match="quality_score"):
        select_best_checkpoint_id(metadata, metric="quality_score")
