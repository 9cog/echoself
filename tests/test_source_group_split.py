"""Source-group splits must be what the NanEcho data audit can accept."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_nanecho_data import audit
from nanecho_corpus import assign_source_groups, write_source_group_split

TOKENIZER = {
    "name": "gpt2",
    "vocab_size": 50257,
    "eos_token": "<|endoftext|>",
    "eos_token_id": 50256,
}


def _groups():
    return {
        "docs/persona.md": list(range(1000, 4200)),
        "echoself.md": list(range(8000, 11200)),
        "notes/kernel.md": list(range(20000, 23000)),
    }


def test_groups_are_held_out_whole():
    train, val = assign_source_groups(["notes/kernel.md", "echoself.md", "docs/persona.md"])
    assert set(train).isdisjoint(val)
    assert val == ["notes/kernel.md"]
    assert train == ["docs/persona.md", "echoself.md"]


def test_curated_split_passes_audit_without_invented_overlap(tmp_path):
    train_count, val_count = write_source_group_split(
        tmp_path,
        _groups(),
        tokenizer=TOKENIZER,
        reviewed_by="corpus-steward",
        source_revision="reviewed-source-digest",
        synthetic_samples=0,
    )
    assert train_count > 0 and val_count > 0
    report = audit(tmp_path, block_size=64, model_vocab=50257)
    assert report["passed"], report["errors"]
    assert report["overlapping_64_token_windows_fraction"] <= 0.01
    manifest = json.loads((tmp_path / "source_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["train"]["source_groups"]).isdisjoint(manifest["val"]["source_groups"])
    assert manifest["synthetic_samples"] == 0
    assert manifest["reviewed_by"] == "corpus-steward"


def test_missing_reviewer_does_not_pass_the_audit(tmp_path):
    write_source_group_split(
        tmp_path,
        _groups(),
        tokenizer=TOKENIZER,
        reviewed_by="",
        source_revision="content-digest-only",
        synthetic_samples=0,
    )
    report = audit(tmp_path, block_size=64, model_vocab=50257)
    assert report["passed"] is False
    assert any("reviewed_by" in error for error in report["errors"])


def test_synthetic_count_is_not_hidden_from_the_audit(tmp_path):
    write_source_group_split(
        tmp_path,
        _groups(),
        tokenizer=TOKENIZER,
        reviewed_by="corpus-steward",
        source_revision="reviewed-source-digest",
        synthetic_samples=4,
    )
    report = audit(tmp_path, block_size=64, model_vocab=50257)
    assert report["passed"] is False
    assert any("synthetic" in error for error in report["errors"])
