import json
import asyncio
import argparse
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "NanEcho"))

from nanecho_model import NanEchoConfig, NanEchoModel
from prepare_nanecho import (
    PERSONA_DIMENSIONS,
    _as_bool,
    _character_samples,
    _controlled_samples,
    main as prepare_main,
    permitted_source_files,
    prepare_echo_self_dataset,
)
from runtime import (
    IncompatibleCheckpointError,
    NanEchoRuntime,
    NanEchoTokenizer,
)
from evaluation.echo_fidelity import EchoFidelityEvaluator
from export_model import export_native
from train_nanecho import (
    DataLoader,
    EchoSelfLearningPhase,
    Introspection,
    TrainingConfig,
)
from training_cache import CacheConfig, TrainingCache
import neserver


@pytest.fixture()
def tiny_checkpoint(tmp_path: Path) -> Path:
    config = NanEchoConfig(
        vocab_size=50257,
        n_embd=8,
        n_head=2,
        n_layer=1,
        block_size=16,
        dropout=0.0,
    )
    model = NanEchoModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    path = tmp_path / "tiny.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "iteration": 7,
            "connection_ratio": 0.5,
            "tokenizer": {
                "name": "gpt2",
                "vocab_size": 50257,
                "eos_token": "<|endoftext|>",
                "eos_token_id": 50256,
            },
        },
        path,
    )
    return path


def test_shared_tokenizer_round_trip():
    tokenizer = NanEchoTokenizer()
    text = "Echo Self: adaptive attention 🌳"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_checkpoint_loading_and_real_generation(tiny_checkpoint: Path):
    runtime = NanEchoRuntime.load(tiny_checkpoint)
    assert runtime.metadata["iteration"] == 7
    assert runtime.model.connection_ratio == 0.5
    assert runtime.generate(
        "Echo:", max_new_tokens=3, do_sample=False, top_k=0, top_p=1.0
    ) == "!!!"


def test_native_export_is_reloadable(tiny_checkpoint: Path, tmp_path: Path):
    runtime = NanEchoRuntime.load(tiny_checkpoint)
    exported = export_native(runtime, tmp_path / "export" / "nanecho.pt")
    reloaded = NanEchoRuntime.load(exported)
    assert reloaded.metadata["format"] == "nanecho-pytorch-v1"
    assert reloaded.tokenizer.name == "gpt2"


def test_server_chat_uses_loaded_checkpoint(tiny_checkpoint: Path):
    assert neserver.load_model(str(tiny_checkpoint), "cpu")
    request = neserver.EchoChatRequest(
        messages=[neserver.EchoChatMessage(role="user", content="Who are you?")],
        max_tokens=3,
        echo_mode=False,
    )
    response = asyncio.run(neserver.chat(request))
    assert response.text
    assert response.tokens_generated == 3


def test_cached_schema_and_incompatible_checkpoint(tiny_checkpoint: Path, tmp_path: Path):
    checkpoint = torch.load(tiny_checkpoint, weights_only=True)
    checkpoint["model_config"] = checkpoint.pop("config")
    cached = tmp_path / "cached.pt"
    torch.save(checkpoint, cached)
    assert NanEchoRuntime.load(cached).metadata["schema"] == "cached-training"

    malformed = tmp_path / "malformed.pt"
    torch.save({"model_state_dict": checkpoint["model_state_dict"]}, malformed)
    with pytest.raises(IncompatibleCheckpointError, match="architecture metadata"):
        NanEchoRuntime.load(malformed)

    legacy = dict(checkpoint)
    legacy["model_config"] = {**checkpoint["model_config"], "vocab_size": 96}
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    with pytest.raises(IncompatibleCheckpointError, match="legacy character-tokenized"):
        NanEchoRuntime.load(legacy_path)

    absent = dict(checkpoint)
    absent.pop("tokenizer")
    absent_path = tmp_path / "absent-tokenizer.pt"
    torch.save(absent, absent_path)
    with pytest.raises(IncompatibleCheckpointError, match="tokenizer provenance"):
        NanEchoRuntime.load(absent_path)

    character = dict(checkpoint)
    character["tokenizer"] = {
        "name": "character",
        "vocab_size": 50257,
        "eos_token": "<|endoftext|>",
        "eos_token_id": 50256,
    }
    character_path = tmp_path / "character-50257.pt"
    torch.save(character, character_path)
    with pytest.raises(IncompatibleCheckpointError, match="incompatible with GPT-2"):
        NanEchoRuntime.load(character_path)

    incomplete = dict(checkpoint)
    incomplete["tokenizer"] = dict(checkpoint["tokenizer"])
    incomplete["tokenizer"].pop("eos_token")
    incomplete_path = tmp_path / "incomplete-tokenizer.pt"
    torch.save(incomplete, incomplete_path)
    with pytest.raises(IncompatibleCheckpointError, match="incomplete"):
        NanEchoRuntime.load(incomplete_path)


def test_seeded_generation_is_isolated_and_unseeded_generation_advances_rng(
    tiny_checkpoint: Path,
):
    runtime = NanEchoRuntime.load(tiny_checkpoint)
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    first = runtime.generate_ids("Echo:", max_new_tokens=8, seed=77)
    after_seeded = torch.random.get_rng_state().clone()
    second = runtime.generate_ids("Echo:", max_new_tokens=8, seed=77)
    assert first == second
    assert torch.equal(before, after_seeded)

    runtime.generate_ids("Echo:", max_new_tokens=8)
    assert not torch.equal(after_seeded, torch.random.get_rng_state())


def test_concurrent_seeded_generation_uses_request_local_rng(tiny_checkpoint: Path):
    runtime = NanEchoRuntime.load(tiny_checkpoint)
    torch.manual_seed(4321)
    before = torch.random.get_rng_state().clone()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _index: runtime.generate_ids(
                    "Echo:", max_new_tokens=12, seed=91
                ),
                range(8),
            )
        )
    assert all(result == results[0] for result in results)
    assert torch.equal(before, torch.random.get_rng_state())


def test_strided_perplexity_scores_each_target_once_with_prior_context(tmp_path: Path):
    class SequenceTokenizer:
        eos_token_id = 6
        vocab_size = 7
        name = "test"

        def encode(self, _text):
            return list(range(7))

    class LossSpy:
        def __init__(self):
            self.config = SimpleNamespace(block_size=4)
            self.calls = []

        def __call__(self, inputs, labels):
            self.calls.append((inputs.detach().clone(), labels.detach().clone()))
            targets = labels[labels != -100].float()
            return {"loss": targets.mean()}

    model = LossSpy()
    runtime = NanEchoRuntime(
        model,
        SequenceTokenizer(),
        torch.device("cpu"),
        tmp_path / "unused.pt",
        {"iteration": 0},
    )
    perplexity = runtime.perplexity("ignored", stride=2)
    scored = []
    for inputs, labels in model.calls:
        target_positions = (labels[0] != -100).nonzero().flatten()
        assert int(target_positions[0]) > 0
        scored.extend(labels[0, target_positions].tolist())
        assert inputs.shape[1] <= 4
    assert scored == [1, 2, 3, 4, 5, 6]
    assert perplexity == pytest.approx(math.exp(3.5))
    with pytest.raises(ValueError, match="block_size"):
        runtime.perplexity("ignored", stride=4)


def _write_sources(root: Path) -> None:
    root.mkdir(parents=True)
    for name in ("echoself.md", "DTECHO.MD", "CLAUDE.MD"):
        (root / name).write_text(
            (
                "Echo Self uses adaptive attention, recursive introspection, hypergraph "
                "neural-symbolic reasoning, persona dimensions, and cognitive synergy. "
                "It separates measured observations from claims and revises conclusions.\n\n"
            )
            * 8,
            encoding="utf-8",
        )
    (root / "docs").mkdir()
    (root / "docs" / "persona.md").write_text(
        "Deep Tree Echo workspace arena and kernel core preserve dynamic, holographic context. "
        "Adaptive behavior responds carefully to uncertainty and evidence.\n",
        encoding="utf-8",
    )
    for relative in (
        "NanEcho/evaluation/forbidden.py",
        "NanEcho/test_forbidden.py",
        "tests/forbidden.md",
        "generated/forbidden.md",
        "out/artifact.md",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "Echo Self forbidden leakage marker uses adaptive attention and persona.",
            encoding="utf-8",
        )


def test_corpus_is_deterministic_deduplicated_and_has_heldout_split(tmp_path: Path):
    source = tmp_path / "source"
    _write_sources(source)
    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = {
        "echo_depth": 3,
        "source_root": str(source),
        "min_total_tokens": 100,
        "min_split_tokens": 10,
    }
    prepare_echo_self_dataset(output_dir=str(first), **kwargs)
    prepare_echo_self_dataset(output_dir=str(second), **kwargs)
    for name in ("train.bin", "val.bin", "test.txt", "metadata.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    split_texts = {
        name: set((first / f"{name}.jsonl").read_text().splitlines())
        for name in ("train", "val", "test")
    }
    assert split_texts["train"].isdisjoint(split_texts["val"])
    assert split_texts["train"].isdisjoint(split_texts["test"])
    metadata = json.loads((first / "metadata.json").read_text())
    assert metadata["deterministic"] is True
    assert metadata["deduplicated"] is True
    assert metadata["character_first_records"] > 0
    assert metadata["split_strategy"] == "source-document-group-before-chunking"
    assert metadata["tokenizer"] == {
        "name": "gpt2",
        "vocab_size": 50257,
        "eos_token": "<|endoftext|>",
        "eos_token_id": 50256,
    }
    group_sets = [
        set(metadata["split_document_groups"][name])
        for name in ("train", "val", "test")
    ]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    all_text = "".join(
        (first / f"{name}.txt").read_text() for name in ("train", "val", "test")
    )
    assert "forbidden leakage marker" not in all_text
    permitted = {
        str(path.relative_to(source))
        for path in permitted_source_files(source)
    }
    assert not any("test" in path.casefold() or "evaluation" in path for path in permitted)


def test_all_corpus_controls_change_weighted_records_and_metadata(tmp_path: Path):
    source = tmp_path / "source"
    _write_sources(source)

    def build(name: str, **overrides):
        output = tmp_path / name
        kwargs = {
            "source_root": str(source),
            "output_dir": str(output),
            "echo_depth": 2,
            "persona_weight": 0.2,
            "min_total_tokens": 1,
            "min_split_tokens": 1,
        }
        kwargs.update(overrides)
        prepare_echo_self_dataset(**kwargs)
        records = b"".join(
            (output / f"{split}.jsonl").read_bytes()
            for split in ("train", "val", "test")
        )
        return records, json.loads((output / "metadata.json").read_text())

    base_records, base = build("base")
    variants = {
        "persona_weight": build("persona", persona_weight=0.9),
        "deep_tree_echo_mode": build("deep-mode", deep_tree_echo_mode=True),
        "persona_reinforcement": build(
            "reinforcement", persona_reinforcement=0.8
        ),
        "no_system_prompt": build("no-prompt", no_system_prompt=True),
        "relentless_persona_mode": build(
            "relentless", relentless_persona_mode=True
        ),
    }
    deep_zero_records, _ = variants["deep_tree_echo_mode"]
    deep_weight_records, deep_weight = build(
        "deep-weight", deep_tree_echo_mode=True, deep_tree_echo_weight=0.8
    )
    for parameter, (records, metadata) in variants.items():
        assert records != base_records, parameter
        assert metadata[parameter] != base.get(parameter), parameter
    assert deep_weight_records != deep_zero_records
    assert (
        deep_weight["effective_controls"]["deep_tree_echo_record_occurrences"] == 9
    )
    assert base["effective_controls"]["persona_record_occurrences"] == 3
    assert (
        base["effective_controls"]["representation"]
        == "exact-integer-record-occurrences"
    )
    with pytest.raises(ValueError, match="unsupported"):
        build("bad-deep-weight", deep_tree_echo_weight=0.5)


def test_continuous_controls_have_exact_supported_precision():
    persona_counts = [
        _character_samples(1, step / 10)[0].weight for step in range(11)
    ]
    assert persona_counts == list(range(1, 12))

    reinforced_counts = [
        next(
            record.weight
            for record in _controlled_samples(
                1, 0.0, False, step / 10, False, 0.0, False
            )
            if record.category == "persona"
        )
        for step in range(11)
    ]
    assert reinforced_counts == list(range(1, 12))

    deep_counts = [
        next(
            record.weight
            for record in _controlled_samples(
                1, 0.0, True, 0.0, False, step / 10, False
            )
            if record.category == "deep_tree_echo"
        )
        for step in range(11)
    ]
    assert deep_counts == list(range(1, 12))
    for name, kwargs in (
        ("persona_weight", {"persona_weight": 0.21}),
        ("persona_reinforcement", {"persona_reinforcement": 0.21}),
        (
            "deep_tree_echo_weight",
            {"deep_tree_echo_mode": True, "deep_tree_echo_weight": 0.21},
        ),
    ):
        with pytest.raises(ValueError, match=f"{name}.*unsupported precision"):
            prepare_echo_self_dataset(min_total_tokens=1, min_split_tokens=1, **kwargs)


def test_persona_derivatives_share_original_document_group(tmp_path: Path):
    source = tmp_path / "source"
    _write_sources(source)
    output = tmp_path / "grouped"
    prepare_echo_self_dataset(
        source_root=str(source),
        output_dir=str(output),
        echo_depth=2,
        persona_weight=0.2,
        deep_tree_echo_mode=True,
        deep_tree_echo_weight=0.3,
        no_system_prompt=True,
        relentless_persona_mode=True,
        min_total_tokens=1,
        min_split_tokens=1,
    )
    grouped: dict[str, dict[str, set[str]]] = {}
    for split in ("train", "val", "test"):
        for line in (output / f"{split}.jsonl").read_text().splitlines():
            record = json.loads(line)
            entry = grouped.setdefault(
                record["document_group"], {"splits": set(), "categories": set()}
            )
            entry["splits"].add(split)
            entry["categories"].add(record["category"])
            if record["category"] in {
                "deep_tree_echo",
                "zero_prompt",
                "relentless",
            }:
                assert record["origin_group"] == record["document_group"]

    derivative_categories = {"deep_tree_echo", "zero_prompt", "relentless"}
    derivative_groups = [
        entry
        for entry in grouped.values()
        if entry["categories"] & derivative_categories
    ]
    assert derivative_groups
    for entry in derivative_groups:
        assert len(entry["splits"]) == 1
        assert entry["categories"] & {"behavior", "identity"}


def test_boolean_cli_parser_rejects_unknown_values(monkeypatch):
    assert _as_bool("YES") is True
    assert _as_bool("off") is False
    with pytest.raises(argparse.ArgumentTypeError, match="invalid boolean value"):
        _as_bool("truthy")
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_nanecho.py", "--deep_tree_echo_mode", "truthy"],
    )
    with pytest.raises(SystemExit) as exc:
        prepare_main()
    assert exc.value.code == 2


def test_training_rejects_incomplete_or_incompatible_dataset_tokenizer(
    tmp_path: Path,
):
    source = tmp_path / "source"
    _write_sources(source)
    output = tmp_path / "dataset"
    prepare_echo_self_dataset(
        source_root=str(source),
        output_dir=str(output),
        min_total_tokens=1,
        min_split_tokens=1,
    )
    config = TrainingConfig(data_dir=str(output), block_size=8, device="cpu")
    metadata_path = output / "metadata.json"
    valid = json.loads(metadata_path.read_text())
    loader = DataLoader(config)
    loader.load_data()
    assert loader.tokenizer_provenance == valid["tokenizer"]

    for field in ("name", "vocab_size", "eos_token", "eos_token_id"):
        malformed = json.loads(json.dumps(valid))
        malformed["tokenizer"].pop(field)
        metadata_path.write_text(json.dumps(malformed), encoding="utf-8")
        with pytest.raises(ValueError, match=f"incomplete.*{field}"):
            DataLoader(config).load_data()

    incompatible = json.loads(json.dumps(valid))
    incompatible["tokenizer"]["eos_token_id"] = 0
    metadata_path.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible with GPT-2"):
        DataLoader(config).load_data()


def test_corpus_minimum_fails_without_fallback(tmp_path: Path):
    source = tmp_path / "source"
    _write_sources(source)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="configured minimum"):
        prepare_echo_self_dataset(
            source_root=str(source),
            output_dir=str(output),
            min_total_tokens=10_000_000,
        )
    assert not (output / "train.bin").exists()


def test_training_phases_progress_weights_and_never_make_fallback_data(tmp_path: Path):
    config = TrainingConfig(
        data_dir=str(tmp_path / "missing"),
        max_iters=100,
        n_embd=8,
        n_head=2,
        n_layer=1,
        block_size=16,
    )
    phases = EchoSelfLearningPhase(config)
    early = phases.get_dimension_weights(0)
    late = phases.get_dimension_weights(90)
    assert early["cognitive"] > early["dynamic"]
    assert len(set(late.values())) == 1
    with pytest.raises(FileNotFoundError, match="never generates fallback"):
        DataLoader(config).load_data()
    assert not Path(config.data_dir).exists()


def test_all_curriculum_dimensions_are_instantiated_and_consumed():
    config = NanEchoConfig(
        vocab_size=32,
        n_embd=8,
        n_head=2,
        n_layer=1,
        block_size=8,
        dropout=0.0,
    )
    model = NanEchoModel(config).eval()
    block = model.blocks[0]
    assert list(block.persona_dims) == PERSONA_DIMENSIONS
    inputs = torch.tensor([[1, 2, 3]], dtype=torch.long)
    config.dimension_weights = {dimension: 0.0 for dimension in PERSONA_DIMENSIONS}
    without_persona = model(inputs)["logits"]
    config.dimension_weights["dynamic"] = 1.0
    with_dynamic = model(inputs)["logits"]
    assert not torch.allclose(without_persona, with_dynamic)


def test_four_dimension_checkpoint_remains_loadable_and_reports_limit(tmp_path: Path):
    config = NanEchoConfig(
        vocab_size=50257,
        n_embd=8,
        n_head=2,
        n_layer=1,
        block_size=8,
        dropout=0.0,
        persona_dimensions=PERSONA_DIMENSIONS[:4],
    )
    model = NanEchoModel(config)
    checkpoint = tmp_path / "four-dimensions.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                **asdict(config),
                # Historical metadata listed all eight despite constructing four.
                "persona_dimensions": PERSONA_DIMENSIONS,
            },
            "tokenizer": {
                "name": "gpt2",
                "vocab_size": 50257,
                "eos_token": "<|endoftext|>",
                "eos_token_id": 50256,
            },
        },
        checkpoint,
    )
    runtime = NanEchoRuntime.load(checkpoint)
    assert runtime.metadata["active_persona_dimensions"] == PERSONA_DIMENSIONS[:4]
    assert list(runtime.model.blocks[0].persona_dims) == PERSONA_DIMENSIONS[:4]


def test_training_cache_writes_explicit_gpt2_provenance(tmp_path: Path):
    cache = TrainingCache(
        CacheConfig(cache_dir=str(tmp_path / "cache"), auto_cleanup=False)
    )
    model = NanEchoModel(
        NanEchoConfig(
            vocab_size=50257,
            n_embd=8,
            n_head=2,
            n_layer=1,
            block_size=8,
            dropout=0.0,
        )
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    checkpoint_id = cache.save_checkpoint(
        model,
        optimizer,
        None,
        iteration=1,
        epoch=0,
        train_loss=2.0,
        val_loss=2.0,
        learning_rate=0.01,
        model_config=asdict(model.config),
        training_config={},
        data_config={"dataset": "test"},
        force_save=True,
    )
    checkpoint = torch.load(
        tmp_path / "cache" / "checkpoints" / f"{checkpoint_id}.pt",
        weights_only=True,
    )
    assert checkpoint["tokenizer"] == {
        "name": "gpt2",
        "vocab_size": 50257,
        "eos_token": "<|endoftext|>",
        "eos_token_id": 50256,
    }


def test_underperforming_dimensions_create_feedback_artifact(
    tiny_checkpoint: Path, tmp_path: Path
):
    runtime = NanEchoRuntime.load(tiny_checkpoint)
    config = TrainingConfig(
        eval_dir=str(tmp_path / "eval"),
        max_iters=10,
        n_embd=8,
        n_head=2,
        n_layer=1,
        block_size=16,
        device="cpu",
    )
    introspection = Introspection(runtime.model, config)
    metrics = introspection.evaluate_echo_self_quality(1)
    feedback = tmp_path / "eval" / "persona_feedback" / "feedback_00000001.json"
    assert metrics["persona_consistency"] == 0.0
    artifact = json.loads(feedback.read_text())
    assert artifact["convergence_claimed"] is False
    assert set(artifact["underperforming_dimensions"]) == set(PERSONA_DIMENSIONS)


def test_evaluation_uses_generation_and_heldout_perplexity(
    tiny_checkpoint: Path, tmp_path: Path
):
    heldout = tmp_path / "test.txt"
    heldout.write_text("Echo Self uses adaptive attention and careful reasoning. " * 5)
    evaluator = EchoFidelityEvaluator(
        NanEchoRuntime.load(tiny_checkpoint), heldout, max_new_tokens=2
    )
    evaluator.test_prompts = {
        category: prompts[:1] for category, prompts in evaluator.test_prompts.items()
    }
    report = evaluator.run()
    assert report["heldout"]["perplexity"] > 1
    assert len(report["zero_system_prompt"]["details"]) == 6
    assert len(report["prompted"]["details"]) == 6
    assert report["convergence_claimed"] is False
