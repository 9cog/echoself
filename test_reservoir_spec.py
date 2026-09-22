"""Phase 0 regression tests: feature-flag seam + spec extraction.

Pins current behavior with ``reservoir_mode='off'`` (the default) so the
reservoir evolution cannot silently change the legacy GPT-2 path.
"""

import numpy as np
import pytest
import torch

from NanEcho.spec import (
    GPT2_SPEC,
    CHAR_SPEC,
    ModelSpec,
    TokenizerSpec,
    CharTokenizer,
    TokenizerAdapter,
    RESERVOIR_MODES,
)
from NanEcho.runtime import (
    NanEchoTokenizer,
    TOKENIZER_NAME,
    TOKENIZER_VOCAB_SIZE,
    TOKENIZER_EOS_TOKEN,
    TOKENIZER_EOS_TOKEN_ID,
)
from nanecho_model import NanEchoConfig, NanEchoModel
from train_nanecho import TrainingConfig


# ---- spec module ---------------------------------------------------------


def test_gpt2_spec_matches_legacy_constants():
    assert GPT2_SPEC.name == TOKENIZER_NAME == "gpt2"
    assert GPT2_SPEC.vocab_size == TOKENIZER_VOCAB_SIZE == 50257
    assert GPT2_SPEC.eos_token == TOKENIZER_EOS_TOKEN == "<|endoftext|>"
    assert GPT2_SPEC.eos_token_id == TOKENIZER_EOS_TOKEN_ID == 50256


def test_tokenizer_spec_provenance_roundtrip():
    prov = GPT2_SPEC.provenance()
    assert set(prov) == {"name", "vocab_size", "eos_token", "eos_token_id"}
    assert TokenizerSpec.from_provenance(prov) == GPT2_SPEC


def test_model_spec_rejects_bad_mode():
    with pytest.raises(ValueError):
        ModelSpec(reservoir_mode="ludicrous")
    for mode in RESERVOIR_MODES:
        ModelSpec(reservoir_mode=mode)  # should not raise


def test_model_spec_head_divisibility():
    with pytest.raises(ValueError):
        ModelSpec(n_embd=10, n_head=3)


# ---- tokenizer adapters ---------------------------------------------------


def test_nanecho_tokenizer_satisfies_adapter():
    tok = NanEchoTokenizer()
    assert isinstance(tok, TokenizerAdapter)
    assert tok.provenance() == GPT2_SPEC.provenance()


def test_char_tokenizer_satisfies_adapter():
    tok = CharTokenizer()
    assert isinstance(tok, TokenizerAdapter)
    assert tok.provenance() == CHAR_SPEC.provenance()
    ids = tok.encode("Echo")
    assert tok.decode(ids) == "Echo"


def test_nanecho_tokenizer_explicit_spec():
    tok = NanEchoTokenizer(GPT2_SPEC)
    assert tok.vocab_size == 50257
    with pytest.raises(RuntimeError):
        # A spec that disagrees with the installed encoding must fail fast.
        NanEchoTokenizer(TokenizerSpec("gpt2", 123, "<|endoftext|>", 50256))


# ---- feature-flag defaults (zero behavior change) -------------------------


def test_training_config_defaults_reservoir_off():
    cfg = TrainingConfig()
    assert cfg.reservoir_mode == "off"
    assert cfg.reservoir_units == 256


def test_training_config_rejects_bad_mode():
    with pytest.raises(ValueError):
        TrainingConfig(reservoir_mode="bogus")


def test_model_config_defaults_reservoir_off():
    cfg = NanEchoConfig()
    assert cfg.reservoir_mode == "off"
    with pytest.raises(ValueError):
        NanEchoConfig(reservoir_mode="bogus")


def test_model_forward_identical_with_flag_off():
    """A tiny model with the flag off behaves exactly as before."""
    torch.manual_seed(0)
    cfg = NanEchoConfig(
        vocab_size=128, n_embd=16, n_head=2, n_layer=2, block_size=16,
        dropout=0.0,
    )
    model = NanEchoModel(cfg)
    ids = torch.randint(0, 128, (2, 8))
    out = model(ids)
    assert out["logits"].shape == (2, 8, 128)
    # No reservoir attributes should exist on the legacy path.
    assert not hasattr(model, "reservoir")
    assert model.connection_ratio == cfg.initial_connections
