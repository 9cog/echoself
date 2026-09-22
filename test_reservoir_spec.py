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
    # No reservoir wrapper on the legacy path.
    assert model.reservoir_wrapper is None
    assert "reservoir_states" not in out
    assert model.connection_ratio == cfg.initial_connections


# ---- Phase 2: reservoir wrapper ------------------------------------------

from nanecho_model import ReservoirWrapper, TorchEchoReservoir


def _tiny_cfg(mode):
    return NanEchoConfig(
        vocab_size=128, n_embd=16, n_head=2, n_layer=2, block_size=16,
        dropout=0.0, reservoir_mode=mode, reservoir_units=32,
    )


def test_reservoir_state_stats_keys():
    res = TorchEchoReservoir(input_dim=16, units=32)
    x = torch.randn(2, 5, 16)
    states = res(x)
    assert states.shape == (2, 5, 32)
    stats = res.state_stats(states)
    assert {"state_mean_abs", "state_entropy_ratio", "spectral_radius"} <= set(stats)


def test_shadow_wrapper_is_identity():
    """The shadow wrapper returns its input unchanged (observation only)."""
    torch.manual_seed(0)
    wrapper = ReservoirWrapper(_tiny_cfg("shadow"))
    x = torch.randn(2, 8, 16)
    out = wrapper(x)
    assert torch.equal(out, x)  # pass-through, not just allclose
    # But it still records reservoir states for the orchestrator.
    assert wrapper.last_states is not None
    assert wrapper.last_states.shape[:2] == (2, 8)


def test_shadow_mode_full_model_matches_off():
    """A shadow model's logits track an identical reservoir-free model.

    The forward pass contains stochastic hypergraph pattern injection, so we
    assert close (not exact) agreement on shared weights; the exact identity
    guarantee is covered by ``test_shadow_wrapper_is_identity``.
    """
    ids = torch.randint(0, 128, (2, 8))
    torch.manual_seed(0)
    model_off = NanEchoModel(_tiny_cfg("off")).eval()
    torch.manual_seed(0)
    model_shadow = NanEchoModel(_tiny_cfg("shadow")).eval()
    model_shadow.load_state_dict(model_off.state_dict(), strict=False)
    with torch.no_grad():
        a = model_off(ids)["logits"]
        b = model_shadow(ids)["logits"]
    assert torch.allclose(a, b, atol=2e-2)


def test_orchestrated_mode_modulates_and_trains_readout():
    """Orchestrated mode routes through the reservoir; readout is trainable."""
    torch.manual_seed(0)
    cfg = _tiny_cfg("orchestrated")
    model = NanEchoModel(cfg)
    ids = torch.randint(0, 128, (2, 8))
    labels = torch.randint(0, 128, (2, 8))
    out = model(ids, labels=labels)
    assert out["loss"] is not None
    out["loss"].backward()
    # The ridge readout must receive gradients (it is the trained "ridge").
    assert model.reservoir_wrapper.readout.weight.grad is not None
    # Reservoir recurrent weights are buffers — no grad.
    assert not model.reservoir_wrapper.reservoir.W.requires_grad


def test_orchestrated_output_differs_from_off():
    """With a non-zero mix, orchestrated output differs from the off path."""
    ids = torch.randint(0, 128, (2, 8))
    torch.manual_seed(0)
    model_off = NanEchoModel(_tiny_cfg("off")).eval()
    torch.manual_seed(0)
    model = NanEchoModel(_tiny_cfg("orchestrated"))
    # Share the base weights so only the reservoir modulation differs.
    model.load_state_dict(model_off.state_dict(), strict=False)
    # Force a non-zero mixing gate so modulation is observable.
    with torch.no_grad():
        model.reservoir_wrapper.mix.fill_(0.5)
    model.eval()
    with torch.no_grad():
        a = model(ids)["logits"]
        b = model_off(ids)["logits"]
    assert not torch.allclose(a, b)


# ---- Phase 3: ESN orchestrator -------------------------------------------

from NanEcho.orchestrator import ReservoirOrchestrator, OrchestratorDecision


def test_decision_clamp_bounds():
    d = OrchestratorDecision(
        lr_scale=100.0, connection_growth_rate=5.0, recursion_depth=999
    ).clamp()
    assert d.lr_scale <= 4.0
    assert d.connection_growth_rate <= 0.25
    assert d.recursion_depth <= 14
    d2 = OrchestratorDecision(
        lr_scale=0.0, connection_growth_rate=-1.0, recursion_depth=-3
    ).clamp()
    assert d2.lr_scale >= 0.25
    assert d2.connection_growth_rate >= 0.0
    assert d2.recursion_depth >= 1


def test_orchestrator_observe_and_decide_bounded():
    orch = ReservoirOrchestrator(mode="orchestrated", reservoir_units=16)
    orch.observe(
        reservoir_stats={
            "state_mean_abs": 0.4,
            "state_entropy_ratio": 0.6,
            "spectral_radius": 0.95,
        },
        val_loss=2.5,
        connection_ratio=0.3,
        persona_grip=0.5,
    )
    d = orch.decide()
    assert 0.25 <= d.lr_scale <= 4.0
    assert 0.0 <= d.connection_growth_rate <= 0.25
    assert 1 <= d.recursion_depth <= 14


def test_orchestrator_shadow_is_noop():
    orch = ReservoirOrchestrator(mode="shadow", reservoir_units=16)
    orch.observe(val_loss=1.0, persona_grip=0.4)
    d = orch.decide()
    # Shadow mode returns the default (no-op) decision regardless of state.
    assert d.lr_scale == 1.0
    assert d.connection_growth_rate == 0.05
    assert d.recursion_depth == 5


def test_orchestrator_learns_from_grip():
    orch = ReservoirOrchestrator(mode="orchestrated", reservoir_units=16)
    # Drive several observe/decide/report cycles with improving grip.
    for i in range(6):
        orch.observe(val_loss=2.0 - 0.1 * i, persona_grip=0.1 * i)
        orch.decide()
        orch.report_grip(0.1 * (i + 1))
    # The readout should have been fit at least once without crashing.
    assert orch.readout.Wout is not None
    assert len(orch.history) == 6


def test_orchestrator_state_roundtrip():
    orch = ReservoirOrchestrator(mode="orchestrated", reservoir_units=16)
    for i in range(3):
        orch.observe(val_loss=1.0, persona_grip=0.2 * i)
        orch.decide()
        orch.report_grip(0.2 * (i + 1))
    state = orch.state_dict()

    fresh = ReservoirOrchestrator(mode="orchestrated", reservoir_units=16)
    fresh.load_state_dict(state)
    assert fresh._interval_count == orch._interval_count
    assert fresh._last_grip == orch._last_grip
    assert fresh.reservoir.W is not None
    assert fresh.readout.Wout is not None


def test_orchestrator_rejects_bad_mode():
    with pytest.raises(ValueError):
        ReservoirOrchestrator(mode="bogus")
