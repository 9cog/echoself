"""Tokenizer and model specifications for Deep Tree Echo training.

The primary objective is dynamic, persona-driven configuration of
tokenization, topology, and model size. The GPT-2 architecture is one
instance of a spec — not the law. This module centralizes the constants
that were previously hard-coded so they can be selected (and eventually
searched) per persona grip.

Reservoir modes
---------------
- ``off``:          legacy transformer-only path (default; zero behavior change)
- ``shadow``:       reservoir computes alongside the transformer but does not
                    alter outputs — used for observation and metric collection
- ``orchestrated``: reservoir drives embedding modulation and the ESN
                    orchestrator controls training hyperparameters
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

# Valid reservoir modes
RESERVOIR_MODES = ("off", "shadow", "orchestrated")


@dataclass(frozen=True)
class TokenizerSpec:
    """Portable identity of a tokenizer used for a dataset/checkpoint.

    The fields mirror the provenance schema written into dataset
    ``metadata.json`` and checkpoint ``tokenizer`` blocks, so provenance
    validation becomes a spec-equality check rather than a GPT-2 equality
    check.
    """

    name: str
    vocab_size: int
    eos_token: str
    eos_token_id: int

    def provenance(self) -> Dict[str, Any]:
        """Return the provenance dict persisted in datasets and checkpoints."""
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "eos_token": self.eos_token,
            "eos_token_id": self.eos_token_id,
        }

    @classmethod
    def from_provenance(cls, declared: Dict[str, Any]) -> "TokenizerSpec":
        return cls(
            name=str(declared["name"]),
            vocab_size=int(declared["vocab_size"]),
            eos_token=str(declared["eos_token"]),
            eos_token_id=int(declared["eos_token_id"]),
        )


#: The historical default. GPT-2 remains the fallback until a persona-fit
#: tokenizer is selected by the Phase-1 search.
GPT2_SPEC = TokenizerSpec(
    name="gpt2",
    vocab_size=50257,
    eos_token="<|endoftext|>",
    eos_token_id=50256,
)


@dataclass(frozen=True)
class ModelSpec:
    """Topology and size declaration for a NanEcho model instance.

    These are the knobs Phase 4 adapts dynamically. ``reservoir_units`` is
    only used when ``reservoir_mode != 'off'``.
    """

    vocab_size: int = GPT2_SPEC.vocab_size
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    block_size: int = 1024
    reservoir_mode: str = "off"
    reservoir_units: int = 256
    reservoir_spectral_radius: float = 0.95

    def __post_init__(self) -> None:
        if self.reservoir_mode not in RESERVOIR_MODES:
            raise ValueError(
                f"reservoir_mode must be one of {RESERVOIR_MODES}, "
                f"got {self.reservoir_mode!r}"
            )
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSpec":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


@runtime_checkable
class TokenizerAdapter(Protocol):
    """Interface every tokenizer usable in training/serving must satisfy.

    ``NanEchoTokenizer`` (GPT-2/tiktoken), the custom ``dte_tokenizer``
    assets, and char-level fallbacks are all interchangeable through this
    protocol, enabling persona-gripped tokenizer selection in Phase 1.
    """

    name: str
    eos_token: str
    eos_token_id: int
    vocab_size: int

    def encode(self, text: str) -> List[int]: ...

    def decode(self, token_ids: Iterable[int]) -> str: ...

    def provenance(self) -> Dict[str, Any]: ...


class CharTokenizer:
    """Character-level tokenizer used as a dependency-free fallback.

    Useful for tiny smoke tests and for the tokenizer-search baseline; it
    satisfies ``TokenizerAdapter`` without requiring tiktoken or network
    access.
    """

    name = "char"
    eos_token = "<|endoftext|>"

    def __init__(self) -> None:
        # 0..255 byte values + one EOS id
        self.vocab_size = 256
        self.eos_token_id = 255

    def encode(self, text: str) -> List[int]:
        return [ord(c) % 256 for c in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(chr(int(t) % 256) for t in token_ids if int(t) != self.eos_token_id)

    def provenance(self) -> Dict[str, Any]:
        return TokenizerSpec(
            name=self.name,
            vocab_size=self.vocab_size,
            eos_token=self.eos_token,
            eos_token_id=self.eos_token_id,
        ).provenance()


CHAR_SPEC = TokenizerSpec(
    name="char", vocab_size=256, eos_token="<|endoftext|>", eos_token_id=255
)


def tokenizer_from_spec(spec: TokenizerSpec) -> TokenizerAdapter:
    """Instantiate a tokenizer adapter from its spec.

    The ``char`` tokenizer is dependency-free; anything else is assumed to be
    a tiktoken encoding (e.g. ``gpt2``). Additional persona-fit tokenizers can
    be registered here as the Phase-1 search discovers them.
    """
    if spec.name == CHAR_SPEC.name:
        return CharTokenizer()
    from NanEcho.runtime import NanEchoTokenizer

    return NanEchoTokenizer(spec)
