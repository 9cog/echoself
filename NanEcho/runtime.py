"""Production inference runtime shared by NanEcho CLI, API, and evaluation."""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import tiktoken
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanecho_model import NanEchoConfig, NanEchoModel


TOKENIZER_NAME = "gpt2"
TOKENIZER_VOCAB_SIZE = 50257
TOKENIZER_EOS_TOKEN = "<|endoftext|>"
TOKENIZER_EOS_TOKEN_ID = 50256
CHECKPOINT_FORMAT = "nanecho-pytorch-v1"
# exp(80) remains finite in standard Python floats while bounding unusable losses.
MAX_PERPLEXITY_LOSS = 80.0


class IncompatibleCheckpointError(ValueError):
    """Raised when a checkpoint cannot safely instantiate NanEchoModel."""


class NanEchoTokenizer:
    """Single GPT-2 tokenizer implementation used throughout NanEcho."""

    name = TOKENIZER_NAME
    eos_token = TOKENIZER_EOS_TOKEN

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding(self.name)
        self.eos_token_id = self._encoding.eot_token
        self.vocab_size = self._encoding.n_vocab
        if (
            self.vocab_size != TOKENIZER_VOCAB_SIZE
            or self.eos_token_id != TOKENIZER_EOS_TOKEN_ID
        ):
            raise RuntimeError("Installed GPT-2 tokenizer metadata is incompatible")

    def provenance(self) -> Dict[str, Any]:
        """Return the complete, portable tokenizer identity declaration."""
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "eos_token": self.eos_token,
            "eos_token_id": self.eos_token_id,
        }

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, allowed_special=set(), disallowed_special=())

    def decode(self, token_ids: Iterable[int]) -> str:
        ids = [int(token_id) for token_id in token_ids]
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in ids):
            raise ValueError("Cannot decode token outside the GPT-2 tokenizer vocabulary")
        return self._encoding.decode(ids)

    def token_bytes(self, token_id: int) -> bytes:
        return self._encoding.decode_single_token_bytes(int(token_id))


def _normalise_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    prefix = "_orig_mod."
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _validate_tokenizer_provenance(
    declared: Any, tokenizer: NanEchoTokenizer
) -> Dict[str, Any]:
    """Require an unambiguous tokenizer declaration for every checkpoint."""
    if not isinstance(declared, dict):
        raise IncompatibleCheckpointError(
            "Missing tokenizer provenance. Legacy checkpoints, including character-tokenized "
            "checkpoints with vocab_size 50257, must be retrained or explicitly migrated."
        )
    required = ("name", "vocab_size", "eos_token", "eos_token_id")
    missing = [key for key in required if key not in declared]
    if missing:
        raise IncompatibleCheckpointError(
            "Tokenizer provenance is incomplete; missing " + ", ".join(missing)
        )
    expected = tokenizer.provenance()
    incompatible = [
        f"{key}={declared.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if declared.get(key) != value
    ]
    if incompatible:
        raise IncompatibleCheckpointError(
            "Checkpoint tokenizer provenance is incompatible with GPT-2: "
            + "; ".join(incompatible)
        )
    return expected


def _state_persona_dimensions(state_dict: Dict[str, Any]) -> list[str]:
    """Return persona modules actually represented by a checkpoint."""
    dimensions: list[str] = []
    marker = ".persona_dims."
    for key in state_dict:
        if marker not in key:
            continue
        dimension = key.split(marker, 1)[1].split(".", 1)[0]
        if dimension and dimension not in dimensions:
            dimensions.append(dimension)
    return dimensions


def _checkpoint_parts(checkpoint: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    if not isinstance(checkpoint, dict):
        raise IncompatibleCheckpointError("Checkpoint root must be a dictionary")
    if "model_state_dict" not in checkpoint:
        if "model" in checkpoint and "model_args" in checkpoint:
            raise IncompatibleCheckpointError(
                "nanoGPT checkpoints are not NanEchoModel checkpoints; use NanEcho/server.py "
                "or retrain/export with train_nanecho.py"
            )
        raise IncompatibleCheckpointError("Missing required 'model_state_dict'")

    if isinstance(checkpoint.get("config"), dict):
        return checkpoint["model_state_dict"], checkpoint["config"], "training"
    if isinstance(checkpoint.get("model_config"), dict):
        return checkpoint["model_state_dict"], checkpoint["model_config"], "cached-training"
    raise IncompatibleCheckpointError(
        "Missing NanEcho architecture metadata: expected 'config' or 'model_config'"
    )


def _build_config(raw_config: Dict[str, Any], tokenizer: NanEchoTokenizer) -> NanEchoConfig:
    required = ("vocab_size", "n_embd", "n_head", "n_layer", "block_size", "bias")
    missing = [key for key in required if key not in raw_config]
    if missing:
        raise IncompatibleCheckpointError(
            "Checkpoint architecture metadata is incomplete; missing " + ", ".join(missing)
        )

    allowed = {field.name for field in fields(NanEchoConfig)}
    values = {key: value for key, value in raw_config.items() if key in allowed}
    values["dropout"] = 0.0
    try:
        config = NanEchoConfig(**values)
    except (TypeError, ValueError) as exc:
        raise IncompatibleCheckpointError(f"Invalid NanEcho configuration: {exc}") from exc

    if config.n_embd <= 0 or config.n_head <= 0 or config.n_embd % config.n_head:
        raise IncompatibleCheckpointError("n_embd must be positive and divisible by n_head")
    if config.n_layer <= 0 or config.block_size <= 1:
        raise IncompatibleCheckpointError("n_layer must be positive and block_size must exceed 1")
    if config.vocab_size < tokenizer.vocab_size:
        raise IncompatibleCheckpointError(
            f"Checkpoint vocabulary ({config.vocab_size}) is smaller than GPT-2 "
            f"tokenizer vocabulary ({tokenizer.vocab_size}); legacy character-tokenized "
            "checkpoints are not supported"
        )
    return config


class NanEchoRuntime:
    """Loaded NanEchoModel plus generation, scoring, and export metadata."""

    def __init__(
        self,
        model: NanEchoModel,
        tokenizer: NanEchoTokenizer,
        device: torch.device,
        checkpoint_path: Path,
        metadata: Dict[str, Any],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.metadata = metadata
        self.config = model.config

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str = "cpu") -> "NanEchoRuntime":
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"NanEcho checkpoint not found: {path}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{device}', but CUDA is unavailable")
        resolved_device = torch.device(device)

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch 2.0 compatibility
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            # Existing trainer checkpoints may contain NumPy scalar metrics, which the
            # restricted loader rejects. The path is an operator-supplied local artifact,
            # never request data; deployment documentation requires it to be trusted.
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:
                raise IncompatibleCheckpointError(
                    f"Unable to read checkpoint: {exc}"
                ) from exc

        state_dict, raw_config, schema = _checkpoint_parts(checkpoint)
        if not isinstance(state_dict, dict) or not state_dict:
            raise IncompatibleCheckpointError("'model_state_dict' must be a non-empty dictionary")
        state_dict = _normalise_state_dict(state_dict)

        tokenizer = NanEchoTokenizer()
        tokenizer_metadata = _validate_tokenizer_provenance(
            checkpoint.get("tokenizer"), tokenizer
        )

        raw_config = dict(raw_config)
        active_dimensions = _state_persona_dimensions(state_dict)
        if active_dimensions and raw_config.get("enable_persona_dimensions", True):
            declared_dimensions = raw_config.get("persona_dimensions") or []
            ordered = [
                dimension
                for dimension in declared_dimensions
                if dimension in active_dimensions
            ]
            ordered.extend(
                dimension
                for dimension in active_dimensions
                if dimension not in ordered
            )
            # Older NanEcho checkpoints instantiated only the first four
            # configured dimensions. Restrict construction to modules proven
            # by the state dict so those checkpoints remain loadable.
            raw_config["persona_dimensions"] = ordered
        config = _build_config(raw_config, tokenizer)
        model = NanEchoModel(config)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise IncompatibleCheckpointError(
                f"State dictionary does not match declared NanEcho architecture: {exc}"
            ) from exc

        model.connection_ratio = float(
            checkpoint.get("connection_ratio", config.initial_connections)
        )
        model.current_iteration = int(
            checkpoint.get("current_iteration", checkpoint.get("iteration", 0))
        )
        model.to(resolved_device).eval()
        metadata = {
            "format": checkpoint.get("format", CHECKPOINT_FORMAT),
            "schema": schema,
            "iteration": int(checkpoint.get("iteration", 0)),
            "metrics": checkpoint.get("metrics", {}),
            "tokenizer": tokenizer_metadata,
            "active_persona_dimensions": list(config.persona_dimensions)
            if config.enable_persona_dimensions
            else [],
        }
        return cls(model, tokenizer, resolved_device, path, metadata)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        logits = logits / temperature
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            cutoff = torch.topk(logits, top_k).values[..., -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        if 0 < top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            mask = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
            logits = logits.masked_fill(mask, float("-inf"))
        probabilities = F.softmax(logits, dim=-1)
        return (
            torch.multinomial(probabilities, num_samples=1, generator=generator)
            if do_sample
            else torch.argmax(probabilities, dim=-1, keepdim=True)
        )

    def generate_ids(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        seed: Optional[int] = None,
        token_callback: Optional[Callable[[int], bool]] = None,
    ) -> list[int]:
        if not prompt:
            raise ValueError("prompt must not be empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        prompt_ids = self.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.eos_token_id]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated: list[int] = []
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.config.block_size :]
                logits = self.model(context, generator=generator)["logits"][
                    :, -1, : self.tokenizer.vocab_size
                ]
                next_token = self._sample(
                    logits, temperature, top_k, top_p, do_sample, generator
                )
                token_id = int(next_token.item())
                input_ids = torch.cat((input_ids, next_token), dim=1)
                if token_id == self.tokenizer.eos_token_id:
                    break
                generated.append(token_id)
                if token_callback is not None and not token_callback(token_id):
                    break
        return generated

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self.decode(self.generate_ids(prompt, **kwargs))

    def perplexity(self, text: str, stride: Optional[int] = None) -> float:
        token_ids = self.encode(text)
        if len(token_ids) < 2:
            raise ValueError("Perplexity text must encode to at least two tokens")
        block_size = self.config.block_size
        stride = max(1, block_size // 2) if stride is None else stride
        if not 1 <= stride < block_size:
            raise ValueError("stride must be between 1 and block_size - 1")
        losses: list[tuple[float, int]] = []
        with torch.inference_mode():
            # Targets are absolute token positions [target_start, target_end).
            # Each position 1..N-1 appears in exactly one range, while the
            # window retains as much preceding context as block_size permits.
            for target_start in range(1, len(token_ids), stride):
                target_end = min(target_start + stride, len(token_ids))
                context_start = max(0, target_end - block_size)
                chunk = token_ids[context_start:target_end]
                inputs = torch.tensor([chunk], dtype=torch.long, device=self.device)
                labels = torch.full_like(inputs, -100)
                first_target = target_start - context_start
                labels[:, first_target:] = inputs[:, first_target:]
                loss = self.model(inputs, labels=labels)["loss"]
                predicted = target_end - target_start
                losses.append((float(loss.item()), predicted))
        if not losses:
            raise ValueError("No valid perplexity windows were produced")
        mean_loss = sum(loss * count for loss, count in losses) / sum(
            count for _, count in losses
        )
        return math.exp(min(mean_loss, MAX_PERPLEXITY_LOSS))

    def native_checkpoint(self) -> Dict[str, Any]:
        """Return a deployment checkpoint with no optimizer or training state."""
        return {
            "format": CHECKPOINT_FORMAT,
            "model_state_dict": self.model.state_dict(),
            "config": asdict(self.config),
            "tokenizer": self.tokenizer.provenance(),
            "iteration": self.metadata["iteration"],
            "connection_ratio": self.model.connection_ratio,
            "metrics": self.metadata.get("metrics", {}),
        }
