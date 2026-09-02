#!/usr/bin/env python3
"""Export genuine NanEcho deployment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from runtime import NanEchoRuntime


class _LogitsOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)["logits"]


def export_native(runtime: NanEchoRuntime, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(runtime.native_checkpoint(), output_path)
    return output_path


def export_onnx(runtime: NanEchoRuntime, output_path: Path, opset: int = 17) -> Path:
    """Export logits inference when the installed PyTorch ONNX stack supports it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = _LogitsOnly(runtime.model).eval()
    example_length = min(16, runtime.config.block_size)
    example = torch.ones((1, example_length), dtype=torch.long, device=runtime.device)
    torch.onnx.export(
        model,
        example,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export NanEcho as native PyTorch and optionally ONNX"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="exports/nanecho")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--onnx", action="store_true", help="Also attempt ONNX export")
    args = parser.parse_args()

    runtime = NanEchoRuntime.load(args.model_path, args.device)
    output_dir = Path(args.output_dir)
    artifacts = {
        "pytorch": str(export_native(runtime, output_dir / "nanecho.pt")),
        "onnx": None,
    }
    if args.onnx:
        try:
            artifacts["onnx"] = str(export_onnx(runtime, output_dir / "nanecho.onnx"))
        except Exception as exc:
            print(f"ONNX export is not viable in this environment: {exc}")
            return 2

    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifacts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
