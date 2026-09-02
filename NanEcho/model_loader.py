#!/usr/bin/env python3
"""
NanEcho Model Loader Utility

Central utility for loading NanEcho model checkpoints for inference,
evaluation, and server deployments. Replaces scattered model loading
code with a single, consistent interface.

This module provides:
- Model loading from checkpoints
- Model validation and compatibility checking
- Convenient wrapper for common use cases

Security Note:
    Model checkpoints are loaded using PyTorch's torch.load() which
    deserializes Python objects. Only load checkpoints from trusted
    sources. Untrusted checkpoints could contain malicious code.
    
    For production deployments, consider:
    - Verifying checkpoint signatures/hashes
    - Loading only from controlled artifact storage
    - Running inference in sandboxed environments
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directories to path for imports
MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import IncompatibleCheckpointError, NanEchoRuntime, NanEchoTokenizer


class ModelLoader:
    """
    Central utility for loading and validating NanEcho model checkpoints.
    
    This class provides a consistent interface for loading models across
    different use cases (CLI, server, evaluation) while handling validation
    and error reporting.
    
    Example usage:
        loader = ModelLoader()
        runtime = loader.load("path/to/checkpoint.pt", device="cuda")
        
        # Generate text
        response = runtime.generate("Hello, Echo Self!")
        
        # Get model info
        info = loader.get_model_info(runtime)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the model loader.
        
        Args:
            verbose: If True, print status messages during operations
        """
        self.verbose = verbose
        self._loaded_runtimes: Dict[str, NanEchoRuntime] = {}
    
    def _log(self, message: str) -> None:
        """Print a message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def load(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        cache: bool = True
    ) -> NanEchoRuntime:
        """
        Load a NanEcho model from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the model checkpoint file
            device: Device to load the model on ("cpu" or "cuda")
            cache: If True, cache the loaded runtime for future calls
            
        Returns:
            NanEchoRuntime instance with the loaded model
            
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist
            IncompatibleCheckpointError: If the checkpoint format is invalid
            RuntimeError: If CUDA is requested but unavailable
        """
        path = Path(checkpoint_path).expanduser().resolve()
        cache_key = f"{path}:{device}"
        
        # Return cached runtime if available
        if cache and cache_key in self._loaded_runtimes:
            self._log(f"✓ Using cached model from {path}")
            return self._loaded_runtimes[cache_key]
        
        self._log(f"Loading NanEcho model from {path}...")
        
        try:
            runtime = NanEchoRuntime.load(str(path), device)
            
            # Cache the runtime
            if cache:
                self._loaded_runtimes[cache_key] = runtime
            
            self._log(f"✓ Model loaded successfully")
            self._log(f"  - Checkpoint iteration: {runtime.metadata['iteration']}")
            self._log(f"  - Parameters: {sum(p.numel() for p in runtime.model.parameters()):,}")
            self._log(f"  - Device: {runtime.device}")
            
            return runtime
            
        except FileNotFoundError as e:
            self._log(f"✗ Checkpoint not found: {path}")
            raise
        except IncompatibleCheckpointError as e:
            self._log(f"✗ Incompatible checkpoint: {e}")
            raise
        except RuntimeError as e:
            self._log(f"✗ Runtime error: {e}")
            raise
    
    def validate_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        """
        Validate a checkpoint file without fully loading the model.
        
        Args:
            checkpoint_path: Path to the checkpoint file to validate
            
        Returns:
            Dictionary with validation results:
            - valid: True if checkpoint is valid
            - format: Checkpoint format identifier
            - schema: Source schema (training/cached-training)
            - iteration: Training iteration number
            - vocab_size: Model vocabulary size
            - errors: List of validation errors (if any)
        """
        import torch
        
        path = Path(checkpoint_path).expanduser().resolve()
        result = {
            "valid": False,
            "path": str(path),
            "errors": []
        }
        
        if not path.is_file():
            result["errors"].append(f"File not found: {path}")
            return result
        
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            
            # Check required keys
            if "model_state_dict" not in checkpoint:
                result["errors"].append("Missing 'model_state_dict'")
                return result
            
            # Extract metadata
            if "config" in checkpoint:
                config = checkpoint["config"]
                result["schema"] = "training"
            elif "model_config" in checkpoint:
                config = checkpoint["model_config"]
                result["schema"] = "cached-training"
            else:
                result["errors"].append("Missing architecture metadata")
                return result
            
            result["format"] = checkpoint.get("format", "unknown")
            result["iteration"] = checkpoint.get("iteration", 0)
            result["vocab_size"] = config.get("vocab_size", 0)
            result["n_layer"] = config.get("n_layer", 0)
            result["n_head"] = config.get("n_head", 0)
            result["n_embd"] = config.get("n_embd", 0)
            
            # Validate tokenizer provenance
            tokenizer_meta = checkpoint.get("tokenizer")
            if not isinstance(tokenizer_meta, dict):
                result["errors"].append("Missing or invalid tokenizer provenance")
            else:
                result["tokenizer"] = tokenizer_meta
            
            if not result["errors"]:
                result["valid"] = True
            
            return result
            
        except Exception as e:
            result["errors"].append(f"Failed to load checkpoint: {e}")
            return result
    
    def get_model_info(self, runtime: NanEchoRuntime) -> Dict[str, Any]:
        """
        Get comprehensive information about a loaded model.
        
        Args:
            runtime: The NanEchoRuntime instance to query
            
        Returns:
            Dictionary with model information
        """
        config = runtime.config
        model = runtime.model
        
        return {
            "checkpoint_path": str(runtime.checkpoint_path),
            "checkpoint_iteration": runtime.metadata["iteration"],
            "format": runtime.metadata.get("format", "unknown"),
            "schema": runtime.metadata.get("schema", "unknown"),
            "device": str(runtime.device),
            "architecture": {
                "vocab_size": config.vocab_size,
                "n_embd": config.n_embd,
                "n_head": config.n_head,
                "n_layer": config.n_layer,
                "block_size": config.block_size,
                "bias": config.bias,
            },
            "echo_features": {
                "enable_adaptive_attention": config.enable_adaptive_attention,
                "enable_recursive_reasoning": config.enable_recursive_reasoning,
                "enable_hypergraph_patterns": config.enable_hypergraph_patterns,
                "enable_persona_dimensions": config.enable_persona_dimensions,
                "max_recursion_depth": config.max_recursion_depth,
            },
            "persona_dimensions": list(config.persona_dimensions) if config.enable_persona_dimensions else [],
            "parameters": {
                "total": sum(p.numel() for p in model.parameters()),
                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
            "connection_ratio": model.connection_ratio,
            "metrics": runtime.metadata.get("metrics", {}),
            "tokenizer": runtime.metadata.get("tokenizer", {}),
        }
    
    def clear_cache(self) -> None:
        """Clear all cached model runtimes."""
        self._loaded_runtimes.clear()
        self._log("Model cache cleared")


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> NanEchoRuntime:
    """
    Convenience function to load a NanEcho model.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to use ("cpu" or "cuda")
        
    Returns:
        NanEchoRuntime instance
    """
    loader = ModelLoader(verbose=False)
    return loader.load(checkpoint_path, device)


def main() -> int:
    """Command-line interface for model inspection."""
    parser = argparse.ArgumentParser(
        description="NanEcho Model Loader - Inspect and validate checkpoints"
    )
    parser.add_argument(
        "checkpoint_path",
        help="Path to the NanEcho checkpoint file"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate the checkpoint without loading"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load on (cpu/cuda)"
    )
    parser.add_argument(
        "--generate",
        type=str,
        help="Generate text from this prompt"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    loader = ModelLoader()
    
    if args.validate:
        result = loader.validate_checkpoint(args.checkpoint_path)
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] else 1
    
    try:
        runtime = loader.load(args.checkpoint_path, args.device)
        
        if args.generate:
            print(f"\n--- Generating from prompt ---")
            print(f"Prompt: {args.generate}")
            print(f"Response: ", end="", flush=True)
            response = runtime.generate(
                args.generate,
                max_new_tokens=args.max_tokens,
                temperature=0.7
            )
            print(response)
            print()
        else:
            info = loader.get_model_info(runtime)
            print("\n--- Model Information ---")
            print(json.dumps(info, indent=2, default=str))
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
