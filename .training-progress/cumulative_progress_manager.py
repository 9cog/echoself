#!/usr/bin/env python3
"""
Cumulative Training Progress Manager

This module manages the cumulative training progress tracking system for
Deep Tree Echo persona training. It enables training sessions to accumulate
iterations across multiple runs, automatically scaling model parameters
based on total training progress.

Usage:
    # Get current progress and recommended parameters
    python cumulative_progress_manager.py get_params

    # Update progress after training session
    python cumulative_progress_manager.py update --iterations 200 --val_loss 1.97 --train_loss 2.22

    # Get next session's target iteration range
    python cumulative_progress_manager.py next_session --session_iters 200

Example GitHub Actions usage:
    # At start of training
    PROGRESS=$(python cumulative_progress_manager.py get_params)
    START_ITER=$(echo $PROGRESS | jq -r '.start_iteration')
    END_ITER=$(echo $PROGRESS | jq -r '.end_iteration')
    
    # After training
    python cumulative_progress_manager.py update --iterations $TRAINED_ITERS --val_loss $VAL_LOSS
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class CumulativeProgressManager:
    """Manages cumulative training progress across sessions."""
    
    def __init__(self, progress_file: str = None):
        """Initialize the progress manager.
        
        Args:
            progress_file: Path to the cumulative progress JSON file.
                          Defaults to .training-progress/cumulative_progress.json
        """
        if progress_file is None:
            # Find the .training-progress directory relative to this script or repo root
            script_dir = Path(__file__).parent
            if script_dir.name == '.training-progress':
                progress_file = script_dir / 'cumulative_progress.json'
            else:
                progress_file = script_dir / '.training-progress' / 'cumulative_progress.json'
        
        self.progress_file = Path(progress_file)
        self.progress_data = self._load_progress()
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load progress data from file, or create default if not exists."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        else:
            return self._create_default_progress()
    
    def _create_default_progress(self) -> Dict[str, Any]:
        """Create default progress structure."""
        return {
            "version": "1.0.0",
            "description": "Cumulative training progress tracking for Deep Tree Echo persona",
            "total_iterations_completed": 0,
            "total_sessions": 0,
            "current_model_params": {
                "n_layer": 4,
                "n_head": 4,
                "n_embd": 256,
                "block_size": 1024,
                "vocab_size": 50304
            },
            "best_metrics": {
                "best_val_loss": float('inf'),
                "final_train_loss": float('inf')
            },
            "scaling_schedule": self._get_default_scaling_schedule(),
            "session_history": [],
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }
    
    def _get_default_scaling_schedule(self) -> Dict[str, Any]:
        """Get the default model scaling schedule."""
        return {
            "description": "Model scaling based on cumulative training progress",
            "thresholds": [
                {
                    "iterations": 0,
                    "n_layer": 4,
                    "n_head": 4,
                    "n_embd": 256,
                    "learning_rate": "2e-4",
                    "batch_size": 2
                },
                {
                    "iterations": 500,
                    "n_layer": 6,
                    "n_head": 6,
                    "n_embd": 384,
                    "learning_rate": "1e-4",
                    "batch_size": 4
                },
                {
                    "iterations": 2000,
                    "n_layer": 8,
                    "n_head": 8,
                    "n_embd": 512,
                    "learning_rate": "6e-5",
                    "batch_size": 6
                },
                {
                    "iterations": 10000,
                    "n_layer": 12,
                    "n_head": 12,
                    "n_embd": 768,
                    "learning_rate": "3e-5",
                    "batch_size": 8
                },
                {
                    "iterations": 50000,
                    "n_layer": 16,
                    "n_head": 16,
                    "n_embd": 1024,
                    "learning_rate": "1e-5",
                    "batch_size": 12
                }
            ]
        }
    
    def _save_progress(self) -> None:
        """Save progress data to file."""
        self.progress_data["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        # Ensure directory exists
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress_data, f, indent=2)
    
    def get_total_iterations(self) -> int:
        """Get the total number of iterations completed across all sessions."""
        return self.progress_data.get("total_iterations_completed", 0)
    
    def get_current_params(self) -> Dict[str, Any]:
        """Get current model parameters based on cumulative progress."""
        total_iters = self.get_total_iterations()
        thresholds = self.progress_data.get("scaling_schedule", {}).get("thresholds", [])
        
        # Find the appropriate scaling tier based on total iterations
        current_params = None
        for threshold in sorted(thresholds, key=lambda x: x["iterations"], reverse=True):
            if total_iters >= threshold["iterations"]:
                current_params = {
                    "n_layer": threshold["n_layer"],
                    "n_head": threshold["n_head"],
                    "n_embd": threshold["n_embd"],
                    "learning_rate": threshold["learning_rate"],
                    "batch_size": threshold["batch_size"]
                }
                break
        
        if current_params is None:
            # Default to smallest config
            current_params = {
                "n_layer": 4,
                "n_head": 4,
                "n_embd": 256,
                "learning_rate": "2e-4",
                "batch_size": 2
            }
        
        return current_params
    
    def get_next_session_params(self, session_iterations: int = 200) -> Dict[str, Any]:
        """Get parameters for the next training session.
        
        Args:
            session_iterations: Number of iterations to run in this session.
        
        Returns:
            Dictionary with start_iteration, end_iteration, and model parameters.
        """
        start_iter = self.get_total_iterations()
        end_iter = start_iter + session_iterations
        params = self.get_current_params()
        
        # Check if we'll cross a scaling threshold during this session
        thresholds = self.progress_data.get("scaling_schedule", {}).get("thresholds", [])
        crossing_threshold = None
        for threshold in thresholds:
            if start_iter < threshold["iterations"] <= end_iter:
                crossing_threshold = threshold
                break
        
        return {
            "start_iteration": start_iter,
            "end_iteration": end_iter,
            "session_iterations": session_iterations,
            "total_sessions_so_far": self.progress_data.get("total_sessions", 0),
            "crossing_threshold": crossing_threshold is not None,
            "threshold_at": crossing_threshold["iterations"] if crossing_threshold else None,
            "current_params": params,
            "next_params": crossing_threshold if crossing_threshold else params,
            "best_val_loss": self.progress_data.get("best_metrics", {}).get("best_val_loss", None),
            "should_scale_model": crossing_threshold is not None
        }
    
    def update_progress(
        self,
        iterations_completed: int,
        val_loss: float,
        train_loss: float,
        model_params: Optional[Dict[str, Any]] = None,
        workflow: str = "agent-neuro-train.yml",
        trigger: str = "unknown"
    ) -> Dict[str, Any]:
        """Update progress after a training session.
        
        Args:
            iterations_completed: Number of iterations completed in this session.
            val_loss: Final validation loss.
            train_loss: Final training loss.
            model_params: Model parameters used (optional).
            workflow: Name of the workflow that ran this session.
            trigger: What triggered the training (ci, schedule, manual, etc.).
        
        Returns:
            Updated progress summary.
        """
        start_iter = self.get_total_iterations()
        end_iter = start_iter + iterations_completed
        
        # Update total iterations
        self.progress_data["total_iterations_completed"] = end_iter
        self.progress_data["total_sessions"] = self.progress_data.get("total_sessions", 0) + 1
        
        # Update best metrics
        best_metrics = self.progress_data.get("best_metrics", {})
        if val_loss < best_metrics.get("best_val_loss", float('inf')):
            best_metrics["best_val_loss"] = val_loss
        best_metrics["final_train_loss"] = train_loss
        self.progress_data["best_metrics"] = best_metrics
        
        # Update current model params
        if model_params:
            self.progress_data["current_model_params"].update(model_params)
        
        # Add to session history
        session_entry = {
            "session_id": f"session_{self.progress_data['total_sessions']}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "iterations_start": start_iter,
            "iterations_end": end_iter,
            "iterations_this_session": iterations_completed,
            "model_params": model_params or self.get_current_params(),
            "metrics": {
                "final_train_loss": train_loss,
                "best_val_loss": val_loss
            },
            "workflow": workflow,
            "trigger": trigger
        }
        
        # Keep only last 100 sessions in history to prevent file bloat
        history = self.progress_data.get("session_history", [])
        history.append(session_entry)
        if len(history) > 100:
            history = history[-100:]
        self.progress_data["session_history"] = history
        
        # Save updated progress
        self._save_progress()
        
        # Check if we should recommend scaling up
        next_params = self.get_current_params()
        current_params = model_params or {}
        
        return {
            "total_iterations": end_iter,
            "total_sessions": self.progress_data["total_sessions"],
            "best_val_loss": best_metrics["best_val_loss"],
            "session_summary": session_entry,
            "recommend_scale_up": (
                next_params.get("n_layer", 4) > current_params.get("n_layer", 4) or
                next_params.get("n_embd", 256) > current_params.get("n_embd", 256)
            ),
            "next_recommended_params": next_params
        }
    
    def get_github_outputs(self, session_iterations: int = 200) -> str:
        """Generate GitHub Actions output format for cumulative training params.
        
        Args:
            session_iterations: Number of iterations for this session.
        
        Returns:
            String formatted for GitHub Actions outputs (KEY=VALUE format).
        """
        params = self.get_next_session_params(session_iterations)
        current = params["current_params"]
        
        lines = [
            f"start_iteration={params['start_iteration']}",
            f"end_iteration={params['end_iteration']}",
            f"session_iterations={session_iterations}",
            f"total_iterations_so_far={params['start_iteration']}",
            f"n_layer={current['n_layer']}",
            f"n_head={current['n_head']}",
            f"n_embd={current['n_embd']}",
            f"learning_rate={current['learning_rate']}",
            f"batch_size={current['batch_size']}",
            f"should_scale_model={str(params['should_scale_model']).lower()}",
            f"total_sessions={params['total_sessions_so_far']}",
        ]
        
        if params['best_val_loss'] is not None:
            lines.append(f"best_val_loss={params['best_val_loss']}")
        
        return '\n'.join(lines)


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Manage cumulative training progress for Deep Tree Echo"
    )
    parser.add_argument(
        "command",
        choices=["get_params", "update", "next_session", "github_outputs", "summary"],
        help="Command to execute"
    )
    parser.add_argument(
        "--progress-file",
        help="Path to cumulative_progress.json",
        default=None
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of iterations completed or to plan"
    )
    parser.add_argument(
        "--val-loss",
        type=float,
        help="Validation loss for update command"
    )
    parser.add_argument(
        "--train-loss",
        type=float,
        help="Training loss for update command"
    )
    parser.add_argument(
        "--workflow",
        default="agent-neuro-train.yml",
        help="Workflow name for session tracking"
    )
    parser.add_argument(
        "--trigger",
        default="unknown",
        help="Training trigger (ci, schedule, manual)"
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        help="Model n_layer parameter"
    )
    parser.add_argument(
        "--n-head",
        type=int,
        help="Model n_head parameter"
    )
    parser.add_argument(
        "--n-embd",
        type=int,
        help="Model n_embd parameter"
    )
    parser.add_argument(
        "--session-iters",
        type=int,
        default=200,
        help="Iterations for next session planning"
    )
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = CumulativeProgressManager(args.progress_file)
    
    if args.command == "get_params":
        params = manager.get_current_params()
        print(json.dumps(params, indent=2))
    
    elif args.command == "next_session":
        params = manager.get_next_session_params(args.session_iters)
        print(json.dumps(params, indent=2))
    
    elif args.command == "github_outputs":
        outputs = manager.get_github_outputs(args.session_iters)
        print(outputs)
    
    elif args.command == "update":
        if args.val_loss is None or args.train_loss is None:
            print("Error: --val-loss and --train-loss required for update", file=sys.stderr)
            sys.exit(1)
        
        model_params = {}
        if args.n_layer:
            model_params["n_layer"] = args.n_layer
        if args.n_head:
            model_params["n_head"] = args.n_head
        if args.n_embd:
            model_params["n_embd"] = args.n_embd
        
        result = manager.update_progress(
            iterations_completed=args.iterations,
            val_loss=args.val_loss,
            train_loss=args.train_loss,
            model_params=model_params if model_params else None,
            workflow=args.workflow,
            trigger=args.trigger
        )
        print(json.dumps(result, indent=2))
    
    elif args.command == "summary":
        summary = {
            "total_iterations": manager.get_total_iterations(),
            "total_sessions": manager.progress_data.get("total_sessions", 0),
            "current_params": manager.get_current_params(),
            "best_metrics": manager.progress_data.get("best_metrics", {}),
            "last_updated": manager.progress_data.get("last_updated", "never")
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
