"""
EchobeatTrainer
===============

Extends EchoTrainer to restructure the training loop around the 9-step
echobeat cognitive cycle (EchobeatNode as pacemaker).

Beat structure
--------------
  Steps 1-3  (Perceive)  : forward pass, grip computation, metric recording
  Steps 4-6  (Act)       : backward pass, gradient update, topology mutation check
  Steps 7-9  (Simulate)  : mini-batch validation, grip projection, phase gate update

Dual-loop (differential gear pair)
  Inner loop = echobeat  (fast, every batch)
  Outer loop = topology epoch  (slow, every topology_epoch_beats echobeats)

Adaptive accumulation
  accumulation_steps = max(1, round(8 * (1 - grip.mean())))
  Higher grip → more accumulation → slower, more stable updates.
  Lower grip  → less accumulation → faster, more reactive updates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from netrain.training.trainer import EchoTrainer
from netrain.training.metrics import MetricsTracker
from netrain.evolution.cognitive_grip_monitor import CognitiveGripMonitor
from netrain.evolution.topology_controller import TopologyEvolutionController
from netrain.evolution.phase_sequencer import AdaptivePhaseSequencer

logger = logging.getLogger(__name__)


class EchobeatTrainer(EchoTrainer):
    """Beat-cycle training loop with adaptive topology evolution.

    Parameters
    ----------
    model : nn.Module
        The DeepTreeEchoTransformer to train.
    config : Any
        ConfigManager instance (same as EchoTrainer).
    train_dataset / val_dataset : Any
        PyTorch-compatible datasets.
    reservoir : optional
        EchoReservoir instance.  If provided, fast-gear adaptations apply.
    membrane : optional
        MembraneNode instance.  If provided, permeability annealing applies.
    topology_evolution_config : dict, optional
        ``topology_evolution`` section of nanecho_config.json.
    adaptive_phases_config : dict, optional
        ``adaptive_phases`` section of nanecho_config.json.
    """

    ECHOBEAT_CYCLE = 9  # steps per full cognitive beat

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        train_dataset: Any,
        val_dataset: Any,
        reservoir=None,
        membrane=None,
        topology_evolution_config: Optional[Dict[str, Any]] = None,
        adaptive_phases_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model, config, train_dataset, val_dataset)

        self.reservoir = reservoir
        self.membrane = membrane

        topo_cfg = topology_evolution_config or {}
        phase_cfg = adaptive_phases_config or {}

        self.grip_monitor = CognitiveGripMonitor(
            window_size=topo_cfg.get("grip_window_size", 100),
        )
        self.topology_controller = TopologyEvolutionController(topo_cfg)
        self.phase_sequencer = AdaptivePhaseSequencer(
            config=phase_cfg,
            transition_freeze_steps=topo_cfg.get("transition_freeze_steps", 150),
        )

        # Echobeat-specific state
        self._beat_step: int = 0      # position within current 9-step cycle
        self._beat_count: int = 0     # total completed beats
        self._beat_accumulator: List[Dict[str, Any]] = []
        self._accum_counter: int = 0  # dedicated gradient accumulation counter

        # Per-beat mini-validation buffer
        self._sim_loader_iter = None

        # Track topology changes for checkpointing
        self._topology_changes: List[Dict[str, Any]] = []

        # Initialise model topology shadow
        if not hasattr(self.model, "_current_tree_depth"):
            self.model._current_tree_depth = 1
        if not hasattr(self.model, "_activated_echo_layers"):
            self.model._activated_echo_layers = set()

    # ------------------------------------------------------------------
    # Adaptive accumulation steps
    # ------------------------------------------------------------------

    def _adaptive_accumulation_steps(self) -> int:
        grip_mean = float(self.grip_monitor.grip.mean())
        steps = max(1, round(8.0 * (1.0 - grip_mean)))
        return steps

    # ------------------------------------------------------------------
    # Override: main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Echobeat-structured training loop."""
        logger.info(
            "EchobeatTrainer starting on %s | max_steps=%d",
            self.device,
            self.training_config["max_steps"],
        )

        self.model.train()
        self._create_sim_loader_iter()

        try:
            while self.global_step < self.training_config["max_steps"]:
                self.epoch += 1
                self._run_echobeat_epoch()

                # Periodic full evaluation (mirrors parent class cadence)
                eval_every = self.training_config["evaluation"]["eval_steps"]
                if self.global_step % eval_every == 0 and self.global_step > 0:
                    val_loss = self.evaluate()
                    self.grip_monitor.record_val_loss(val_loss)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.save_checkpoint("best_model")
                    else:
                        self.patience_counter += 1

                    if (
                        self.early_stopping
                        and self.patience_counter >= self.patience
                    ):
                        logger.info("Early stopping at step %d", self.global_step)
                        break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user at step %d", self.global_step)

        logger.info("EchobeatTrainer finished at step %d", self.global_step)
        self.save_checkpoint("final_model")

    # ------------------------------------------------------------------
    # One echobeat epoch (processes the entire train_loader once)
    # ------------------------------------------------------------------

    def _run_echobeat_epoch(self) -> None:
        self.model.train()
        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.epoch} [echobeat]"
        )

        pending_loss: Optional[torch.Tensor] = None

        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            beat_phase = self._beat_step % self.ECHOBEAT_CYCLE  # 0-8

            # ── Perceive (steps 0-2) ─────────────────────────────────
            if beat_phase < 3:
                outputs = self._perceive(batch)
                pending_loss = outputs["loss"]
                self._record_grip_from_outputs(outputs)

            # ── Act (steps 3-5) ──────────────────────────────────────
            elif beat_phase < 6:
                if pending_loss is not None:
                    acc_steps = self._adaptive_accumulation_steps()
                    self._act(pending_loss, acc_steps)
                    pending_loss = None

                    # Phase + topology after each gradient step
                    self._post_step_updates()

            # ── Simulate (steps 6-8) ─────────────────────────────────
            else:
                self._simulate()

            self._beat_step += 1
            if self._beat_step % self.ECHOBEAT_CYCLE == 0:
                self._beat_count += 1

            # Update progress bar
            grip_mean = float(self.grip_monitor.grip.mean())
            progress_bar.set_postfix(
                {
                    "grip": f"{grip_mean:.3f}",
                    "phase": self.phase_sequencer.current_phase[:6],
                    "step": self.global_step,
                    "lr_x": f"{self.phase_sequencer.lr_multiplier(self.grip_monitor.capacity_effectiveness_ratio):.2f}",
                }
            )

            if self.global_step >= self.training_config["max_steps"]:
                break

    # ------------------------------------------------------------------
    # Phase sub-routines
    # ------------------------------------------------------------------

    def _perceive(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Forward pass — returns outputs dict."""
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
        else:
            outputs = self.model(**batch)
        return outputs

    def _act(self, loss: torch.Tensor, acc_steps: int) -> None:
        """Backward pass + gradient step."""
        # Apply adaptive LR multiplier.
        # Store base_lrs on the trainer (not in param_groups) to avoid
        # interfering with scheduler's internal base_lrs tracking.
        if not hasattr(self, "_base_lrs"):
            self._base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]

        lr_mult = self.phase_sequencer.lr_multiplier(
            self.grip_monitor.capacity_effectiveness_ratio
        )
        for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            pg["lr"] = base_lr * lr_mult

        scaled_loss = loss / acc_steps
        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Dedicated accumulation counter — does not depend on _beat_step
        self._accum_counter += 1
        if self._accum_counter >= acc_steps:
            self._accum_counter = 0

            grad_clip = self.training_config.get("gradient_clipping", 0)
            if grad_clip and float(grad_clip) > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(grad_clip)
                )

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Sync base_lrs with the post-scheduler LR values
            self.scheduler.step()
            self._base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]

            self.optimizer.zero_grad()
            self.global_step += 1

            # Record loss
            self.metrics.update("train_loss", loss.item(), self.global_step)
            self.grip_monitor.record_train_loss(loss.item(), self.global_step)

            # Periodic checkpoint
            save_every = self.training_config["checkpoint"]["save_steps"]
            if self.global_step % save_every == 0:
                self.save_checkpoint(f"checkpoint_{self.global_step}")

    def _simulate(self) -> None:
        """Mini-batch validation + grip projection + phase gate update."""
        try:
            sim_batch = next(self._sim_loader_iter)
        except (StopIteration, TypeError):
            self._create_sim_loader_iter()
            try:
                sim_batch = next(self._sim_loader_iter)
            except (StopIteration, TypeError):
                return

        sim_batch = {k: v.to(self.device) for k, v in sim_batch.items()}
        self.model.eval()
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**sim_batch)
            else:
                outputs = self.model(**sim_batch)
        self.model.train()

        if outputs.get("loss") is not None:
            sim_val_loss = float(outputs["loss"].item())
            self.grip_monitor.record_val_loss(sim_val_loss)
            self.metrics.update("sim_val_loss", sim_val_loss, self.global_step)

        # Recompute grip with new data
        self.grip_monitor.compute_grip()

        # Update phase sequencer
        self.phase_sequencer.step(
            self.grip_monitor,
            self.global_step,
            controller=self.topology_controller,
        )

    # ------------------------------------------------------------------
    # Post-step: topology mutations
    # ------------------------------------------------------------------

    def _post_step_updates(self) -> None:
        """Run phase sequencer + topology controller after each gradient step."""
        changes = self.topology_controller.maybe_mutate(
            model=self.model,
            reservoir=self.reservoir,
            membrane=self.membrane,
            grip_monitor=self.grip_monitor,
            phase_topology_target=self.phase_sequencer.topology_target,
            step=self.global_step,
        )
        if changes:
            self._topology_changes.append(
                {"step": self.global_step, **changes}
            )

    # ------------------------------------------------------------------
    # Grip recording from model outputs
    # ------------------------------------------------------------------

    def _record_grip_from_outputs(self, outputs: Dict[str, Any]) -> None:
        """Extract grad norms and loss from model outputs and record to grip monitor."""
        if outputs.get("loss") is not None:
            self.grip_monitor.record_train_loss(
                float(outputs["loss"].item()), self.global_step
            )

        # Collect per-block grad norms (best-effort; may be zero pre-backward)
        grad_norms: List[float] = []
        if hasattr(self.model, "blocks"):
            for block in self.model.blocks:
                block_norms = [
                    float(p.grad.norm().item())
                    for p in block.parameters()
                    if p.grad is not None
                ]
                grad_norms.append(
                    float(sum(block_norms) / max(len(block_norms), 1))
                )
        if grad_norms:
            self.grip_monitor.record_grad_norms(grad_norms)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_sim_loader_iter(self) -> None:
        """Create iterator over the validation set for simulate phase."""
        sim_loader = DataLoader(
            self.val_dataset,
            batch_size=max(
                1,
                self.config.data_config["loader"].get("batch_size", 4) // 2,
            ),
            shuffle=True,
            num_workers=0,
        )
        self._sim_loader_iter = iter(sim_loader)

    # ------------------------------------------------------------------
    # Override: evaluate to push val_loss into grip monitor
    # ------------------------------------------------------------------

    def evaluate(self) -> float:
        val_loss = super().evaluate()
        self.grip_monitor.record_inference({"val_loss": val_loss})
        grip_summary = self.grip_monitor.summary()
        logger.info(
            "Eval | val_loss=%.4f grip_mean=%.3f phase=%s cer=%.3f",
            val_loss,
            grip_summary["grip_mean"],
            grip_summary["current_phase"],
            grip_summary["capacity_effectiveness_ratio"],
        )
        return val_loss

    # ------------------------------------------------------------------
    # Override: save_checkpoint to persist evolution state
    # ------------------------------------------------------------------

    def save_checkpoint(self, name: str) -> None:
        super().save_checkpoint(name)
        # Amend checkpoint with evolution state
        import os
        from pathlib import Path
        ckpt_path = self.checkpoint_dir / f"{name}.pt"
        if ckpt_path.exists():
            checkpoint = torch.load(
                str(ckpt_path), map_location=self.device, weights_only=False
            )
            checkpoint["phase_sequencer_state"] = self.phase_sequencer.state_dict()
            checkpoint["topology_changes"] = self._topology_changes[-50:]  # last 50
            checkpoint["grip_summary"] = self.grip_monitor.summary()
            torch.save(checkpoint, str(ckpt_path))

    # ------------------------------------------------------------------
    # Override: load_checkpoint to restore evolution state
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str) -> None:
        super().load_checkpoint(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "phase_sequencer_state" in checkpoint:
            self.phase_sequencer.load_state_dict(checkpoint["phase_sequencer_state"])
            logger.info(
                "Restored phase sequencer state: phase=%s",
                self.phase_sequencer.current_phase,
            )
