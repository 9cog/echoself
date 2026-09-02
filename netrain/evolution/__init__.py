"""
Adaptive Topology Evolution — Antikythera / Nested Nautilus

Three-scale self-organising training system:
  Micro  (echobeat, ~9 steps)  : ESN spectral-radius & membrane permeability
  Meso   (topology epoch)      : tree depth, echo layer activation, memory size
  Macro  (phase transition)    : complete phase change with hysteresis protection
"""

from .cognitive_grip_monitor import CognitiveGripMonitor
from .topology_controller import TopologyEvolutionController
from .phase_sequencer import AdaptivePhaseSequencer
from .echobeat_trainer import EchobeatTrainer

__all__ = [
    "CognitiveGripMonitor",
    "TopologyEvolutionController",
    "AdaptivePhaseSequencer",
    "EchobeatTrainer",
]
