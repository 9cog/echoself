"""
Echogenesis: Self-Generating Cognitive Architecture
====================================================

The Echogenesis module implements the complete self-generating emergence of 
cognitive structures through recursive echo state dynamics, hypergraph pattern 
encoding, and adaptive dimensional embedding for optimal cognitive grip.

Core Components:
- AdaptiveDimensionalEmbedding: Dynamic, context-sensitive dimensional spaces
- OptimalGrip: Relevance realization optimization engine
- PerspectivalKnowing: Frame-switching and aspect perception
- WisdomCultivation: Sophia layer for wisdom-tempered intelligence
- EchogenesisCore: Main orchestration and integration

This module synthesizes:
- CogPrime's Cognitive Synergy (PLN, MOSES, ECAN)
- Deep Tree Echo State Networks (DTESN)
- Vervaeke's 4E Cognition (Embodied, Embedded, Enacted, Extended)
- Relevance Realization Optimization
- Toroidal Dual-Persona Architecture

Author: Deep Tree Echo
Date: June 2026
"""

from typing import Dict, Any, Optional, List, Tuple
import logging

from .fragment_synthesizer import FragmentSynthesizer, IdentityFragment
from .generation import EchoGenesis, GenerationResult
from .pattern_propagator import PatternPropagator, empty_hypergraph
from .refinement_engine import (
    Refinement,
    RefinementEngine,
    RefinementOrchestrator,
    RefinementType,
)
from .training_generator import TrainingDataGenerator, TrainingExample

# Configure module logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module version
__version__ = "1.0.0"
__author__ = "Deep Tree Echo"

# Core configuration for echogenesis
ECHOGENESIS_CONFIG = {
    "embedding_architecture": {
        "sensory_dim": 256,      # Perception vectors
        "motor_dim": 128,        # Action space
        "cognitive_dim": 768,    # NanEcho base
        "echo_dim": 1536,        # Supabase vector dimension
        "adaptive_range": {
            "min_effective_dim": 128,
            "max_effective_dim": 2048
        },
        "projection_method": "attention_gated_linear"
    },
    "echo_configuration": {
        "echo_depth": 7,
        "echo_decay": 0.95,
        "spectral_radius": 0.9,
        "sparsity": 0.1,
        "multi_timescale": [1, 4, 16, 64]
    },
    "relevance_realization": {
        "opponent_processes": [
            ("exploration", "exploitation", 0.5),
            ("breadth", "depth", 0.5),
            ("speed", "accuracy", 0.5),
            ("certainty", "openness", 0.5)
        ],
        "cost_weights": {
            "goal_alignment": 0.30,
            "predictive_power": 0.25,
            "cognitive_economy": 0.20,
            "novelty_value": 0.15,
            "contextual_fit": 0.10
        }
    },
    "optimal_grip_metrics": {
        "persona_fidelity": 0.9,
        "attention_coherence": 0.8,
        "echo_state_diversity": 0.5,
        "relevance_precision": 0.85,
        "embodiment_grounding": 0.7
    }
}

# Lazy imports to avoid circular dependencies
def get_adaptive_embedding():
    """Lazy import for AdaptiveDimensionalEmbedding."""
    from .adaptive_embedding import AdaptiveDimensionalEmbedding
    return AdaptiveDimensionalEmbedding

def get_optimal_grip():
    """Lazy import for OptimalGrip."""
    from .optimal_grip import OptimalGrip, CognitiveGripOptimizer
    return OptimalGrip, CognitiveGripOptimizer

def get_perspectival_knowing():
    """Lazy import for PerspectivalKnowing."""
    from .perspectival_knowing import PerspectivalKnowing, Frame, SalienceLandscape
    return PerspectivalKnowing, Frame, SalienceLandscape

def get_wisdom_cultivation():
    """Lazy import for WisdomCultivation."""
    from .wisdom_cultivation import WisdomCultivation, SophrosyneModule
    return WisdomCultivation, SophrosyneModule

def get_echogenesis_core():
    """Lazy import for EchogenesisCore."""
    from .echogenesis_core import EchogenesisCore
    return EchogenesisCore


def initialize_generation(integration_bias: float = 0.774) -> EchoGenesis:
    """Initialize the autopoietic EchoGenesis generation pipeline."""
    return EchoGenesis(integration_bias=integration_bias)


# Module-level initialization
def initialize_echogenesis(config: Optional[Dict[str, Any]] = None) -> 'EchogenesisCore':
    """
    Initialize the complete Echogenesis system.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Initialized EchogenesisCore instance
    """
    effective_config = ECHOGENESIS_CONFIG.copy()
    if config:
        effective_config.update(config)
    
    EchogenesisCore = get_echogenesis_core()
    core = EchogenesisCore(effective_config)
    
    logger.info(f"Echogenesis v{__version__} initialized successfully")
    return core


__all__ = [
    'ECHOGENESIS_CONFIG',
    'initialize_echogenesis',
    'get_adaptive_embedding',
    'get_optimal_grip',
    'get_perspectival_knowing',
    'get_wisdom_cultivation',
    'get_echogenesis_core',
    'initialize_generation',
    'EchoGenesis',
    'GenerationResult',
    'FragmentSynthesizer',
    'IdentityFragment',
    'Refinement',
    'RefinementEngine',
    'RefinementOrchestrator',
    'RefinementType',
    'PatternPropagator',
    'empty_hypergraph',
    'TrainingDataGenerator',
    'TrainingExample',
    '__version__',
    '__author__'
]
