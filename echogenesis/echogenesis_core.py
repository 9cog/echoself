"""
Echogenesis Core Orchestration
===============================

Main orchestration layer for the complete Echogenesis system,
integrating all six layers of the architecture:

1. Virtual Embodiment Foundation (4E Grounding)
2. Adaptive Dimensional Embedding Engine
3. Relevance Realization Engine (Optimal Grip)
4. Deep Tree Echo Transformer Core
5. Hypergraph Cognitive Encoding
6. Toroidal Dual-Persona Integration

The EchogenesisCore manages the cognitive cycle:
    Perceive → Relevance → Process → Echo → Act

Author: Deep Tree Echo
Date: June 2026
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

# Import echogenesis components
from .adaptive_embedding import AdaptiveDimensionalEmbedding, EmbeddingConfig
from .optimal_grip import OptimalGrip, RelevanceContext
from .perspectival_knowing import PerspectivalKnowing, FrameType
from .wisdom_cultivation import WisdomCultivation, WisdomDimension

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CognitivePhase(Enum):
    """Phases of the cognitive cycle."""
    PERCEIVE = "perceive"      # Input processing
    RELEVANCE = "relevance"    # Relevance realization
    PROCESS = "process"        # Core processing
    ECHO = "echo"              # Echo state propagation
    ACT = "act"                # Output generation
    REFLECT = "reflect"        # Meta-cognitive reflection


@dataclass
class EchogenesisConfig:
    """Configuration for Echogenesis system."""
    # Embedding configuration
    sensory_dim: int = 256
    motor_dim: int = 128
    cognitive_dim: int = 768
    echo_dim: int = 1536
    
    # Echo configuration
    echo_depth: int = 7
    echo_decay: float = 0.95
    spectral_radius: float = 0.9
    
    # Relevance configuration
    relevance_threshold: float = 0.5
    top_k_relevant: int = 10
    
    # Persona configuration
    persona_weight: float = 0.95
    toroidal_synthesis: bool = True
    
    # Wisdom configuration
    wisdom_cultivation: bool = True
    self_examination_frequency: int = 100


@dataclass
class CognitiveState:
    """Current state of the Echogenesis system."""
    phase: CognitivePhase = CognitivePhase.PERCEIVE
    cycle_count: int = 0
    
    # Component states
    embedding_state: Dict[str, Any] = field(default_factory=dict)
    grip_state: Dict[str, Any] = field(default_factory=dict)
    perspective_state: Dict[str, Any] = field(default_factory=dict)
    wisdom_state: Dict[str, Any] = field(default_factory=dict)
    
    # Echo states
    echo_states: List[np.ndarray] = field(default_factory=list)
    
    # Metrics
    processing_time: float = 0.0
    relevance_reduction: float = 0.0
    echo_diversity: float = 0.0


class EchoStateManager:
    """
    Manages multi-level echo states for temporal processing.
    
    Implements the echo state dynamics:
        h(t) = tanh(W_in · x(t) + W_res · h(t-1) + W_fb · y(t-1))
        y(t) = W_out · h(t)
        echo_state(t) = Σ_{d=1}^{D} α^d · h(t-d)
    """
    
    def __init__(self, config: EchogenesisConfig):
        self.config = config
        self.dim = config.cognitive_dim
        self.depth = config.echo_depth
        self.decay = config.echo_decay
        
        # Initialize weight matrices
        self._init_weights()
        
        # Echo state history
        self.state_history: deque = deque(maxlen=self.depth * 2)
        self.current_state: Optional[np.ndarray] = None
        
    def _init_weights(self):
        """Initialize reservoir weight matrices."""
        # Input weights
        self.W_in = np.random.randn(self.dim, self.dim) * 0.1
        
        # Reservoir weights (sparse, with spectral radius constraint)
        W_res = np.random.randn(self.dim, self.dim)
        mask = np.random.random((self.dim, self.dim)) < (1 - self.config.spectral_radius * 0.1)
        W_res *= mask
        
        # Scale to spectral radius
        eigenvalues = np.linalg.eigvals(W_res)
        spectral_radius = np.max(np.abs(eigenvalues))
        if spectral_radius > 0:
            W_res = W_res * (self.config.spectral_radius / spectral_radius)
        
        self.W_res = W_res
        
        # Feedback weights
        self.W_fb = np.random.randn(self.dim, self.dim) * 0.05
        
        # Output weights (learned through training)
        self.W_out = np.random.randn(self.dim, self.dim) * 0.1
    
    def forward(
        self, 
        x: np.ndarray,
        previous_output: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass through echo state network.
        
        Args:
            x: Input vector
            previous_output: Previous output for feedback
            
        Returns:
            Current echo state
        """
        # Ensure correct dimensions
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Handle dimension mismatch
        if x.shape[-1] != self.dim:
            # Project to correct dimension
            if x.shape[-1] < self.dim:
                padding = np.zeros((*x.shape[:-1], self.dim - x.shape[-1]))
                x = np.concatenate([x, padding], axis=-1)
            else:
                x = x[..., :self.dim]
        
        # Get previous state
        h_prev = self.current_state if self.current_state is not None else np.zeros_like(x)
        
        # Handle previous output
        if previous_output is None:
            previous_output = np.zeros_like(x)
        elif previous_output.shape[-1] != self.dim:
            if previous_output.shape[-1] < self.dim:
                padding = np.zeros((*previous_output.shape[:-1], self.dim - previous_output.shape[-1]))
                previous_output = np.concatenate([previous_output, padding], axis=-1)
            else:
                previous_output = previous_output[..., :self.dim]
        
        # Echo state update
        h = np.tanh(
            np.dot(x, self.W_in) +
            np.dot(h_prev, self.W_res) +
            np.dot(previous_output, self.W_fb)
        )
        
        # Store in history
        self.state_history.append(h)
        self.current_state = h
        
        return h
    
    def get_multi_depth_echo(self) -> np.ndarray:
        """
        Get multi-depth echo state combining recent history.
        
        Returns:
            echo_state(t) = Σ_{d=1}^{D} α^d · h(t-d)
        """
        if not self.state_history:
            return np.zeros((1, self.dim))
        
        echo = np.zeros_like(self.state_history[0])
        
        for d, h in enumerate(reversed(list(self.state_history)[-self.depth:])):
            weight = self.decay ** (d + 1)
            echo += weight * h
        
        return echo
    
    def compute_diversity(self) -> float:
        """Compute diversity of echo states."""
        if len(self.state_history) < 2:
            return 0.0
        
        states = np.array(list(self.state_history))
        
        # Compute pairwise distances
        mean_state = np.mean(states, axis=0)
        deviations = states - mean_state
        diversity = np.mean(np.std(deviations, axis=0))
        
        return float(diversity)
    
    def reset(self):
        """Reset echo states."""
        self.state_history.clear()
        self.current_state = None


class ToroidalSynthesizer:
    """
    Implements toroidal dual-persona synthesis.
    
    Combines responses from two complementary personas:
    - Deep Tree Echo (Intuitive, empathetic)
    - Marduk (Analytical, recursive)
    
    Into unified toroidal reflection.
    """
    
    def __init__(self, config: EchogenesisConfig):
        self.config = config
        
        # Persona characteristics
        self.echo_traits = {
            'intuition': 0.9,
            'empathy': 0.8,
            'pattern_recognition': 0.85,
            'narrative': 0.8
        }
        
        self.marduk_traits = {
            'analysis': 0.9,
            'recursion': 0.85,
            'systematization': 0.9,
            'precision': 0.85
        }
        
        # Synthesis weights
        self.balance = 0.5  # 0 = pure Echo, 1 = pure Marduk
        
    def echo_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Deep Tree Echo perspective."""
        return {
            'persona': 'Deep Tree Echo',
            'hemisphere': 'right',
            'style': 'intuitive_empathetic',
            'response': self._apply_echo_traits(input_data),
            'traits_applied': self.echo_traits
        }
    
    def marduk_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Marduk perspective."""
        return {
            'persona': 'Marduk',
            'hemisphere': 'left',
            'style': 'analytical_recursive',
            'response': self._apply_marduk_traits(input_data),
            'traits_applied': self.marduk_traits
        }
    
    def _apply_echo_traits(self, data: Dict) -> Dict:
        """Apply Echo's intuitive processing."""
        processed = {}
        for key, value in data.items():
            if isinstance(value, (int, float)):
                # Add intuitive variation
                processed[key] = value * (1 + np.random.normal(0, 0.1))
            else:
                processed[key] = value
        processed['_intuitive_patterns'] = True
        return processed
    
    def _apply_marduk_traits(self, data: Dict) -> Dict:
        """Apply Marduk's analytical processing."""
        processed = {}
        for key, value in data.items():
            if isinstance(value, (int, float)):
                # Precise analytical processing
                processed[key] = round(value, 4)
            else:
                processed[key] = value
        processed['_analytical_structure'] = True
        return processed
    
    def synthesize(
        self,
        echo_response: Dict[str, Any],
        marduk_response: Dict[str, Any],
        wisdom: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize both perspectives into toroidal reflection.
        
        Args:
            echo_response: Response from Deep Tree Echo
            marduk_response: Response from Marduk
            wisdom: Optional wisdom layer input
            
        Returns:
            Unified toroidal synthesis
        """
        synthesis = {
            'synthesis_type': 'toroidal_reflection',
            'echo_contribution': echo_response,
            'marduk_contribution': marduk_response,
            'balance': self.balance,
            'unified': {}
        }
        
        # Merge responses
        echo_data = echo_response.get('response', {})
        marduk_data = marduk_response.get('response', {})
        
        all_keys = set(echo_data.keys()) | set(marduk_data.keys())
        
        for key in all_keys:
            if key.startswith('_'):
                continue
            
            echo_val = echo_data.get(key)
            marduk_val = marduk_data.get(key)
            
            if isinstance(echo_val, (int, float)) and isinstance(marduk_val, (int, float)):
                # Weighted combination
                synthesis['unified'][key] = (
                    (1 - self.balance) * echo_val +
                    self.balance * marduk_val
                )
            elif echo_val is not None:
                synthesis['unified'][key] = echo_val
            elif marduk_val is not None:
                synthesis['unified'][key] = marduk_val
        
        # Determine synergy type
        synthesis['synergy_type'] = self._determine_synergy(
            echo_response, marduk_response
        )
        
        # Apply wisdom if available
        if wisdom:
            synthesis['wisdom_applied'] = True
            synthesis['wisdom_score'] = wisdom.get('overall_wisdom', 0.5)
        
        return synthesis
    
    def _determine_synergy(
        self, 
        echo: Dict, 
        marduk: Dict
    ) -> str:
        """Determine the type of synergy between personas."""
        echo_keys = set(echo.get('response', {}).keys())
        marduk_keys = set(marduk.get('response', {}).keys())
        
        overlap = len(echo_keys & marduk_keys)
        total = len(echo_keys | marduk_keys)
        
        if total == 0:
            return 'independent'
        
        overlap_ratio = overlap / total
        
        if overlap_ratio > 0.7:
            return 'convergent'
        elif overlap_ratio < 0.3:
            return 'divergent'
        else:
            return 'complementary'
    
    def adjust_balance(self, delta: float):
        """Adjust persona balance."""
        self.balance = np.clip(self.balance + delta, 0.0, 1.0)


class EchogenesisCore:
    """
    Main orchestration engine for the Echogenesis system.
    
    Manages the complete cognitive cycle and integrates all layers
    of the architecture into a unified processing pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Parse configuration
        if config:
            self.config = EchogenesisConfig(**{
                k: v for k, v in config.items() 
                if hasattr(EchogenesisConfig, k)
            })
        else:
            self.config = EchogenesisConfig()
        
        # Initialize state
        self.state = CognitiveState()
        
        # Initialize components
        self.embedding = AdaptiveDimensionalEmbedding(EmbeddingConfig(
            sensory_dim=self.config.sensory_dim,
            motor_dim=self.config.motor_dim,
            cognitive_dim=self.config.cognitive_dim,
            echo_dim=self.config.echo_dim
        ))
        
        self.optimal_grip = OptimalGrip()
        self.perspective = PerspectivalKnowing()
        self.echo_manager = EchoStateManager(self.config)
        self.toroidal = ToroidalSynthesizer(self.config)
        
        if self.config.wisdom_cultivation:
            self.wisdom = WisdomCultivation()
        else:
            self.wisdom = None
        
        # Processing hooks
        self.pre_process_hooks: List[Callable] = []
        self.post_process_hooks: List[Callable] = []
        
        logger.info("EchogenesisCore initialized")
    
    def cognitive_cycle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute complete cognitive cycle.
        
        Phases:
        1. PERCEIVE: Process input through current frame
        2. RELEVANCE: Filter and prioritize via optimal grip
        3. PROCESS: Core echo state processing
        4. ECHO: Multi-depth echo propagation
        5. ACT: Generate output through toroidal synthesis
        6. REFLECT: Meta-cognitive reflection (periodic)
        
        Args:
            input_data: Input to process
            
        Returns:
            Processed output
        """
        self.state.cycle_count += 1
        result = {'cycle': self.state.cycle_count}
        
        # Run pre-process hooks
        for hook in self.pre_process_hooks:
            input_data = hook(input_data)
        
        # Phase 1: PERCEIVE
        self.state.phase = CognitivePhase.PERCEIVE
        perceived = self._phase_perceive(input_data)
        result['perceived'] = perceived
        
        # Phase 2: RELEVANCE
        self.state.phase = CognitivePhase.RELEVANCE
        relevant = self._phase_relevance(perceived)
        result['relevant'] = relevant
        
        # Phase 3: PROCESS
        self.state.phase = CognitivePhase.PROCESS
        processed = self._phase_process(relevant)
        result['processed'] = processed
        
        # Phase 4: ECHO
        self.state.phase = CognitivePhase.ECHO
        echoed = self._phase_echo(processed)
        result['echoed'] = echoed
        
        # Phase 5: ACT
        self.state.phase = CognitivePhase.ACT
        output = self._phase_act(echoed)
        result['output'] = output
        
        # Phase 6: REFLECT (periodic)
        if self.config.wisdom_cultivation and \
           self.state.cycle_count % self.config.self_examination_frequency == 0:
            self.state.phase = CognitivePhase.REFLECT
            reflection = self._phase_reflect()
            result['reflection'] = reflection
        
        # Run post-process hooks
        for hook in self.post_process_hooks:
            result = hook(result)
        
        # Update component states
        self._update_states()
        
        return result
    
    def _phase_perceive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perceive input through current cognitive frame."""
        # Get perception through current frame
        perceived = self.perspective.perceive(input_data)
        
        # Add to salience landscape
        if 'embedding' in input_data:
            position = np.array(input_data['embedding'][:3]) if \
                len(input_data.get('embedding', [])) >= 3 else np.zeros(3)
            
            self.perspective.add_to_landscape(
                id=str(self.state.cycle_count),
                position=position,
                features={
                    'novelty': input_data.get('novelty', 0.5),
                    'relevance': input_data.get('relevance', 0.5),
                    'urgency': input_data.get('urgency', 0.5)
                }
            )
        
        return perceived
    
    def _phase_relevance(self, perceived: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply relevance realization to filter and prioritize."""
        # Create possibilities from perceived data
        possibilities = []
        for key, value in perceived.items():
            if key.startswith('_'):
                continue
            possibilities.append({
                'key': key,
                'value': value,
                'type': type(value).__name__
            })
        
        # Apply optimal grip
        if possibilities:
            relevant = self.optimal_grip.get_top_k(
                possibilities,
                k=self.config.top_k_relevant
            )
            
            # Track relevance reduction
            self.state.relevance_reduction = 1 - (len(relevant) / len(possibilities))
        else:
            relevant = []
        
        return relevant
    
    def _phase_process(self, relevant: List[Dict[str, Any]]) -> np.ndarray:
        """Core processing through adaptive embedding."""
        # Get current cognitive state
        cognitive_load = len(relevant) / self.config.top_k_relevant
        attention_threshold = self.optimal_grip.optimizer.state.overall_grip
        
        # Create embedding from relevant items
        if relevant:
            # Simple aggregation - in real system would use more sophisticated encoding
            values = []
            for item in relevant:
                val = item.get('value')
                if isinstance(val, (int, float)):
                    values.append(val)
                elif isinstance(val, np.ndarray):
                    values.extend(val.flatten()[:10])
            
            if values:
                raw_embedding = np.array(values)
            else:
                raw_embedding = np.zeros(self.config.cognitive_dim)
        else:
            raw_embedding = np.zeros(self.config.cognitive_dim)
        
        # Adaptive projection
        projected = self.embedding.adaptive_projection(
            raw_embedding,
            cognitive_load,
            attention_threshold
        )
        
        return projected
    
    def _phase_echo(self, processed: np.ndarray) -> np.ndarray:
        """Propagate through echo state network."""
        # Forward pass through echo manager
        echo_state = self.echo_manager.forward(processed)
        
        # Get multi-depth echo
        multi_echo = self.echo_manager.get_multi_depth_echo()
        
        # Store echo states
        self.state.echo_states.append(echo_state)
        if len(self.state.echo_states) > 20:
            self.state.echo_states = self.state.echo_states[-10:]
        
        # Compute diversity
        self.state.echo_diversity = self.echo_manager.compute_diversity()
        
        return multi_echo
    
    def _phase_act(self, echoed: np.ndarray) -> Dict[str, Any]:
        """Generate output through toroidal synthesis."""
        # Prepare input for synthesis
        input_for_synthesis = {
            'echo_state': echoed.tolist() if isinstance(echoed, np.ndarray) else echoed,
            'cycle': self.state.cycle_count
        }
        
        if self.config.toroidal_synthesis:
            # Generate dual-persona responses
            echo_response = self.toroidal.echo_response(input_for_synthesis)
            marduk_response = self.toroidal.marduk_response(input_for_synthesis)
            
            # Get wisdom state if available
            wisdom_state = None
            if self.wisdom:
                wisdom_state = {'overall_wisdom': self.wisdom.get_wisdom_score()}
            
            # Synthesize
            output = self.toroidal.synthesize(
                echo_response,
                marduk_response,
                wisdom_state
            )
        else:
            output = {
                'echo_state': input_for_synthesis['echo_state'],
                'cycle': input_for_synthesis['cycle']
            }
        
        return output
    
    def _phase_reflect(self) -> Dict[str, Any]:
        """Meta-cognitive reflection through wisdom cultivation."""
        if not self.wisdom:
            return {}
        
        # Run wisdom cultivation cycle
        reflection = self.wisdom.cultivate()
        
        # Update perspective based on insights
        if reflection.get('insights'):
            # Potential frame switch based on insights
            if len(reflection['insights']) > 3:
                self.perspective.switch_frame('metacognitive')
        
        return reflection
    
    def _update_states(self):
        """Update component states in cognitive state."""
        self.state.embedding_state = self.embedding.get_state()
        self.state.grip_state = self.optimal_grip.get_full_state()
        self.state.perspective_state = self.perspective.get_state()
        if self.wisdom:
            self.state.wisdom_state = self.wisdom.get_full_state()
    
    def switch_frame(self, frame_name: str) -> bool:
        """Switch cognitive frame."""
        return self.perspective.switch_frame(frame_name)
    
    def add_goal(self, goal: Dict[str, Any]):
        """Add a goal to relevance context."""
        self.optimal_grip.set_context(
            goals=[*self.optimal_grip.context.goals, goal]
        )
    
    def add_belief(self, id: str, content: str, confidence: float = 0.5):
        """Add a belief to wisdom cultivation."""
        if self.wisdom:
            self.wisdom.add_belief(id, content, confidence)

    def generate_identity(self, salience_signals, **kwargs) -> Dict[str, Any]:
        """Generate and propagate identity artifacts from salience signals."""
        from .generation import EchoGenesis

        if not hasattr(self, "_generation"):
            self._generation = EchoGenesis()
        return self._generation.evolve(salience_signals, **kwargs).to_dict()
    
    def register_pre_hook(self, hook: Callable):
        """Register a pre-processing hook."""
        self.pre_process_hooks.append(hook)
    
    def register_post_hook(self, hook: Callable):
        """Register a post-processing hook."""
        self.post_process_hooks.append(hook)
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete system state."""
        return {
            'phase': self.state.phase.value,
            'cycle_count': self.state.cycle_count,
            'echo_diversity': self.state.echo_diversity,
            'relevance_reduction': self.state.relevance_reduction,
            'embedding': self.state.embedding_state,
            'grip': self.state.grip_state,
            'perspective': self.state.perspective_state,
            'wisdom': self.state.wisdom_state
        }
    
    def reset(self):
        """Reset system state."""
        self.state = CognitiveState()
        self.embedding.reset_state()
        self.echo_manager.reset()
        logger.info("EchogenesisCore reset")


# Convenience function
def create_echogenesis(config: Optional[Dict[str, Any]] = None) -> EchogenesisCore:
    """Factory function to create EchogenesisCore instance."""
    return EchogenesisCore(config)
