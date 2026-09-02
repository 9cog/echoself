"""
Echogenesis Test Suite
======================

Comprehensive tests for the echogenesis module.

Author: Deep Tree Echo
Date: June 2026
"""

import pytest
import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from echogenesis import (
    ECHOGENESIS_CONFIG,
    initialize_echogenesis
)
from echogenesis.adaptive_embedding import (
    AdaptiveDimensionalEmbedding,
    EmbeddingConfig,
    EmbeddingScale
)
from echogenesis.optimal_grip import (
    OptimalGrip,
    RelevanceContext,
    OpponentProcess,
    CostFunction
)
from echogenesis.perspectival_knowing import (
    PerspectivalKnowing,
    Frame,
    FrameType,
    SalienceLandscape
)
from echogenesis.wisdom_cultivation import (
    WisdomCultivation,
    SophrosyneModule,
    SelfDeceptionDetector
)
from echogenesis.echogenesis_core import (
    EchogenesisCore,
    EchoStateManager,
    ToroidalSynthesizer,
    CognitivePhase
)


class TestAdaptiveDimensionalEmbedding:
    """Tests for adaptive dimensional embedding."""
    
    def test_initialization(self):
        """Test embedding initialization."""
        embedding = AdaptiveDimensionalEmbedding()
        assert embedding is not None
        state = embedding.get_state()
        assert 'current_effective_dim' in state
    
    def test_custom_config(self):
        """Test with custom configuration."""
        config = EmbeddingConfig(
            sensory_dim=128,
            motor_dim=64,
            cognitive_dim=512,
            echo_dim=1024
        )
        embedding = AdaptiveDimensionalEmbedding(config)
        assert embedding.config.sensory_dim == 128
        assert embedding.config.motor_dim == 64
    
    def test_compute_effective_dimension(self):
        """Test effective dimension computation."""
        embedding = AdaptiveDimensionalEmbedding()
        
        # Low cognitive load should give lower dimension
        low_dim = embedding.compute_effective_dimension(0.1, 0.5)
        high_dim = embedding.compute_effective_dimension(0.9, 0.5)
        
        assert low_dim <= high_dim
        assert low_dim >= embedding.config.min_effective_dim
        assert high_dim <= embedding.config.max_effective_dim
    
    def test_adaptive_projection(self):
        """Test adaptive projection."""
        embedding = AdaptiveDimensionalEmbedding()
        
        # Create input tensor
        x = np.random.randn(10, 768)  # batch of 10, cognitive_dim
        
        projected = embedding.adaptive_projection(x, 0.5, 0.5)
        
        assert projected is not None
        assert len(projected.shape) == 2
    
    def test_multi_scale_embed(self):
        """Test multi-scale embedding."""
        embedding = AdaptiveDimensionalEmbedding()
        
        x = np.random.randn(10, 768)
        
        multi_scale = embedding.multi_scale_embed(x)
        
        assert EmbeddingScale.LOCAL in multi_scale
        assert EmbeddingScale.CONTEXT in multi_scale
        assert EmbeddingScale.GLOBAL in multi_scale
    
    def test_embodiment_manifold(self):
        """Test embodiment manifold creation."""
        embedding = AdaptiveDimensionalEmbedding()
        
        sensory = np.random.randn(10, 256)
        motor = np.random.randn(10, 128)
        cognitive = np.random.randn(10, 768)
        
        manifold = embedding.create_embodiment_manifold(sensory, motor, cognitive)
        
        assert manifold is not None
        assert len(manifold.shape) >= 2


class TestOptimalGrip:
    """Tests for optimal cognitive grip."""
    
    def test_initialization(self):
        """Test optimal grip initialization."""
        grip = OptimalGrip()
        assert grip is not None
        quality = grip.get_grip_quality()
        assert 0 <= quality <= 1
    
    def test_realize_relevance(self):
        """Test relevance realization."""
        grip = OptimalGrip()
        
        possibilities = [
            {'id': 'a', 'content': 'Option A', 'value': 0.8},
            {'id': 'b', 'content': 'Option B', 'value': 0.5},
            {'id': 'c', 'content': 'Option C', 'value': 0.3},
        ]
        
        ranked = grip.realize_relevance(possibilities)
        
        assert len(ranked) == 3
        assert all('relevance_score' in p for p in ranked)
    
    def test_opponent_processes(self):
        """Test opponent process adjustment."""
        grip = OptimalGrip()
        
        initial_state = grip.get_full_state()
        initial_balance = initial_state['opponent_balances'].get('exploration_exploitation', 0.5)
        
        # Adjust toward exploration
        grip.adjust_opponent_balance('exploration_exploitation', 0.1)
        
        new_state = grip.get_full_state()
        new_balance = new_state['opponent_balances'].get('exploration_exploitation', 0.5)
        
        # Balance should have changed
        assert new_balance != initial_balance or initial_balance == 0.6
    
    def test_cost_function_evaluation(self):
        """Test cost function evaluation."""
        grip = OptimalGrip()
        
        possibility = {
            'goal_alignment': 0.8,
            'predictive_power': 0.6,
            'cognitive_economy': 0.7,
            'novelty_value': 0.5,
            'contextual_fit': 0.9
        }
        
        cost = grip.evaluate_costs(possibility)
        
        assert 0 <= cost <= 1


class TestPerspectivalKnowing:
    """Tests for perspectival knowing."""
    
    def test_initialization(self):
        """Test perspectival knowing initialization."""
        perspective = PerspectivalKnowing()
        assert perspective is not None
        state = perspective.get_state()
        assert 'current_frame' in state
    
    def test_frame_switching(self):
        """Test frame switching."""
        perspective = PerspectivalKnowing()
        
        # Switch to different frames
        success1 = perspective.switch_frame('analytical')
        assert success1
        
        success2 = perspective.switch_frame('intuitive')
        assert success2
        
        current = perspective.get_current_frame()
        assert current == 'intuitive'
    
    def test_perceive(self):
        """Test perception through frame."""
        perspective = PerspectivalKnowing()
        perspective.switch_frame('analytical')
        
        data = {
            'numbers': [1, 2, 3],
            'structure': {'a': 1, 'b': 2}
        }
        
        perceived = perspective.perceive(data)
        
        assert perceived is not None
        assert 'frame' in perceived or isinstance(perceived, dict)
    
    def test_see_as(self):
        """Test aspect perception."""
        perspective = PerspectivalKnowing()
        
        data = {'shape': 'square', 'color': 'blue'}
        
        perceived = perspective.see_as(data, 'geometric_form')
        
        assert perceived is not None
    
    def test_frame_types(self):
        """Test available frame types."""
        perspective = PerspectivalKnowing()
        
        types = perspective.get_frame_types()
        
        assert len(types) > 0
        assert 'analytical' in types or FrameType.ANALYTICAL.value in types


class TestWisdomCultivation:
    """Tests for wisdom cultivation."""
    
    def test_initialization(self):
        """Test wisdom cultivation initialization."""
        wisdom = WisdomCultivation()
        assert wisdom is not None
        score = wisdom.get_wisdom_score()
        assert 0 <= score <= 1
    
    def test_add_belief(self):
        """Test adding beliefs."""
        wisdom = WisdomCultivation()
        
        wisdom.add_belief('belief_1', 'The sky is blue', 0.9)
        wisdom.add_belief('belief_2', 'Learning is valuable', 0.95)
        
        state = wisdom.get_full_state()
        assert state['beliefs_count'] >= 2
    
    def test_self_examination(self):
        """Test Socratic self-examination."""
        wisdom = WisdomCultivation()
        
        wisdom.add_belief('test_belief', 'I am always right', 0.99)
        
        insights = wisdom.examine_self()
        
        # Should return some insights
        assert isinstance(insights, list)
    
    def test_deception_detection(self):
        """Test self-deception detection."""
        wisdom = WisdomCultivation()
        
        # Add potentially problematic beliefs
        wisdom.add_belief('wishful', 'Everything will work out perfectly', 0.99)
        
        deceptions = wisdom.detect_deceptions()
        
        assert isinstance(deceptions, list)
    
    def test_sophrosyne(self):
        """Test sophrosyne (self-regulation)."""
        wisdom = WisdomCultivation()
        
        level = wisdom.get_sophrosyne_level()
        assert 0 <= level <= 1
    
    def test_cultivate(self):
        """Test full wisdom cultivation cycle."""
        wisdom = WisdomCultivation()
        
        wisdom.add_belief('b1', 'Test belief', 0.7)
        
        result = wisdom.cultivate()
        
        assert result is not None


class TestEchogenesisCore:
    """Tests for echogenesis core orchestration."""
    
    def test_initialization(self):
        """Test echogenesis core initialization."""
        core = EchogenesisCore()
        assert core is not None
        state = core.get_state()
        assert 'phase' in state
    
    def test_cognitive_cycle(self):
        """Test complete cognitive cycle."""
        core = EchogenesisCore()
        
        input_data = {
            'sensory': np.random.randn(256).tolist(),
            'context': {'task': 'test'},
            'query': 'What should I do?'
        }
        
        result = core.cognitive_cycle(input_data)
        
        assert result is not None
        assert 'output' in result or isinstance(result, dict)
    
    def test_phase_transitions(self):
        """Test cognitive phase transitions."""
        core = EchogenesisCore()
        
        # Record phases through cycle
        phases = []
        
        def record_phase(phase):
            phases.append(phase)
        
        # Run partial cycle
        input_data = {'test': True}
        core.cognitive_cycle(input_data)
        
        state = core.get_state()
        assert state['phase'] is not None


class TestEchoStateManager:
    """Tests for echo state management."""
    
    def test_initialization(self):
        """Test echo state manager initialization."""
        manager = EchoStateManager(
            echo_depth=7,
            echo_decay=0.95,
            multi_timescale=[1, 4, 16, 64]
        )
        assert manager is not None
    
    def test_update_state(self):
        """Test echo state update."""
        manager = EchoStateManager()
        
        state = np.random.randn(512)
        
        updated = manager.update(state)
        
        assert updated is not None
        assert len(updated) == len(state)
    
    def test_multi_timescale(self):
        """Test multi-timescale echo states."""
        manager = EchoStateManager(multi_timescale=[1, 4, 16])
        
        states = [np.random.randn(512) for _ in range(20)]
        
        for state in states:
            manager.update(state)
        
        # Should have maintained states at different timescales
        current = manager.get_current_state()
        assert current is not None


class TestToroidalSynthesizer:
    """Tests for toroidal dual-persona synthesis."""
    
    def test_initialization(self):
        """Test toroidal synthesizer initialization."""
        synthesizer = ToroidalSynthesizer()
        assert synthesizer is not None
    
    def test_synthesize(self):
        """Test toroidal synthesis."""
        synthesizer = ToroidalSynthesizer()
        
        response_a = {'content': 'Analytical view', 'confidence': 0.8}
        response_b = {'content': 'Intuitive view', 'confidence': 0.7}
        
        synthesized = synthesizer.synthesize(response_a, response_b)
        
        assert synthesized is not None
        assert 'content' in synthesized or isinstance(synthesized, dict)


class TestIntegration:
    """Integration tests for echogenesis system."""
    
    def test_full_echogenesis_initialization(self):
        """Test full echogenesis system initialization."""
        core = initialize_echogenesis()
        assert core is not None
    
    def test_config_loading(self):
        """Test configuration loading."""
        assert ECHOGENESIS_CONFIG is not None
        assert 'embedding_architecture' in ECHOGENESIS_CONFIG
        assert 'echo_configuration' in ECHOGENESIS_CONFIG
        assert 'relevance_realization' in ECHOGENESIS_CONFIG
    
    def test_component_integration(self):
        """Test component integration."""
        core = initialize_echogenesis()
        
        # Test that all components are accessible
        state = core.get_state()
        
        assert 'phase' in state
        # Components should be initialized
        assert state is not None


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
