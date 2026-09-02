"""
Integration Tests for Plan Implementation
==========================================

Tests for:
- RelevanceRealizationEngine (Python)
- RelevanceBridge (Python-TypeScript)
- WisdomCultivationSystem (Python)
- VirtualEmbodiment (Python)

Author: Deep Tree Echo
Date: June 2026
"""

import unittest
import sys
import os
import time
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relevance_realization_engine import (
    RelevanceRealizationEngine,
    Possibility,
    RelevanceCriteria,
    OpponentProcess
)

from wisdom_cultivation import (
    WisdomCultivationSystem,
    Belief,
    VirtueType
)

from virtual_embodiment import (
    VirtualEmbodiment,
    SensoryInput,
    ModalityType,
    MotorCommand
)

import numpy as np


class TestRelevanceRealizationEngine(unittest.TestCase):
    """Tests for the Relevance Realization Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = RelevanceRealizationEngine()
    
    def test_engine_initialization(self):
        """Test that engine initializes with correct components"""
        self.assertIsNotNone(self.engine.exploration_exploitation)
        self.assertIsNotNone(self.engine.breadth_depth)
        self.assertIsNotNone(self.engine.speed_accuracy)
        self.assertIsNotNone(self.engine.certainty_openness)
        self.assertEqual(len(self.engine.cost_functions), 3)
    
    def test_opponent_process_initialization(self):
        """Test opponent process initial balance"""
        self.assertAlmostEqual(
            self.engine.exploration_exploitation.balance, 0.5, places=1
        )
    
    def test_opponent_process_shift(self):
        """Test opponent process shifting"""
        op = OpponentProcess(0.5, "test")
        
        op.shift_toward_first(0.2)
        self.assertAlmostEqual(op.balance, 0.3, places=1)
        
        op.shift_toward_second(0.3)
        self.assertAlmostEqual(op.balance, 0.6, places=1)
    
    def test_opponent_process_bounds(self):
        """Test opponent process stays within bounds"""
        op = OpponentProcess(0.1, "test")
        
        op.shift(-0.5)  # Try to go below 0
        self.assertGreaterEqual(op.balance, 0.0)
        
        op.shift(2.0)  # Try to go above 1
        self.assertLessEqual(op.balance, 1.0)
    
    def test_relevance_criteria_score(self):
        """Test relevance criteria scoring"""
        criteria = RelevanceCriteria(
            goal_alignment=0.8,
            predictive_power=0.7,
            cognitive_economy=0.6,
            novelty_value=0.5,
            contextual_fit=0.4
        )
        
        score = criteria.score()
        
        # Score should be weighted average
        expected = (0.8 * 0.3 + 0.7 * 0.25 + 0.6 * 0.2 + 0.5 * 0.15 + 0.4 * 0.1)
        self.assertAlmostEqual(score, expected, places=2)
    
    def test_realize_relevance_basic(self):
        """Test basic relevance realization"""
        # Create test possibilities
        possibilities = []
        for i in range(100):
            p = Possibility(
                id=f"possibility_{i}",
                data={'value': i, 'complexity': np.random.random()}
            )
            possibilities.append(p)
        
        # Realize relevance
        relevant = self.engine.realize_relevance(possibilities)
        
        # Should filter down
        self.assertLess(len(relevant), len(possibilities))
        self.assertGreater(len(relevant), 0)
    
    def test_realize_relevance_with_context(self):
        """Test relevance realization with context"""
        possibilities = [
            Possibility(id="p1", data={'value': 1}),
            Possibility(id="p2", data={'value': 2}),
            Possibility(id="p3", data={'value': 3}),
        ]
        
        context = {
            'goals': [{'id': 'g1', 'priority': 0.9}],
            'novelty_needed': True
        }
        
        relevant = self.engine.realize_relevance(possibilities, context)
        
        # Context should be stored
        self.assertIn('goals', self.engine.current_context)
    
    def test_feed_back_updates_history(self):
        """Test that feedback updates outcome history"""
        chosen = [
            Possibility(id="p1", data={'value': 1})
        ]
        
        outcomes = [
            type('Outcome', (), {'success': True})()
        ]
        
        initial_size = len(self.engine.outcome_history)
        self.engine.feed_back(chosen, outcomes)
        
        self.assertEqual(
            len(self.engine.outcome_history),
            initial_size + 1
        )


class TestWisdomCultivationSystem(unittest.TestCase):
    """Tests for the Wisdom Cultivation System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.wisdom = WisdomCultivationSystem()
    
    def test_system_initialization(self):
        """Test that system initializes correctly"""
        self.assertIsNotNone(self.wisdom.sophrosyne)
        self.assertIsNotNone(self.wisdom.humility)
        self.assertIsNotNone(self.wisdom.deception_detector)
        self.assertIsNotNone(self.wisdom.practices)
        self.assertIsNotNone(self.wisdom.virtues)
    
    def test_add_belief(self):
        """Test adding a belief"""
        belief = self.wisdom.add_belief(
            "test_belief",
            "This is a test belief",
            confidence=0.75,
            evidence=["Evidence 1"]
        )
        
        self.assertEqual(belief.id, "test_belief")
        self.assertEqual(belief.content, "This is a test belief")
        self.assertEqual(belief.confidence, 0.75)
        self.assertIn("test_belief", self.wisdom.beliefs)
    
    def test_belief_bullshit_detection(self):
        """Test detection of bullshit beliefs (Frankfurt's definition)"""
        # Add a bullshit belief (not reality-tested, doesn't care about truth)
        belief = self.wisdom.add_belief(
            "bs_belief",
            "I don't have any blind spots",
            confidence=0.6
        )
        belief.caring_about_truth = 0.3  # Low care for truth
        belief.reality_tested = False
        
        self.assertTrue(belief.is_bullshit)
    
    def test_examine_self(self):
        """Test Socratic self-examination"""
        # Add some beliefs first
        self.wisdom.add_belief("b1", "Belief 1", 0.9)
        self.wisdom.add_belief("b2", "Belief 2", 0.8)
        
        insights = self.wisdom.examine_self()
        
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)
    
    def test_detect_deceptions(self):
        """Test self-deception detection"""
        # Add a problematic belief
        belief = self.wisdom.add_belief(
            "problem_belief",
            "I'm never wrong",
            confidence=0.95
        )
        belief.caring_about_truth = 0.2
        belief.reality_tested = False
        
        deceptions = self.wisdom.detect_deceptions()
        
        self.assertGreater(len(deceptions), 0)
        self.assertEqual(deceptions[0].belief.id, "problem_belief")
    
    def test_cultivate_cycle(self):
        """Test full wisdom cultivation cycle"""
        # Add some beliefs
        self.wisdom.add_belief("b1", "Belief 1", 0.9)
        self.wisdom.add_belief("b2", "Belief 2", 0.7)
        
        results = self.wisdom.cultivate()
        
        self.assertIn('insights', results)
        self.assertIn('deceptions', results)
        self.assertIn('wisdom_score', results)
        self.assertIn('regulation_assessment', results)
    
    def test_wisdom_score_calculation(self):
        """Test wisdom score calculation"""
        score = self.wisdom.get_wisdom_score()
        
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score.overall, 0)
        self.assertLessEqual(score.overall, 1)
        self.assertGreaterEqual(score.morality, 0)
        self.assertGreaterEqual(score.meaning, 0)
        self.assertGreaterEqual(score.mastery, 0)
    
    def test_virtue_practice(self):
        """Test virtue practice increases level"""
        initial_level = self.wisdom.virtues.virtues[
            VirtueType.INTELLECTUAL_HUMILITY
        ].level
        
        self.wisdom.virtues.practice_virtue(VirtueType.INTELLECTUAL_HUMILITY)
        
        new_level = self.wisdom.virtues.virtues[
            VirtueType.INTELLECTUAL_HUMILITY
        ].level
        
        self.assertGreater(new_level, initial_level)
    
    def test_sophrosyne_assessment(self):
        """Test sophrosyne (self-regulation) assessment"""
        state = {
            'dimension_1': 0.5,
            'dimension_2': 0.9,  # Extreme
            'dimension_3': 0.4
        }
        
        assessment = self.wisdom.sophrosyne.assess_regulation(state)
        
        self.assertIn('extremes', assessment)
        self.assertIn('balance_score', assessment)
        self.assertGreater(len(assessment['extremes']), 0)  # Should detect extreme


class TestVirtualEmbodiment(unittest.TestCase):
    """Tests for the Virtual Embodiment Layer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.embodiment = VirtualEmbodiment()
    
    def test_embodiment_initialization(self):
        """Test that embodiment initializes correctly"""
        self.assertIsNotNone(self.embodiment.body_schema)
        self.assertIsNotNone(self.embodiment.forward_model)
        self.assertIsNotNone(self.embodiment.inverse_model)
    
    def test_body_schema_parts(self):
        """Test body schema has expected parts"""
        self.assertIn('head', self.embodiment.body_schema.parts)
        self.assertIn('torso', self.embodiment.body_schema.parts)
    
    def test_sensory_input_creation(self):
        """Test sensory input creation"""
        sensory = SensoryInput(
            modality=ModalityType.VISION,
            data=np.random.rand(64),
            timestamp=time.time()
        )
        
        self.assertEqual(sensory.modality, ModalityType.VISION)
        self.assertEqual(sensory.data.shape, (64,))
    
    def test_motor_command_creation(self):
        """Test motor command creation"""
        motor = MotorCommand(
            action_type="move",
            parameters={"direction": "forward", "speed": 0.5}
        )
        
        self.assertEqual(motor.action_type, "move")
        self.assertIn("direction", motor.parameters)
    
    def test_perceive_act_cycle(self):
        """Test perceive-act cycle"""
        environment = {
            "objects": [
                {"id": "obj1", "position": np.array([0.5, 0.0, 1.5]), "value": 0.8}
            ],
            "terrain": {
                "walkable": [{"center": np.array([1.0, 0.0, 0.0])}]
            }
        }
        
        result = self.embodiment.perceive_act_cycle(environment)
        
        # Result should have 'action' key (may be None if no affordances)
        self.assertIn('action', result)
    
    def test_affordance_detection(self):
        """Test affordance detection"""
        environment = {
            "objects": [
                {"id": "obj1", "position": np.array([0.5, 0.0, 1.5]), "value": 0.8}
            ],
            "terrain": {
                "walkable": [{"center": np.array([1.0, 0.0, 0.0])}]
            }
        }
        
        affordances = self.embodiment.affordance_detector.detect_affordances(environment)
        
        self.assertIsInstance(affordances, list)
    
    def test_proprioception(self):
        """Test proprioceptive sense"""
        proprio = self.embodiment.get_proprioception()
        
        # Check for actual keys in proprioception
        self.assertIn('posture', proprio)
        self.assertIn('energy', proprio)
        self.assertIn('position', proprio)


class TestRelevanceBridge(unittest.TestCase):
    """Tests for the Relevance Bridge (Python-TypeScript integration)"""
    
    def test_bridge_import(self):
        """Test that relevance bridge can be imported"""
        try:
            from integration.relevance_bridge import RelevanceBridge
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"RelevanceBridge not available: {e}")
    
    def test_bridge_initialization(self):
        """Test bridge initialization"""
        try:
            from integration.relevance_bridge import RelevanceBridge
            bridge = RelevanceBridge()
            self.assertIsNotNone(bridge.engine)
        except ImportError:
            self.skipTest("RelevanceBridge not available")
    
    def test_bridge_realize_relevance(self):
        """Test bridge realize_relevance method"""
        try:
            from integration.relevance_bridge import RelevanceBridge
            bridge = RelevanceBridge()
            
            possibilities = [
                {'id': 'p1', 'data': {'value': 1}},
                {'id': 'p2', 'data': {'value': 2}},
                {'id': 'p3', 'data': {'value': 3}},
            ]
            
            result = bridge.realize_relevance(possibilities)
            
            self.assertIsNotNone(result)
            self.assertGreater(result.filtered_count, 0)
        except ImportError:
            self.skipTest("RelevanceBridge not available")


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components"""
    
    def test_rr_engine_with_wisdom(self):
        """Test RR engine informed by wisdom cultivation"""
        wisdom = WisdomCultivationSystem()
        rr = RelevanceRealizationEngine()
        
        # Add beliefs that inform relevance
        wisdom.add_belief(
            "exploration_valuable",
            "Exploration leads to valuable discoveries",
            confidence=0.8
        )
        
        # Cultivate wisdom
        wisdom.cultivate()
        
        # Get wisdom score
        score = wisdom.get_wisdom_score()
        
        # Adjust RR based on wisdom
        if score.overall > 0.6:
            # High wisdom: more exploration
            rr.exploration_exploitation.shift_toward_first(0.1)
        
        # Create possibilities
        possibilities = [
            Possibility(id=f"p{i}", data={'value': i})
            for i in range(20)
        ]
        
        # Realize relevance
        relevant = rr.realize_relevance(possibilities)
        
        self.assertGreater(len(relevant), 0)
    
    def test_embodiment_with_rr(self):
        """Test embodiment informing relevance realization"""
        embodiment = VirtualEmbodiment()
        rr = RelevanceRealizationEngine()
        
        # Get proprioceptive state
        proprio = embodiment.get_proprioception()
        
        # Use embodiment state to set context
        context = {
            'cognitive_load': proprio.get('energy_level', 0.5),
            'novelty_needed': proprio.get('attention_ready', True)
        }
        
        # Realize relevance with embodied context
        possibilities = [
            Possibility(id=f"p{i}", data={'value': i})
            for i in range(10)
        ]
        
        relevant = rr.realize_relevance(possibilities, context)
        
        self.assertIn('cognitive_load', rr.current_context)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRelevanceRealizationEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestWisdomCultivationSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestVirtualEmbodiment))
    suite.addTests(loader.loadTestsFromTestCase(TestRelevanceBridge))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("EchoSelf Implementation Plan - Integration Tests")
    print("=" * 70 + "\n")
    
    result = run_tests()
    
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
