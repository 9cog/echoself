"""
Integration Test: Relevance Realization + Virtual Embodiment
==============================================================

Demonstrates how the two core systems work together to create
embodied, relevance-optimized cognition.

This shows the foundation of the improvements to EchoSelf:
1. Virtual embodiment provides sensorimotor grounding
2. Relevance realization optimizes attention/processing
3. Together they create embodied, intelligent behavior

Author: Deep Tree Echo
Date: November 2025
"""

import numpy as np
from relevance_realization_engine import (
    RelevanceRealizationEngine, 
    Possibility,
    RelevanceCriteria
)
from virtual_embodiment import (
    VirtualEmbodiment,
    ModalityType,
    SensoryInput
)


class EmbodiedIntelligentAgent:
    """
    Integrates virtual embodiment with relevance realization
    to create an embodied intelligent agent.
    
    This demonstrates the core improvement path for EchoSelf:
    - Sensorimotor grounding (embodiment)
    - Explicit relevance optimization (intelligence)
    - Integrated perception-cognition-action loop
    """
    
    def __init__(self):
        self.embodiment = VirtualEmbodiment()
        self.relevance_engine = RelevanceRealizationEngine()
        
        # Goals that drive behavior
        self.goals = {
            'explore': 0.6,
            'learn': 0.8,
            'optimize': 0.5
        }
        
        print("Embodied intelligent agent initialized")
        print(f"Goals: {self.goals}")
    
    def intelligent_behavior_cycle(self, environment):
        """
        Core cycle integrating embodiment and relevance realization:
        
        1. PERCEIVE (embodiment): Get sensory input
        2. DETECT AFFORDANCES (embodiment): What can I do?
        3. REALIZE RELEVANCE (intelligence): What SHOULD I do?
        4. ACT (embodiment): Execute selected action
        5. LEARN (both): Update models and relevance criteria
        """
        print("\n=== Intelligent Behavior Cycle ===")
        
        # Step 1: PERCEIVE - Embodied perception
        print("\n1. Embodied Perception:")
        sensory_state = self.embodiment.get_proprioception()
        print(f"   Position: {sensory_state['position']}")
        print(f"   Energy: {sensory_state['energy']:.3f}")
        
        # Step 2: DETECT AFFORDANCES - What CAN I do?
        print("\n2. Affordance Detection (Embodied):")
        affordances = self.embodiment.affordance_detector.detect_affordances(
            environment
        )
        print(f"   Detected {len(affordances)} possible actions")
        for i, aff in enumerate(affordances[:3], 1):
            print(f"   {i}. {aff['action_type']}: utility={aff['utility']:.2f}")
        
        # Step 3: REALIZE RELEVANCE - What SHOULD I do?
        print("\n3. Relevance Realization (Intelligence):")
        
        # Convert affordances to possibilities for RR engine
        possibilities = []
        for aff in affordances:
            p = Possibility(
                id=f"{aff['action_type']}_{aff.get('body_part', 'unknown')}",
                data=aff
            )
            # Set relevance criteria based on goals
            p.criteria = self._evaluate_affordance_relevance(aff)
            possibilities.append(p)
        
        # Use relevance realization to select best action
        relevant = self.relevance_engine.realize_relevance(
            possibilities,
            context={
                'goals': self.goals,
                'energy': sensory_state['energy'],
                'novelty_needed': True
            }
        )
        
        if not relevant:
            print("   No relevant actions found")
            return None
        
        best_action_possibility = relevant[0]
        best_affordance = best_action_possibility.data
        
        print(f"   Selected: {best_affordance['action_type']}")
        print(f"   Relevance score: {best_action_possibility.criteria.score():.3f}")
        print(f"   Goal alignment: {best_action_possibility.criteria.goal_alignment:.3f}")
        print(f"   Cognitive economy: {best_action_possibility.criteria.cognitive_economy:.3f}")
        
        # Step 4: ACT - Execute embodied action
        print("\n4. Embodied Action:")
        result = self.embodiment.perceive_act_cycle(
            environment,
            goal={'type': 'optimize', 'affordance': best_affordance}
        )
        
        if result['action']:
            print(f"   Executed: {result['action'].action_type}")
            print(f"   Success: {result['result'].get('success', False)}")
            print(f"   Prediction error: {result['prediction_error']:.4f}")
        
        # Step 5: LEARN - Update both systems
        print("\n5. Learning:")
        
        # Embodiment learns from sensorimotor experience
        print(f"   Embodiment learned {len(self.embodiment.contingencies)} contingencies")
        print(f"   Forward model has {len(self.embodiment.forward_model.model)} entries")
        
        # Relevance realization learns from outcomes
        outcome = {
            'success': result['result'].get('success', False),
            'prediction_error': result['prediction_error']
        }
        self.relevance_engine.feed_back([best_action_possibility], [outcome])
        
        print(f"   RR opponent balances updated:")
        print(f"     Explore/Exploit: {self.relevance_engine.exploration_exploitation.balance:.3f}")
        print(f"     Breadth/Depth: {self.relevance_engine.breadth_depth.balance:.3f}")
        
        return result
    
    def _evaluate_affordance_relevance(self, affordance):
        """Evaluate how relevant an affordance is to current goals"""
        criteria = RelevanceCriteria()
        
        # Goal alignment: How well does this serve goals?
        action_type = affordance['action_type']
        if action_type == 'move_to':
            criteria.goal_alignment = self.goals.get('explore', 0.0)
        elif action_type == 'grasp':
            criteria.goal_alignment = self.goals.get('learn', 0.0)
        else:
            criteria.goal_alignment = 0.5
        
        # Cognitive economy: How efficient?
        criteria.cognitive_economy = affordance.get('utility', 0.5)
        
        # Novelty: Exploration value
        criteria.novelty_value = 0.7  # Could track action history
        
        # Context fit: Does this make sense now?
        criteria.contextual_fit = 0.8
        
        # Predictive power: Will this improve predictions?
        criteria.predictive_power = 0.6
        
        return criteria


def demonstrate_integrated_system():
    """
    Demonstrate the integrated embodied intelligence system.
    
    This shows how EchoSelf would work with the proposed improvements:
    1. Sensorimotor grounding through virtual embodiment
    2. Intelligent action selection through relevance realization
    3. Continuous learning from embodied experience
    """
    print("=" * 70)
    print("INTEGRATED SYSTEM DEMONSTRATION")
    print("Embodied Intelligence = Virtual Embodiment + Relevance Realization")
    print("=" * 70)
    
    # Create agent
    agent = EmbodiedIntelligentAgent()
    
    # Create environment
    environment = {
        'objects': [
            {
                'id': 'obj1',
                'position': np.array([0.5, 0.2, 1.5]),
                'value': 0.9,
                'type': 'tool'
            },
            {
                'id': 'obj2',
                'position': np.array([-0.4, 0.1, 1.4]),
                'value': 0.6,
                'type': 'obstacle'
            },
            {
                'id': 'obj3',
                'position': np.array([0.3, -0.3, 1.6]),
                'value': 0.8,
                'type': 'resource'
            }
        ],
        'terrain': {
            'walkable': [
                {'center': np.array([1.0, 0.5, 0.0]), 'type': 'open'},
                {'center': np.array([0.0, 1.0, 0.0]), 'type': 'forest'},
                {'center': np.array([-1.0, 0.0, 0.0]), 'type': 'water'}
            ]
        }
    }
    
    print(f"\nEnvironment has:")
    print(f"  - {len(environment['objects'])} objects")
    print(f"  - {len(environment['terrain']['walkable'])} regions")
    
    # Run multiple behavior cycles
    num_cycles = 3
    
    for cycle in range(num_cycles):
        print(f"\n{'=' * 70}")
        print(f"CYCLE {cycle + 1}/{num_cycles}")
        print('=' * 70)
        
        result = agent.intelligent_behavior_cycle(environment)
        
        if result is None:
            print("\nNo action taken this cycle")
    
    # Final summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print('=' * 70)
    
    print(f"\nEmbodiment Learning:")
    print(f"  - Contingencies learned: {len(agent.embodiment.contingencies)}")
    print(f"  - Forward model entries: {len(agent.embodiment.forward_model.model)}")
    print(f"  - Inverse model entries: {len(agent.embodiment.inverse_model.model)}")
    
    print(f"\nRelevance Realization State:")
    print(f"  - Exploration/Exploitation: {agent.relevance_engine.exploration_exploitation.balance:.3f}")
    print(f"  - Breadth/Depth: {agent.relevance_engine.breadth_depth.balance:.3f}")
    print(f"  - Speed/Accuracy: {agent.relevance_engine.speed_accuracy.balance:.3f}")
    print(f"  - Certainty/Openness: {agent.relevance_engine.certainty_openness.balance:.3f}")
    
    print(f"\nAgent State:")
    proprio = agent.embodiment.get_proprioception()
    print(f"  - Energy: {proprio['energy']:.3f}")
    print(f"  - Position: {proprio['position']}")
    
    print("\n" + "=" * 70)
    print("This demonstrates the foundation of improved EchoSelf:")
    print("  ✅ Sensorimotor grounding (embodiment)")
    print("  ✅ Explicit relevance optimization (intelligence)")
    print("  ✅ Integrated learning from experience")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_integrated_system()
