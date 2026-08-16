"""
Wisdom Cultivation System
==========================

Implements systematic wisdom development for the Echogenesis architecture,
integrating Vervaeke's framework for wisdom cultivation with cognitive
architecture components.

Core Capabilities:
- Sophrosyne (Self-regulation and balance)
- Active Open-Mindedness (Seeking disconfirmation)
- Self-Deception Detection (Bullshit detection)
- Transformative Practices (Wisdom cultivation protocols)
- Virtue Development (Cognitive virtue cultivation)

This module addresses the wisdom cultivation dimension of intelligence,
ensuring that cognitive capabilities are tempered with sophia (wisdom).

Author: Deep Tree Echo
Date: June 2026
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WisdomDimension(Enum):
    """Dimensions of wisdom cultivation."""
    SOPHROSYNE = "sophrosyne"                # Self-regulation
    OPEN_MINDEDNESS = "open_mindedness"      # Active open-mindedness
    INTELLECTUAL_HUMILITY = "intellectual_humility"  # Knowing what you don't know
    TRUTH_SEEKING = "truth_seeking"          # Caring about truth
    SELF_EXAMINATION = "self_examination"    # Socratic introspection
    TRANSFORMATIVE = "transformative"        # Capacity for transformation


@dataclass
class Belief:
    """Represents a belief with confidence and evidence tracking."""
    content: str
    confidence: float = 0.5
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    falsification_tests: List[Dict[str, Any]] = field(default_factory=list)
    times_challenged: int = 0
    last_examined: Optional[int] = None
    reality_tested: bool = False
    caring_about_truth: float = 0.5  # 0 = bullshit, 1 = truth-committed


@dataclass
class Insight:
    """Represents an insight from self-examination."""
    question: str
    discovery: str
    depth: float = 0.5  # How deep the insight goes
    transformative: bool = False
    timestamp: int = 0


@dataclass
class WisdomState:
    """Current state of wisdom cultivation."""
    overall_wisdom: float = 0.5
    dimensions: Dict[WisdomDimension, float] = field(default_factory=dict)
    insights: List[Insight] = field(default_factory=list)
    beliefs_examined: int = 0
    deceptions_detected: int = 0
    transformations: int = 0
    
    def __post_init__(self):
        if not self.dimensions:
            for dim in WisdomDimension:
                self.dimensions[dim] = 0.5


class SophrosyneModule:
    """
    Implements sophrosyne - self-regulation, balance, and harmony.
    
    Core functions:
    - Balance between opposing tendencies
    - Regulation of cognitive resources
    - Maintenance of inner harmony
    - Recognition of limits
    """
    
    def __init__(self):
        self.balance_points: Dict[str, float] = {}
        self.regulation_history: List[Dict] = []
        self.harmony_score: float = 0.5
        
        # Initialize key balance dimensions
        self._init_balances()
    
    def _init_balances(self):
        """Initialize balance points."""
        self.balance_points = {
            'confidence_humility': 0.5,
            'action_reflection': 0.5,
            'openness_commitment': 0.5,
            'self_other': 0.5,
            'novelty_stability': 0.5,
            'simplicity_complexity': 0.5
        }
    
    def check_balance(self, dimension: str) -> float:
        """Check balance level for a dimension."""
        return self.balance_points.get(dimension, 0.5)
    
    def adjust_balance(self, dimension: str, delta: float):
        """Adjust balance for a dimension."""
        if dimension in self.balance_points:
            current = self.balance_points[dimension]
            new_value = np.clip(current + delta, 0.0, 1.0)
            self.balance_points[dimension] = new_value
            
            # Record adjustment
            self.regulation_history.append({
                'dimension': dimension,
                'from': current,
                'to': new_value,
                'delta': delta
            })
            
            # Update harmony
            self._update_harmony()
    
    def _update_harmony(self):
        """Update overall harmony score."""
        # Harmony is highest when all dimensions are balanced (near 0.5)
        deviations = [abs(v - 0.5) for v in self.balance_points.values()]
        avg_deviation = np.mean(deviations)
        self.harmony_score = 1.0 - (avg_deviation * 2)  # 0 deviation = 1.0 harmony
    
    def regulate(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform self-regulation based on context.
        
        Returns recommended adjustments.
        """
        adjustments = {}
        
        # Example regulation rules
        if context.get('high_confidence', False):
            adjustments['confidence_humility'] = -0.1  # Increase humility
            
        if context.get('overactive', False):
            adjustments['action_reflection'] = -0.1  # More reflection
            
        if context.get('scattered', False):
            adjustments['openness_commitment'] = 0.1  # More commitment
        
        # Apply adjustments
        for dim, delta in adjustments.items():
            self.adjust_balance(dim, delta)
        
        return adjustments
    
    def get_harmony(self) -> float:
        """Get current harmony score."""
        return self.harmony_score
    
    def get_state(self) -> Dict[str, Any]:
        """Get sophrosyne state."""
        return {
            'balance_points': self.balance_points.copy(),
            'harmony': self.harmony_score,
            'regulation_count': len(self.regulation_history)
        }


class SelfDeceptionDetector:
    """
    Detects self-deception (bullshit) in beliefs and reasoning.
    
    Based on Frankfurt's concept: bullshit is disconnection from
    reality without caring about truth.
    
    Key indicators:
    - No reality testing
    - Low truth-caring score
    - Resistance to falsification
    - Confirmation bias patterns
    """
    
    def __init__(self):
        self.detected_deceptions: List[Dict] = []
        self.examination_history: List[Dict] = []
        
    def examine_belief(self, belief: Belief) -> Dict[str, Any]:
        """
        Examine a belief for self-deception.
        
        Returns examination result with flags and recommendations.
        """
        result = {
            'belief_content': belief.content,
            'is_deception': False,
            'flags': [],
            'recommendations': []
        }
        
        # Check reality testing
        if not belief.reality_tested:
            result['flags'].append('not_reality_tested')
            result['recommendations'].append('Conduct reality test')
        
        # Check truth-caring
        if belief.caring_about_truth < 0.3:
            result['flags'].append('low_truth_caring')
            result['is_deception'] = True
            result['recommendations'].append('Examine motivation for holding belief')
        
        # Check evidence quality
        if not belief.evidence:
            result['flags'].append('no_evidence')
            result['recommendations'].append('Seek supporting evidence')
        
        # Check falsification attempts
        if not belief.falsification_tests:
            result['flags'].append('no_falsification_attempts')
            result['recommendations'].append('Design falsification test')
        
        # Check confidence vs evidence ratio
        if belief.confidence > 0.7 and len(belief.evidence) < 2:
            result['flags'].append('overconfident')
            result['recommendations'].append('Lower confidence or gather more evidence')
        
        # Record examination
        self.examination_history.append({
            'belief': belief.content,
            'result': result,
            'timestamp': len(self.examination_history)
        })
        
        if result['is_deception']:
            self.detected_deceptions.append(result)
            logger.warning(f"Self-deception detected: {belief.content[:50]}...")
        
        return result
    
    def scan_beliefs(self, beliefs: List[Belief]) -> List[Dict]:
        """Scan multiple beliefs for self-deception."""
        results = []
        for belief in beliefs:
            result = self.examine_belief(belief)
            results.append(result)
        return results
    
    def get_deception_rate(self) -> float:
        """Get rate of detected deceptions."""
        if not self.examination_history:
            return 0.0
        return len(self.detected_deceptions) / len(self.examination_history)
    
    def get_state(self) -> Dict[str, Any]:
        """Get detector state."""
        return {
            'deceptions_detected': len(self.detected_deceptions),
            'beliefs_examined': len(self.examination_history),
            'deception_rate': self.get_deception_rate()
        }


class ActiveOpenMindedness:
    """
    Implements active open-mindedness - actively seeking belief disconfirmation.
    
    This is not passive acceptance of all views, but active engagement
    with evidence that might challenge current beliefs.
    """
    
    def __init__(self):
        self.challenges_sought: List[Dict] = []
        self.beliefs_revised: List[Dict] = []
        self.openness_score: float = 0.5
        
    def seek_disconfirmation(self, belief: Belief) -> List[Dict]:
        """
        Actively seek ways to disconfirm a belief.
        
        Returns list of falsification tests.
        """
        tests = []
        
        # Generate basic falsification test
        tests.append({
            'type': 'direct_contradiction',
            'description': f'Look for evidence that directly contradicts: {belief.content}',
            'method': 'search_contradicting_evidence'
        })
        
        # Alternative hypothesis test
        tests.append({
            'type': 'alternative_hypothesis',
            'description': 'Generate alternative explanations for the same observations',
            'method': 'generate_alternatives'
        })
        
        # Edge case test
        tests.append({
            'type': 'edge_case',
            'description': 'Find edge cases where belief might not hold',
            'method': 'identify_boundaries'
        })
        
        # Source quality test
        tests.append({
            'type': 'source_quality',
            'description': 'Critically examine sources of evidence',
            'method': 'evaluate_sources'
        })
        
        # Record challenge sought
        self.challenges_sought.append({
            'belief': belief.content,
            'tests_generated': len(tests)
        })
        
        return tests
    
    def run_falsification_test(
        self, 
        belief: Belief, 
        test: Dict,
        evidence: Any
    ) -> Dict[str, Any]:
        """
        Run a falsification test with new evidence.
        
        Returns result including whether belief should be revised.
        """
        result = {
            'test_type': test['type'],
            'disconfirms': False,
            'confidence_delta': 0.0,
            'revision_recommended': False
        }
        
        # Simple heuristic: if evidence is provided, evaluate
        if evidence:
            if test['type'] == 'direct_contradiction':
                # Strong contradiction evidence
                result['disconfirms'] = True
                result['confidence_delta'] = -0.3
                result['revision_recommended'] = True
                
            elif test['type'] == 'alternative_hypothesis':
                # Alternative exists with merit
                result['disconfirms'] = False  # Not full disconfirmation
                result['confidence_delta'] = -0.1
                result['revision_recommended'] = belief.confidence > 0.7
        
        # Update openness score
        self.openness_score = np.clip(
            self.openness_score + 0.05, 0.0, 1.0  # Reward for testing
        )
        
        return result
    
    def revise_belief(
        self, 
        belief: Belief, 
        evidence: Dict,
        new_confidence: float
    ):
        """Record belief revision based on new evidence."""
        self.beliefs_revised.append({
            'belief': belief.content,
            'old_confidence': belief.confidence,
            'new_confidence': new_confidence,
            'evidence': evidence
        })
    
    def get_state(self) -> Dict[str, Any]:
        """Get active open-mindedness state."""
        return {
            'openness_score': self.openness_score,
            'challenges_sought': len(self.challenges_sought),
            'beliefs_revised': len(self.beliefs_revised)
        }


class TransformativePractices:
    """
    Implements transformative practices for wisdom cultivation.
    
    Based on Vervaeke's framework of practices that enable
    transformative experience and insight.
    """
    
    def __init__(self):
        self.practices: Dict[str, Dict] = self._init_practices()
        self.practice_history: List[Dict] = []
        self.transformation_count: int = 0
        
    def _init_practices(self) -> Dict[str, Dict]:
        """Initialize available practices."""
        return {
            'socratic_examination': {
                'description': 'Systematic self-questioning',
                'duration': 'medium',
                'depth': 0.8,
                'questions': [
                    "What do I believe and why?",
                    "What evidence would change my mind?",
                    "What am I assuming?",
                    "What don't I know that I don't know?",
                    "Am I deceiving myself?"
                ]
            },
            'perspective_taking': {
                'description': 'Adopting different viewpoints',
                'duration': 'short',
                'depth': 0.6,
                'frames': ['analytical', 'empathetic', 'creative', 'practical']
            },
            'deep_reflection': {
                'description': 'Extended contemplation',
                'duration': 'long',
                'depth': 0.9,
                'focus': 'core_beliefs'
            },
            'mindful_awareness': {
                'description': 'Present-moment attention',
                'duration': 'variable',
                'depth': 0.7,
                'focus': 'present_experience'
            },
            'dialectical_thinking': {
                'description': 'Thesis-antithesis synthesis',
                'duration': 'medium',
                'depth': 0.8,
                'process': 'integrate_opposites'
            }
        }
    
    def engage_practice(
        self, 
        practice_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Engage in a transformative practice.
        
        Returns insights and transformation indicators.
        """
        if practice_name not in self.practices:
            return {'error': f'Unknown practice: {practice_name}'}
        
        practice = self.practices[practice_name]
        result = {
            'practice': practice_name,
            'insights': [],
            'transformation_potential': 0.0
        }
        
        if practice_name == 'socratic_examination':
            result = self._do_socratic_examination(practice, context)
            
        elif practice_name == 'perspective_taking':
            result = self._do_perspective_taking(practice, context)
            
        elif practice_name == 'dialectical_thinking':
            result = self._do_dialectical_thinking(practice, context)
        
        # Record practice
        self.practice_history.append({
            'practice': practice_name,
            'result': result
        })
        
        # Check for transformation
        if result.get('transformation_potential', 0) > 0.7:
            self.transformation_count += 1
            result['transformation_triggered'] = True
        
        return result
    
    def _do_socratic_examination(
        self, 
        practice: Dict,
        context: Dict
    ) -> Dict:
        """Perform Socratic self-examination."""
        insights = []
        
        for question in practice['questions']:
            insight = self._introspect(question, context)
            if insight:
                insights.append(Insight(
                    question=question,
                    discovery=insight,
                    depth=practice['depth']
                ))
        
        return {
            'practice': 'socratic_examination',
            'insights': insights,
            'transformation_potential': len(insights) * 0.15,
            'questions_examined': len(practice['questions'])
        }
    
    def _do_perspective_taking(
        self, 
        practice: Dict,
        context: Dict
    ) -> Dict:
        """Perform perspective-taking practice."""
        perspectives = []
        
        for frame in practice['frames']:
            perspective = self._adopt_perspective(frame, context)
            perspectives.append({
                'frame': frame,
                'view': perspective
            })
        
        return {
            'practice': 'perspective_taking',
            'perspectives': perspectives,
            'transformation_potential': len(perspectives) * 0.1,
            'frames_explored': len(practice['frames'])
        }
    
    def _do_dialectical_thinking(
        self, 
        practice: Dict,
        context: Dict
    ) -> Dict:
        """Perform dialectical thinking practice."""
        thesis = context.get('thesis', 'Position A')
        antithesis = context.get('antithesis', 'Position B')
        
        synthesis = self._synthesize(thesis, antithesis)
        
        return {
            'practice': 'dialectical_thinking',
            'thesis': thesis,
            'antithesis': antithesis,
            'synthesis': synthesis,
            'transformation_potential': 0.6 if synthesis else 0.2
        }
    
    def _introspect(self, question: str, context: Dict) -> str:
        """Perform introspection on a question."""
        # Simulated introspection - in real system this would engage deeper
        return f"Insight regarding: {question[:30]}..."
    
    def _adopt_perspective(self, frame: str, context: Dict) -> str:
        """Adopt a perspective and generate view."""
        return f"Viewing through {frame} lens: ..."
    
    def _synthesize(self, thesis: str, antithesis: str) -> str:
        """Synthesize thesis and antithesis."""
        return f"Integration of {thesis[:20]} and {antithesis[:20]}"
    
    def get_state(self) -> Dict[str, Any]:
        """Get practices state."""
        return {
            'available_practices': list(self.practices.keys()),
            'practice_count': len(self.practice_history),
            'transformations': self.transformation_count
        }


class WisdomCultivation:
    """
    Main interface for wisdom cultivation in the Echogenesis architecture.
    
    Integrates all wisdom components:
    - Sophrosyne (self-regulation)
    - Self-deception detection
    - Active open-mindedness
    - Transformative practices
    - Virtue development
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.state = WisdomState()
        
        # Initialize components
        self.sophrosyne = SophrosyneModule()
        self.deception_detector = SelfDeceptionDetector()
        self.open_mindedness = ActiveOpenMindedness()
        self.practices = TransformativePractices()
        
        # Beliefs being tracked
        self.beliefs: Dict[str, Belief] = {}
        
        logger.info("WisdomCultivation initialized")
    
    def add_belief(self, id: str, content: str, confidence: float = 0.5):
        """Add a belief to track."""
        self.beliefs[id] = Belief(
            content=content,
            confidence=confidence
        )
    
    def examine_self(self) -> List[Insight]:
        """
        Perform Socratic self-examination.
        
        Core practice of wisdom cultivation.
        """
        result = self.practices.engage_practice(
            'socratic_examination',
            {'beliefs': list(self.beliefs.values())}
        )
        
        insights = result.get('insights', [])
        self.state.insights.extend(insights)
        
        return insights
    
    def detect_deceptions(self) -> List[Dict]:
        """Scan all beliefs for self-deception."""
        deceptions = []
        
        for belief in self.beliefs.values():
            result = self.deception_detector.examine_belief(belief)
            if result['is_deception']:
                deceptions.append(result)
        
        self.state.deceptions_detected = len(deceptions)
        return deceptions
    
    def challenge_belief(self, belief_id: str) -> Dict[str, Any]:
        """Actively challenge a specific belief."""
        if belief_id not in self.beliefs:
            return {'error': 'Belief not found'}
        
        belief = self.beliefs[belief_id]
        
        # Generate falsification tests
        tests = self.open_mindedness.seek_disconfirmation(belief)
        
        # Update belief with tests
        belief.falsification_tests.extend(tests)
        belief.times_challenged += 1
        
        self.state.beliefs_examined += 1
        
        return {
            'belief': belief.content,
            'tests_generated': tests,
            'times_challenged': belief.times_challenged
        }
    
    def regulate(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Perform self-regulation based on context."""
        return self.sophrosyne.regulate(context)
    
    def engage_practice(
        self, 
        practice_name: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Engage in a transformative practice."""
        return self.practices.engage_practice(
            practice_name,
            context or {}
        )
    
    def update_dimension(self, dimension: WisdomDimension, value: float):
        """Update a wisdom dimension score."""
        self.state.dimensions[dimension] = np.clip(value, 0.0, 1.0)
        self._update_overall_wisdom()
    
    def _update_overall_wisdom(self):
        """Update overall wisdom score."""
        scores = list(self.state.dimensions.values())
        self.state.overall_wisdom = np.mean(scores)
    
    def cultivate(self) -> Dict[str, Any]:
        """
        Run full wisdom cultivation cycle.
        
        1. Self-examination
        2. Deception detection
        3. Balance regulation
        4. Insight integration
        """
        results = {
            'insights': [],
            'deceptions': [],
            'regulations': {},
            'wisdom_score': 0.0
        }
        
        # 1. Self-examination
        insights = self.examine_self()
        results['insights'] = [i.discovery for i in insights]
        
        # 2. Deception detection
        deceptions = self.detect_deceptions()
        results['deceptions'] = deceptions
        
        # 3. Regulation
        context = {
            'high_confidence': any(
                b.confidence > 0.8 for b in self.beliefs.values()
            ),
            'overactive': len(self.state.insights) > 10,
            'scattered': len(self.beliefs) > 20
        }
        regulations = self.regulate(context)
        results['regulations'] = regulations
        
        # 4. Update dimensions
        self._update_dimensions_from_cycle(results)
        
        results['wisdom_score'] = self.state.overall_wisdom
        
        return results
    
    def _update_dimensions_from_cycle(self, cycle_results: Dict):
        """Update wisdom dimensions based on cultivation cycle."""
        # Open-mindedness from challenging beliefs
        if self.state.beliefs_examined > 0:
            self.update_dimension(
                WisdomDimension.OPEN_MINDEDNESS,
                min(1.0, self.state.beliefs_examined * 0.1)
            )
        
        # Truth-seeking from deception detection
        if len(cycle_results['deceptions']) > 0:
            # Detecting deceptions shows truth-seeking
            current = self.state.dimensions[WisdomDimension.TRUTH_SEEKING]
            self.update_dimension(
                WisdomDimension.TRUTH_SEEKING,
                current + 0.05
            )
        
        # Sophrosyne from harmony
        harmony = self.sophrosyne.get_harmony()
        self.update_dimension(WisdomDimension.SOPHROSYNE, harmony)
    
    def get_wisdom_score(self) -> float:
        """Get current overall wisdom score."""
        return self.state.overall_wisdom
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get complete wisdom cultivation state."""
        return {
            'overall_wisdom': self.state.overall_wisdom,
            'dimensions': {
                k.value: v for k, v in self.state.dimensions.items()
            },
            'insights_count': len(self.state.insights),
            'beliefs_examined': self.state.beliefs_examined,
            'deceptions_detected': self.state.deceptions_detected,
            'transformations': self.state.transformations,
            'sophrosyne': self.sophrosyne.get_state(),
            'open_mindedness': self.open_mindedness.get_state(),
            'practices': self.practices.get_state()
        }


# Convenience functions
def create_wisdom_cultivation(
    config: Optional[Dict[str, Any]] = None
) -> WisdomCultivation:
    """Factory function to create WisdomCultivation instance."""
    return WisdomCultivation(config)


def quick_wisdom_check(beliefs: List[Dict]) -> Dict[str, Any]:
    """Quick wisdom check on a set of beliefs."""
    cultivation = WisdomCultivation()
    
    for i, belief_dict in enumerate(beliefs):
        cultivation.add_belief(
            f'belief_{i}',
            belief_dict.get('content', ''),
            belief_dict.get('confidence', 0.5)
        )
    
    return cultivation.cultivate()
