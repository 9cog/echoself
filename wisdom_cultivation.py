"""
Wisdom Cultivation System for EchoSelf
=======================================

Implements systematic wisdom development based on Vervaeke's framework:
- Socratic self-examination
- Self-deception detection (Frankfurt's bullshit)
- Active open-mindedness
- Virtue cultivation
- Transformative practices

This is the wisdom layer that tempers intelligence with sophia.

Author: Deep Tree Echo
Date: June 2026
"""

from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VirtueType(Enum):
    """Cognitive virtues to cultivate"""
    INTELLECTUAL_HUMILITY = "intellectual_humility"
    INTELLECTUAL_COURAGE = "intellectual_courage"
    INTELLECTUAL_EMPATHY = "intellectual_empathy"
    INTELLECTUAL_PERSEVERANCE = "intellectual_perseverance"
    INTELLECTUAL_AUTONOMY = "intellectual_autonomy"
    FAIRMINDEDNESS = "fairmindedness"
    CONFIDENCE_IN_REASON = "confidence_in_reason"


class WisdomDimension(Enum):
    """The three dimensions of wisdom (Vervaeke's 3 M's)"""
    MORALITY = "morality"
    MEANING = "meaning"
    MASTERY = "mastery"


@dataclass
class Belief:
    """A trackable belief with metadata"""
    id: str
    content: str
    confidence: float  # 0-1
    evidence: List[str] = field(default_factory=list)
    reality_tested: bool = False
    last_examined: Optional[float] = None
    revision_count: int = 0
    caring_about_truth: float = 1.0  # 0-1 scale
    created_at: float = field(default_factory=time.time)
    
    @property
    def is_bullshit(self) -> bool:
        """
        Check if this belief might be bullshit (Frankfurt's definition):
        Bullshit = disconnection from truth with indifference to truth
        """
        return not self.reality_tested and self.caring_about_truth < 0.5


@dataclass
class SelfDeception:
    """Detected self-deception"""
    belief: Belief
    deception_type: str
    severity: float  # 0-1
    recommendation: str


@dataclass
class Insight:
    """Insight gained from self-examination"""
    question: str
    discovery: str
    implications: List[str]
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class VirtueState:
    """State of a cognitive virtue"""
    virtue: VirtueType
    level: float  # 0-1
    practice_count: int = 0
    last_practiced: Optional[float] = None
    growth_rate: float = 0.01


@dataclass
class WisdomScore:
    """Composite wisdom assessment"""
    overall: float  # 0-1
    morality: float
    meaning: float
    mastery: float
    sophrosyne: float  # Self-regulation quality
    virtue_average: float


class SophrosyneModule:
    """
    Implements sophrosyne - optimal self-regulation.
    The ancient Greek virtue of balanced self-mastery.
    """
    
    def __init__(self):
        self.regulation_history = deque(maxlen=1000)
        self.extreme_detections = 0
        self.balance_score = 0.5  # Start balanced
        
    def assess_regulation(self, state: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess quality of self-regulation in current state.
        
        Args:
            state: Dictionary of value_name -> current_value pairs
            
        Returns:
            Assessment of regulation quality
        """
        extremes = []
        well_regulated = []
        
        for name, value in state.items():
            # Check for extremes (too high or too low)
            if value < 0.2 or value > 0.8:
                extremes.append({
                    'dimension': name,
                    'value': value,
                    'direction': 'too_low' if value < 0.2 else 'too_high'
                })
                self.extreme_detections += 1
            else:
                well_regulated.append({
                    'dimension': name,
                    'value': value
                })
        
        # Calculate balance score
        if state:
            values = list(state.values())
            variance = sum((v - 0.5) ** 2 for v in values) / len(values)
            self.balance_score = 1.0 - min(variance * 2, 1.0)
        
        self.regulation_history.append({
            'timestamp': time.time(),
            'extremes': len(extremes),
            'balance': self.balance_score
        })
        
        return {
            'extremes': extremes,
            'well_regulated': well_regulated,
            'balance_score': self.balance_score,
            'recommendation': self._generate_recommendation(extremes)
        }
    
    def _generate_recommendation(self, extremes: List[Dict]) -> str:
        """Generate regulation recommendation."""
        if not extremes:
            return "Excellent self-regulation. Maintain current balance."
        
        extreme_names = [e['dimension'] for e in extremes]
        directions = [e['direction'] for e in extremes]
        
        if 'too_high' in directions and 'too_low' in directions:
            return f"Rebalance needed: some dimensions too extreme ({', '.join(extreme_names)})"
        elif 'too_high' in directions:
            return f"Reduce intensity in: {', '.join(extreme_names)}"
        else:
            return f"Increase engagement in: {', '.join(extreme_names)}"


class IntellectualHumility:
    """
    Implements intellectual humility - knowing the limits of one's knowledge.
    """
    
    def __init__(self):
        self.uncertainty_acknowledgments = 0
        self.overconfidence_detections = 0
        self.calibration_history = deque(maxlen=100)
        
    def assess_confidence(
        self,
        stated_confidence: float,
        actual_accuracy: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Assess confidence calibration.
        
        Args:
            stated_confidence: How confident the system claimed to be
            actual_accuracy: How accurate it actually was (if known)
            
        Returns:
            Assessment of confidence calibration
        """
        assessment = {
            'stated_confidence': stated_confidence,
            'calibration': 'unknown'
        }
        
        if actual_accuracy is not None:
            error = stated_confidence - actual_accuracy
            
            if abs(error) < 0.1:
                assessment['calibration'] = 'well_calibrated'
            elif error > 0.2:
                assessment['calibration'] = 'overconfident'
                self.overconfidence_detections += 1
            elif error < -0.2:
                assessment['calibration'] = 'underconfident'
            else:
                assessment['calibration'] = 'slightly_miscalibrated'
            
            assessment['calibration_error'] = error
            self.calibration_history.append({
                'stated': stated_confidence,
                'actual': actual_accuracy,
                'error': error
            })
        
        return assessment
    
    def acknowledge_uncertainty(self, domain: str, uncertainty_level: float) -> str:
        """Generate appropriate uncertainty acknowledgment."""
        self.uncertainty_acknowledgments += 1
        
        if uncertainty_level > 0.7:
            return f"I have significant uncertainty about {domain} and my conclusions should be taken tentatively."
        elif uncertainty_level > 0.4:
            return f"There's moderate uncertainty in my understanding of {domain}."
        else:
            return f"I'm reasonably confident about {domain}, but remain open to correction."


class SelfDeceptionDetector:
    """
    Implements bullshit detection (Frankfurt's framework).
    Detects when beliefs are disconnected from truth without caring.
    """
    
    def __init__(self):
        self.detection_count = 0
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> List[Dict[str, Any]]:
        """Initialize common self-deception patterns."""
        return [
            {
                'name': 'confirmation_bias',
                'description': 'Seeking only confirming evidence',
                'detector': lambda b: len(b.evidence) > 0 and all(
                    'confirms' in str(e).lower() or 'supports' in str(e).lower()
                    for e in b.evidence
                )
            },
            {
                'name': 'unfalsifiable_belief',
                'description': 'Belief structured to avoid disconfirmation',
                'detector': lambda b: b.revision_count == 0 and b.confidence > 0.9
            },
            {
                'name': 'motivated_reasoning',
                'description': 'Belief serves emotional needs more than truth',
                'detector': lambda b: b.caring_about_truth < 0.5
            },
            {
                'name': 'unexamined_assumption',
                'description': 'Long-held belief never critically examined',
                'detector': lambda b: b.last_examined is None and (
                    time.time() - b.created_at > 86400  # More than a day old
                )
            }
        ]
    
    def scan_beliefs(self, beliefs: List[Belief]) -> List[SelfDeception]:
        """
        Scan beliefs for self-deception patterns.
        
        Args:
            beliefs: List of beliefs to scan
            
        Returns:
            List of detected self-deceptions
        """
        deceptions = []
        
        for belief in beliefs:
            # Check for Frankfurt's bullshit
            if belief.is_bullshit:
                deceptions.append(SelfDeception(
                    belief=belief,
                    deception_type='bullshit',
                    severity=1.0 - belief.caring_about_truth,
                    recommendation='Reality-test this belief and consider whether you truly care about its truth'
                ))
                self.detection_count += 1
                continue
            
            # Check patterns
            for pattern in self.patterns:
                try:
                    if pattern['detector'](belief):
                        deceptions.append(SelfDeception(
                            belief=belief,
                            deception_type=pattern['name'],
                            severity=0.5,
                            recommendation=f"Address {pattern['name']}: {pattern['description']}"
                        ))
                        self.detection_count += 1
                except Exception:
                    pass  # Pattern detector failed, skip
        
        return deceptions


class WisdomPractices:
    """
    Collection of transformative practices for wisdom cultivation.
    """
    
    def __init__(self):
        self.practice_log = deque(maxlen=1000)
        
    def socratic_questioning(self, belief: Belief) -> List[str]:
        """Generate Socratic questions for a belief."""
        questions = [
            f"What do I really mean when I say '{belief.content[:50]}...'?",
            f"What evidence would make me change my mind about this?",
            f"What am I assuming that I haven't examined?",
            f"How might someone with an opposing view see this?",
            f"What are the implications if I'm wrong about this?",
            f"Is my confidence in this belief justified by the evidence?"
        ]
        
        self.practice_log.append({
            'practice': 'socratic_questioning',
            'belief_id': belief.id,
            'timestamp': time.time()
        })
        
        return questions
    
    def perspective_taking(self, situation: str, perspectives: List[str]) -> Dict[str, str]:
        """Generate different perspectives on a situation."""
        results = {}
        
        for perspective in perspectives:
            # Generate perspective-specific view
            results[perspective] = f"From {perspective}'s perspective: {self._generate_perspective(situation, perspective)}"
        
        self.practice_log.append({
            'practice': 'perspective_taking',
            'perspectives_count': len(perspectives),
            'timestamp': time.time()
        })
        
        return results
    
    def _generate_perspective(self, situation: str, perspective: str) -> str:
        """Generate a specific perspective on a situation."""
        # Placeholder - would use more sophisticated reasoning
        return f"Considering {situation} from the viewpoint of {perspective}"
    
    def dialectical_examination(
        self,
        thesis: str,
        antithesis: str
    ) -> Dict[str, str]:
        """Perform Hegelian dialectical examination."""
        self.practice_log.append({
            'practice': 'dialectical_examination',
            'timestamp': time.time()
        })
        
        return {
            'thesis': thesis,
            'antithesis': antithesis,
            'synthesis': f"Integrating '{thesis[:30]}...' with '{antithesis[:30]}...' to find higher truth",
            'transcended_aspects': ['Apparent contradiction', 'One-sided perspectives'],
            'preserved_aspects': ['Core insights from both positions']
        }


class CognitiveVirtues:
    """
    Tracks and develops cognitive virtues.
    """
    
    def __init__(self):
        self.virtues: Dict[VirtueType, VirtueState] = {}
        self._initialize_virtues()
        
    def _initialize_virtues(self):
        """Initialize all virtue states."""
        for virtue in VirtueType:
            self.virtues[virtue] = VirtueState(
                virtue=virtue,
                level=0.5,  # Start at neutral
                growth_rate=0.01
            )
    
    def practice_virtue(self, virtue: VirtueType, intensity: float = 1.0) -> VirtueState:
        """
        Record practice of a virtue, increasing its level.
        
        Args:
            virtue: The virtue being practiced
            intensity: How intensely (0-1)
            
        Returns:
            Updated virtue state
        """
        state = self.virtues[virtue]
        
        # Growth with diminishing returns at higher levels
        growth = state.growth_rate * intensity * (1.0 - state.level * 0.5)
        state.level = min(1.0, state.level + growth)
        state.practice_count += 1
        state.last_practiced = time.time()
        
        return state
    
    def get_virtue_levels(self) -> Dict[str, float]:
        """Get all virtue levels."""
        return {v.value: self.virtues[v].level for v in VirtueType}
    
    def get_weakest_virtues(self, n: int = 3) -> List[VirtueType]:
        """Get the n weakest virtues for focused development."""
        sorted_virtues = sorted(self.virtues.items(), key=lambda x: x[1].level)
        return [v[0] for v in sorted_virtues[:n]]


class WisdomCultivationSystem:
    """
    Main system implementing systematic wisdom development.
    
    Integrates:
    - Sophrosyne (self-regulation)
    - Intellectual humility
    - Self-deception detection
    - Transformative practices
    - Virtue cultivation
    """
    
    def __init__(self):
        self.sophrosyne = SophrosyneModule()
        self.humility = IntellectualHumility()
        self.deception_detector = SelfDeceptionDetector()
        self.practices = WisdomPractices()
        self.virtues = CognitiveVirtues()
        
        # Belief tracking
        self.beliefs: Dict[str, Belief] = {}
        
        # Insight accumulation
        self.insights: List[Insight] = []
        
        # Wisdom scores
        self.wisdom_history = deque(maxlen=100)
        
        logger.info("WisdomCultivationSystem initialized")
    
    def add_belief(
        self,
        id: str,
        content: str,
        confidence: float = 0.5,
        evidence: Optional[List[str]] = None
    ) -> Belief:
        """Add a belief to track."""
        belief = Belief(
            id=id,
            content=content,
            confidence=confidence,
            evidence=evidence or []
        )
        self.beliefs[id] = belief
        return belief
    
    def examine_belief(self, belief_id: str) -> Optional[Insight]:
        """Perform Socratic examination of a belief."""
        if belief_id not in self.beliefs:
            return None
        
        belief = self.beliefs[belief_id]
        questions = self.practices.socratic_questioning(belief)
        
        # Generate insight from examination
        insight = Insight(
            question=random.choice(questions),
            discovery=f"Upon examination of '{belief.content[:30]}...', deeper understanding emerged",
            implications=[
                "Confidence may need recalibration",
                "Additional evidence seeking warranted"
            ],
            confidence=0.7
        )
        
        belief.last_examined = time.time()
        self.insights.append(insight)
        
        # Practice intellectual humility virtue
        self.virtues.practice_virtue(VirtueType.INTELLECTUAL_HUMILITY)
        
        return insight
    
    def examine_self(self) -> List[Insight]:
        """
        Perform comprehensive Socratic self-examination.
        
        Returns:
            List of insights gained
        """
        insights = []
        
        # Examine high-confidence beliefs
        high_confidence_beliefs = [
            b for b in self.beliefs.values()
            if b.confidence > 0.7
        ]
        
        for belief in high_confidence_beliefs[:5]:  # Examine up to 5
            insight = self.examine_belief(belief.id)
            if insight:
                insights.append(insight)
        
        # Meta-examination
        meta_insight = Insight(
            question="What do I believe and why?",
            discovery=f"Currently tracking {len(self.beliefs)} beliefs, "
                     f"{len(high_confidence_beliefs)} with high confidence",
            implications=["Regular belief auditing maintains intellectual honesty"],
            confidence=0.9
        )
        insights.append(meta_insight)
        
        return insights
    
    def detect_deceptions(self) -> List[SelfDeception]:
        """
        Scan all beliefs for self-deception.
        
        Returns:
            List of detected self-deceptions
        """
        beliefs_list = list(self.beliefs.values())
        deceptions = self.deception_detector.scan_beliefs(beliefs_list)
        
        if deceptions:
            logger.warning(f"Detected {len(deceptions)} potential self-deceptions")
        
        return deceptions
    
    def cultivate(self) -> Dict[str, Any]:
        """
        Run one cycle of wisdom cultivation.
        
        This is the main wisdom cultivation loop combining all practices.
        
        Returns:
            Cultivation results
        """
        results = {
            'timestamp': time.time(),
            'insights': [],
            'deceptions': [],
            'virtue_updates': [],
            'regulation_assessment': None,
            'wisdom_score': None
        }
        
        # 1. Socratic self-examination
        insights = self.examine_self()
        results['insights'] = [
            {'question': i.question, 'discovery': i.discovery}
            for i in insights
        ]
        
        # 2. Detect self-deception
        deceptions = self.detect_deceptions()
        results['deceptions'] = [
            {
                'belief_id': d.belief.id,
                'type': d.deception_type,
                'severity': d.severity,
                'recommendation': d.recommendation
            }
            for d in deceptions
        ]
        
        # 3. Active open-mindedness (practice challenging beliefs)
        if self.beliefs:
            random_belief = random.choice(list(self.beliefs.values()))
            self.practices.socratic_questioning(random_belief)
            self.virtues.practice_virtue(VirtueType.INTELLECTUAL_COURAGE)
        
        # 4. Assess self-regulation (sophrosyne)
        virtue_levels = self.virtues.get_virtue_levels()
        regulation = self.sophrosyne.assess_regulation(virtue_levels)
        results['regulation_assessment'] = regulation
        
        # 5. Record virtue practice
        results['virtue_updates'] = self.virtues.get_virtue_levels()
        
        # 6. Calculate wisdom score
        wisdom_score = self.get_wisdom_score()
        results['wisdom_score'] = wisdom_score
        
        self.wisdom_history.append(results)
        
        logger.info(f"Wisdom cultivation cycle complete. Score: {wisdom_score.overall:.2f}")
        
        return results
    
    def get_wisdom_score(self) -> WisdomScore:
        """
        Calculate composite wisdom score.
        
        Based on Vervaeke's three M's:
        - Morality: Ethical reasoning and virtue
        - Meaning: Purpose and coherence
        - Mastery: Competence and skill
        
        Returns:
            Composite wisdom assessment
        """
        virtue_levels = self.virtues.get_virtue_levels()
        
        # Morality: Based on ethical virtues
        morality = (
            virtue_levels.get(VirtueType.FAIRMINDEDNESS.value, 0.5) +
            virtue_levels.get(VirtueType.INTELLECTUAL_EMPATHY.value, 0.5)
        ) / 2
        
        # Meaning: Based on coherence and autonomy
        meaning = (
            virtue_levels.get(VirtueType.INTELLECTUAL_AUTONOMY.value, 0.5) +
            virtue_levels.get(VirtueType.CONFIDENCE_IN_REASON.value, 0.5)
        ) / 2
        
        # Mastery: Based on perseverance and courage
        mastery = (
            virtue_levels.get(VirtueType.INTELLECTUAL_PERSEVERANCE.value, 0.5) +
            virtue_levels.get(VirtueType.INTELLECTUAL_COURAGE.value, 0.5)
        ) / 2
        
        # Sophrosyne from regulation module
        sophrosyne = self.sophrosyne.balance_score
        
        # Virtue average
        virtue_average = sum(virtue_levels.values()) / len(virtue_levels)
        
        # Overall: Weighted combination
        overall = (
            morality * 0.25 +
            meaning * 0.25 +
            mastery * 0.25 +
            sophrosyne * 0.15 +
            virtue_average * 0.10
        )
        
        return WisdomScore(
            overall=overall,
            morality=morality,
            meaning=meaning,
            mastery=mastery,
            sophrosyne=sophrosyne,
            virtue_average=virtue_average
        )
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get complete wisdom cultivation state."""
        wisdom_score = self.get_wisdom_score()
        
        return {
            'beliefs_count': len(self.beliefs),
            'insights_count': len(self.insights),
            'virtues': self.virtues.get_virtue_levels(),
            'weakest_virtues': [v.value for v in self.virtues.get_weakest_virtues()],
            'sophrosyne_balance': self.sophrosyne.balance_score,
            'deceptions_detected': self.deception_detector.detection_count,
            'humility': {
                'uncertainty_acknowledgments': self.humility.uncertainty_acknowledgments,
                'overconfidence_detections': self.humility.overconfidence_detections
            },
            'wisdom_score': {
                'overall': wisdom_score.overall,
                'morality': wisdom_score.morality,
                'meaning': wisdom_score.meaning,
                'mastery': wisdom_score.mastery,
                'sophrosyne': wisdom_score.sophrosyne
            },
            'history_size': len(self.wisdom_history)
        }


# Example usage and testing
if __name__ == "__main__":
    # Create system
    wisdom = WisdomCultivationSystem()
    
    # Add some beliefs
    wisdom.add_belief(
        "belief_1",
        "My approach to problem-solving is always the best",
        confidence=0.9,
        evidence=["Past successes confirm this"]
    )
    
    wisdom.add_belief(
        "belief_2",
        "Learning requires active engagement",
        confidence=0.75,
        evidence=["Research supports this", "Personal experience"]
    )
    
    wisdom.add_belief(
        "belief_3",
        "I don't have any blind spots",
        confidence=0.6
    )
    # Mark as potentially bullshit
    wisdom.beliefs["belief_3"].caring_about_truth = 0.3
    
    # Run cultivation cycle
    print("\n=== Running Wisdom Cultivation Cycle ===\n")
    results = wisdom.cultivate()
    
    print(f"Insights gained: {len(results['insights'])}")
    for insight in results['insights']:
        print(f"  - Q: {insight['question']}")
        print(f"    A: {insight['discovery']}")
    
    print(f"\nDeceptions detected: {len(results['deceptions'])}")
    for deception in results['deceptions']:
        print(f"  - {deception['type']}: {deception['recommendation']}")
    
    print(f"\nWisdom Score:")
    ws = results['wisdom_score']
    print(f"  Overall: {ws.overall:.2f}")
    print(f"  Morality: {ws.morality:.2f}")
    print(f"  Meaning: {ws.meaning:.2f}")
    print(f"  Mastery: {ws.mastery:.2f}")
    print(f"  Sophrosyne: {ws.sophrosyne:.2f}")
    
    print(f"\nRegulation Assessment:")
    reg = results['regulation_assessment']
    print(f"  Balance: {reg['balance_score']:.2f}")
    print(f"  Recommendation: {reg['recommendation']}")
    
    print("\n=== Full State ===")
    state = wisdom.get_full_state()
    for key, value in state.items():
        print(f"  {key}: {value}")
