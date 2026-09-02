"""
Optimal Cognitive Grip Engine
==============================

Implements the optimization of cognitive grip—the dynamic ability to achieve
relevance realization across all epistemic modalities through opponent
processing, circular causality, and multi-constraint satisfaction.

Core Formulation:
    Grip(t) = argmax_θ [
        Relevance(θ, Context(t)) * 
        Predictive_Power(θ) * 
        Cognitive_Economy(θ) - 
        Uncertainty_Cost(θ)
    ]

The grip optimization integrates with:
- Relevance Realization Engine
- Virtual Embodiment Layer
- Adaptive Dimensional Embedding
- Hypergraph Cognitive Encoding

Author: Deep Tree Echo
Date: June 2026
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GripMode(Enum):
    """Modes of cognitive grip optimization."""
    EXPLORATIVE = "explorative"      # Broader, more uncertain grip
    EXPLOITATIVE = "exploitative"    # Focused, high-certainty grip
    BALANCED = "balanced"            # Dynamic balance
    TRANSFORMATIVE = "transformative"  # Deep restructuring grip


@dataclass
class GripState:
    """Current state of cognitive grip."""
    mode: GripMode = GripMode.BALANCED
    relevance_score: float = 0.5
    predictive_power: float = 0.5
    cognitive_economy: float = 0.5
    uncertainty_cost: float = 0.2
    
    # Grip quality metrics
    overall_grip: float = 0.5
    stability: float = 0.5
    adaptivity: float = 0.5
    
    # History
    grip_history: List[float] = field(default_factory=list)


@dataclass
class RelevanceContext:
    """Context for relevance computation."""
    goals: List[Dict[str, Any]] = field(default_factory=list)
    predictions: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=dict)
    embodiment_state: Dict[str, Any] = field(default_factory=dict)


class OpponentProcess:
    """
    Implements dialectical balance between opposing cognitive tendencies.
    
    Core opponent pairs:
    - Exploration ↔ Exploitation
    - Breadth ↔ Depth
    - Speed ↔ Accuracy
    - Certainty ↔ Openness
    """
    
    def __init__(
        self, 
        name: str,
        pole_a: str, 
        pole_b: str, 
        initial_balance: float = 0.5
    ):
        """
        Initialize opponent process.
        
        Args:
            name: Process name
            pole_a: First pole (at balance=0)
            pole_b: Second pole (at balance=1)
            initial_balance: Starting balance [0, 1]
        """
        self.name = name
        self.pole_a = pole_a
        self.pole_b = pole_b
        self.balance = np.clip(initial_balance, 0.0, 1.0)
        self.history = deque(maxlen=100)
        self.momentum = 0.0
        
    def shift(self, delta: float) -> None:
        """Shift balance by delta (-1.0 to 1.0)."""
        # Apply momentum for smoother transitions
        self.momentum = 0.9 * self.momentum + 0.1 * delta
        old_balance = self.balance
        self.balance = np.clip(self.balance + self.momentum, 0.0, 1.0)
        self.history.append(self.balance)
        
        logger.debug(f"{self.name}: {old_balance:.3f} → {self.balance:.3f}")
    
    def shift_toward_a(self, amount: float = 0.1) -> None:
        """Shift toward first pole."""
        self.shift(-amount)
    
    def shift_toward_b(self, amount: float = 0.1) -> None:
        """Shift toward second pole."""
        self.shift(amount)
    
    def get_weights(self) -> Tuple[float, float]:
        """Get current weights for both poles."""
        return (1.0 - self.balance, self.balance)
    
    def auto_adjust(self, context: RelevanceContext) -> None:
        """Automatically adjust based on context."""
        # Context-sensitive adjustment
        if self.name == "exploration_exploitation":
            # High novelty needed → explore
            if context.goals and any(g.get('novelty_required', False) for g in context.goals):
                self.shift_toward_a(0.05)
            # High certainty needed → exploit
            if context.goals and any(g.get('certainty_required', False) for g in context.goals):
                self.shift_toward_b(0.05)
                
        elif self.name == "speed_accuracy":
            # Time pressure → speed
            if context.resources.get('time_remaining', 1.0) < 0.3:
                self.shift_toward_a(0.05)
            # Critical decision → accuracy
            if any(c.get('critical', False) for c in context.constraints):
                self.shift_toward_b(0.05)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        return {
            'name': self.name,
            'pole_a': self.pole_a,
            'pole_b': self.pole_b,
            'balance': self.balance,
            'momentum': self.momentum,
            'weights': self.get_weights()
        }


class CostFunction:
    """
    Cost function for relevance optimization.
    
    Evaluates possibilities against multiple criteria to determine
    their relevance in the current context.
    """
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        
    def score(self, possibility: Dict[str, Any], context: RelevanceContext) -> float:
        """Score how well possibility satisfies this cost function."""
        if self.name == "goal_alignment":
            return self._score_goal_alignment(possibility, context)
        elif self.name == "predictive_power":
            return self._score_predictive_power(possibility, context)
        elif self.name == "cognitive_economy":
            return self._score_cognitive_economy(possibility, context)
        elif self.name == "novelty_value":
            return self._score_novelty_value(possibility, context)
        elif self.name == "contextual_fit":
            return self._score_contextual_fit(possibility, context)
        else:
            return 0.5
    
    def _score_goal_alignment(self, p: Dict, ctx: RelevanceContext) -> float:
        """How well does this possibility advance current goals?"""
        if not ctx.goals:
            return 0.5
        
        alignment = 0.0
        for goal in ctx.goals:
            goal_vector = goal.get('vector', np.zeros(10))
            poss_vector = p.get('vector', np.zeros(10))
            
            # Ensure same dimensions
            min_dim = min(len(goal_vector), len(poss_vector))
            if min_dim > 0:
                # Cosine similarity
                dot = np.dot(goal_vector[:min_dim], poss_vector[:min_dim])
                norm_g = np.linalg.norm(goal_vector[:min_dim])
                norm_p = np.linalg.norm(poss_vector[:min_dim])
                if norm_g > 0 and norm_p > 0:
                    alignment += (dot / (norm_g * norm_p) + 1) / 2
                else:
                    alignment += 0.5
            else:
                alignment += 0.5
        
        return np.clip(alignment / len(ctx.goals), 0.0, 1.0)
    
    def _score_predictive_power(self, p: Dict, ctx: RelevanceContext) -> float:
        """Does attending to this improve future predictions?"""
        # Information gain heuristic
        if not ctx.predictions:
            return 0.5
        
        # Estimate reduction in uncertainty
        current_entropy = ctx.predictions.get('entropy', 0.5)
        expected_reduction = p.get('info_gain', 0.0)
        
        return np.clip(expected_reduction / max(current_entropy, 0.01), 0.0, 1.0)
    
    def _score_cognitive_economy(self, p: Dict, ctx: RelevanceContext) -> float:
        """Is this efficient to process?"""
        # Resource cost vs benefit
        processing_cost = p.get('processing_cost', 0.5)
        expected_benefit = p.get('expected_benefit', 0.5)
        
        if processing_cost == 0:
            return 1.0
        
        roi = expected_benefit / processing_cost
        return np.clip(roi / 2.0, 0.0, 1.0)  # Normalize to [0, 1]
    
    def _score_novelty_value(self, p: Dict, ctx: RelevanceContext) -> float:
        """Does this provide new information?"""
        familiarity = p.get('familiarity', 0.5)
        return 1.0 - familiarity  # Higher novelty = lower familiarity
    
    def _score_contextual_fit(self, p: Dict, ctx: RelevanceContext) -> float:
        """Does this fit the current context?"""
        # Constraint satisfaction
        if not ctx.constraints:
            return 0.5
        
        satisfied = 0
        for constraint in ctx.constraints:
            if self._satisfies_constraint(p, constraint):
                satisfied += 1
        
        return satisfied / len(ctx.constraints)
    
    def _satisfies_constraint(self, p: Dict, constraint: Dict) -> bool:
        """Check if possibility satisfies constraint."""
        constraint_type = constraint.get('type', 'range')
        
        if constraint_type == 'range':
            key = constraint.get('key', '')
            value = p.get(key, 0)
            min_val = constraint.get('min', float('-inf'))
            max_val = constraint.get('max', float('inf'))
            return min_val <= value <= max_val
        
        elif constraint_type == 'must_have':
            key = constraint.get('key', '')
            return key in p and p[key]
        
        return True


class CircularCausality:
    """
    Tracks and manages circular causality between relevance and processing.
    
    - Processing shapes relevance (what we process becomes more relevant)
    - Relevance shapes processing (what's relevant gets processed more)
    """
    
    def __init__(self):
        self.processing_history: List[Dict] = []
        self.relevance_updates: List[Dict] = []
        self.causality_strength = 0.5
        
    def record_processing(self, item: Dict[str, Any], outcome: Dict[str, Any]):
        """Record that an item was processed."""
        self.processing_history.append({
            'item': item,
            'outcome': outcome,
            'timestamp': len(self.processing_history)
        })
        
        # Processing affects future relevance
        self._update_relevance_from_processing(item, outcome)
    
    def _update_relevance_from_processing(self, item: Dict, outcome: Dict):
        """Update relevance based on processing outcome."""
        success = outcome.get('success', 0.5)
        
        # Successful processing increases relevance of similar items
        update = {
            'item_type': item.get('type', 'unknown'),
            'relevance_delta': success * self.causality_strength,
            'source': 'processing'
        }
        self.relevance_updates.append(update)
    
    def get_relevance_modifier(self, item: Dict) -> float:
        """Get relevance modifier based on processing history."""
        item_type = item.get('type', 'unknown')
        
        modifier = 0.0
        for update in self.relevance_updates[-50:]:  # Recent updates
            if update['item_type'] == item_type:
                modifier += update['relevance_delta']
        
        return np.clip(modifier, -0.5, 0.5)
    
    def adjust_causality_strength(self, feedback: float):
        """Adjust strength of circular causality based on feedback."""
        self.causality_strength = np.clip(
            self.causality_strength + feedback * 0.1,
            0.1, 0.9
        )


class CognitiveGripOptimizer:
    """
    Main optimizer for cognitive grip.
    
    Integrates:
    - Opponent processes (dialectical balance)
    - Cost functions (relevance criteria)
    - Circular causality (processing ↔ relevance)
    - Constraint satisfaction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.state = GripState()
        
        # Initialize opponent processes
        self.opponents = {
            'exploration_exploitation': OpponentProcess(
                'exploration_exploitation', 'exploration', 'exploitation'
            ),
            'breadth_depth': OpponentProcess(
                'breadth_depth', 'breadth', 'depth'
            ),
            'speed_accuracy': OpponentProcess(
                'speed_accuracy', 'speed', 'accuracy'
            ),
            'certainty_openness': OpponentProcess(
                'certainty_openness', 'certainty', 'openness'
            )
        }
        
        # Initialize cost functions
        weights = self.config.get('cost_weights', {
            'goal_alignment': 0.30,
            'predictive_power': 0.25,
            'cognitive_economy': 0.20,
            'novelty_value': 0.15,
            'contextual_fit': 0.10
        })
        
        self.cost_functions = {
            name: CostFunction(name, weight)
            for name, weight in weights.items()
        }
        
        # Circular causality tracker
        self.circular_causality = CircularCausality()
        
        logger.info("CognitiveGripOptimizer initialized")
    
    def compute_grip(
        self,
        possibilities: List[Dict[str, Any]],
        context: RelevanceContext
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Compute optimal grip over possibilities.
        
        Args:
            possibilities: List of possible focus points
            context: Current relevance context
            
        Returns:
            Tuple of (ranked possibilities, overall grip quality)
        """
        # Auto-adjust opponent processes
        for opponent in self.opponents.values():
            opponent.auto_adjust(context)
        
        # Score each possibility
        scored = []
        for poss in possibilities:
            score = self._score_possibility(poss, context)
            scored.append({**poss, '_grip_score': score})
        
        # Sort by grip score
        scored.sort(key=lambda x: x['_grip_score'], reverse=True)
        
        # Compute overall grip quality
        if scored:
            top_scores = [p['_grip_score'] for p in scored[:5]]
            grip_quality = np.mean(top_scores)
        else:
            grip_quality = 0.0
        
        # Update state
        self.state.overall_grip = grip_quality
        self.state.grip_history.append(grip_quality)
        
        # Compute stability and adaptivity
        self._update_grip_metrics()
        
        return scored, grip_quality
    
    def _score_possibility(self, poss: Dict, context: RelevanceContext) -> float:
        """Score a single possibility."""
        total_score = 0.0
        total_weight = 0.0
        
        # Apply cost functions
        for name, cost_fn in self.cost_functions.items():
            score = cost_fn.score(poss, context)
            weighted_score = score * cost_fn.weight
            total_score += weighted_score
            total_weight += cost_fn.weight
        
        # Normalize
        if total_weight > 0:
            base_score = total_score / total_weight
        else:
            base_score = 0.5
        
        # Apply circular causality modifier
        causality_mod = self.circular_causality.get_relevance_modifier(poss)
        base_score += causality_mod
        
        # Apply opponent process modifiers
        opponent_mod = self._get_opponent_modifier(poss)
        base_score += opponent_mod
        
        return np.clip(base_score, 0.0, 1.0)
    
    def _get_opponent_modifier(self, poss: Dict) -> float:
        """Get modifier from opponent processes."""
        modifier = 0.0
        
        # Exploration-exploitation
        exp_exploit = self.opponents['exploration_exploitation']
        w_explore, w_exploit = exp_exploit.get_weights()
        novelty = poss.get('novelty', 0.5)
        reliability = poss.get('reliability', 0.5)
        modifier += w_explore * novelty * 0.1 + w_exploit * reliability * 0.1
        
        # Breadth-depth
        bd = self.opponents['breadth_depth']
        w_breadth, w_depth = bd.get_weights()
        breadth_score = poss.get('breadth', 0.5)
        depth_score = poss.get('depth', 0.5)
        modifier += w_breadth * breadth_score * 0.1 + w_depth * depth_score * 0.1
        
        return modifier
    
    def _update_grip_metrics(self):
        """Update grip stability and adaptivity metrics."""
        history = self.state.grip_history[-20:]
        
        if len(history) >= 5:
            # Stability: low variance in recent grip
            self.state.stability = 1.0 - min(1.0, np.std(history) * 2)
            
            # Adaptivity: correlation with context changes
            if len(history) >= 10:
                recent = np.array(history[-10:])
                older = np.array(history[-20:-10])
                self.state.adaptivity = 1.0 - abs(np.mean(recent) - np.mean(older))
    
    def record_processing_outcome(
        self,
        item: Dict[str, Any],
        outcome: Dict[str, Any]
    ):
        """Record outcome of processing for circular causality."""
        self.circular_causality.record_processing(item, outcome)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current optimizer state."""
        return {
            'mode': self.state.mode.value,
            'overall_grip': self.state.overall_grip,
            'stability': self.state.stability,
            'adaptivity': self.state.adaptivity,
            'opponents': {k: v.get_state() for k, v in self.opponents.items()},
            'causality_strength': self.circular_causality.causality_strength
        }


class OptimalGrip:
    """
    High-level interface for optimal cognitive grip.
    
    Provides unified access to grip optimization, context management,
    and integration with other Echogenesis components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.optimizer = CognitiveGripOptimizer(config)
        self.context = RelevanceContext()
        
        # Integration hooks
        self.embedding_hook: Optional[Callable] = None
        self.embodiment_hook: Optional[Callable] = None
        
        logger.info("OptimalGrip initialized")
    
    def set_context(
        self,
        goals: Optional[List[Dict]] = None,
        predictions: Optional[Dict] = None,
        constraints: Optional[List[Dict]] = None,
        resources: Optional[Dict] = None,
        embodiment_state: Optional[Dict] = None
    ):
        """Update relevance context."""
        if goals is not None:
            self.context.goals = goals
        if predictions is not None:
            self.context.predictions = predictions
        if constraints is not None:
            self.context.constraints = constraints
        if resources is not None:
            self.context.resources = resources
        if embodiment_state is not None:
            self.context.embodiment_state = embodiment_state
    
    def realize_relevance(
        self,
        possibilities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Realize relevance across possibilities.
        
        Main entry point for relevance realization. Computes optimal grip
        and returns ranked possibilities.
        
        Args:
            possibilities: List of possible focus points
            
        Returns:
            Ranked list of possibilities with grip scores
        """
        ranked, grip_quality = self.optimizer.compute_grip(
            possibilities, self.context
        )
        
        logger.debug(f"Realized relevance with grip quality: {grip_quality:.3f}")
        
        return ranked
    
    def filter_by_threshold(
        self,
        possibilities: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Filter possibilities by grip threshold."""
        ranked = self.realize_relevance(possibilities)
        return [p for p in ranked if p.get('_grip_score', 0) >= threshold]
    
    def get_top_k(
        self,
        possibilities: List[Dict[str, Any]],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top-k most relevant possibilities."""
        ranked = self.realize_relevance(possibilities)
        return ranked[:k]
    
    def feedback(self, item: Dict, success: float):
        """Provide feedback on processing outcome."""
        self.optimizer.record_processing_outcome(
            item, 
            {'success': success}
        )
    
    def connect_embedding(self, embedding_func: Callable):
        """Connect to adaptive embedding system."""
        self.embedding_hook = embedding_func
    
    def connect_embodiment(self, embodiment_func: Callable):
        """Connect to virtual embodiment system."""
        self.embodiment_hook = embodiment_func
    
    def get_grip_quality(self) -> float:
        """Get current grip quality."""
        return self.optimizer.state.overall_grip
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get full grip state."""
        return {
            'optimizer': self.optimizer.get_state(),
            'context': {
                'num_goals': len(self.context.goals),
                'num_constraints': len(self.context.constraints),
                'resources': self.context.resources
            }
        }


# Convenience functions
def create_optimal_grip(config: Optional[Dict[str, Any]] = None) -> OptimalGrip:
    """Factory function to create OptimalGrip instance."""
    return OptimalGrip(config)


def quick_relevance_filter(
    possibilities: List[Dict],
    goals: List[Dict],
    threshold: float = 0.5
) -> List[Dict]:
    """Quick relevance filtering without full initialization."""
    grip = OptimalGrip()
    grip.set_context(goals=goals)
    return grip.filter_by_threshold(possibilities, threshold)
