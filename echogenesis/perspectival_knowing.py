"""
Perspectival Knowing System
============================

Implements frame-switching mechanisms, aspect perception ("see-as" capability),
and gestalt shift detection for the Echogenesis architecture.

Core Capabilities:
- Frame switching: Dynamic perspective changes
- Salience landscapes: Context-dependent relevance maps
- Gestalt perception: Figure-ground dynamics
- Aspect seeing: Same data perceived differently
- Transformative shifts: Deep restructuring of perspective

This addresses the critical gap in perspectival knowing identified in the
cognitive architecture analysis - the ability to "see the same thing as
something different."

Author: Deep Tree Echo
Date: June 2026
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameType(Enum):
    """Types of cognitive frames."""
    ANALYTICAL = "analytical"      # Logical, systematic analysis
    INTUITIVE = "intuitive"        # Pattern-based, holistic
    EMPATHETIC = "empathetic"      # Emotional, interpersonal
    CREATIVE = "creative"          # Divergent, innovative
    PRACTICAL = "practical"        # Action-oriented, pragmatic
    PHILOSOPHICAL = "philosophical"  # Abstract, conceptual
    EMBODIED = "embodied"          # Sensorimotor, physical
    METACOGNITIVE = "metacognitive"  # Self-reflective


@dataclass
class SaliencePoint:
    """A point in the salience landscape."""
    id: str
    position: np.ndarray
    salience: float
    features: Dict[str, Any]
    is_figure: bool = True  # True = foreground, False = background


class SalienceLandscape:
    """
    Represents the salience landscape for a given frame.
    
    The salience landscape determines what stands out as relevant
    (figure) versus what recedes to background (ground).
    """
    
    def __init__(self, frame_type: FrameType):
        self.frame_type = frame_type
        self.points: Dict[str, SaliencePoint] = {}
        self.attention_center: Optional[np.ndarray] = None
        self.attention_radius: float = 1.0
        
        # Feature weights for this frame
        self.feature_weights: Dict[str, float] = self._init_weights()
        
    def _init_weights(self) -> Dict[str, float]:
        """Initialize feature weights based on frame type."""
        base_weights = {
            'novelty': 0.5,
            'relevance': 0.5,
            'urgency': 0.5,
            'complexity': 0.5,
            'emotional_valence': 0.5,
            'action_potential': 0.5,
            'logical_coherence': 0.5
        }
        
        # Frame-specific adjustments
        if self.frame_type == FrameType.ANALYTICAL:
            base_weights['logical_coherence'] = 0.9
            base_weights['complexity'] = 0.7
            base_weights['emotional_valence'] = 0.2
            
        elif self.frame_type == FrameType.INTUITIVE:
            base_weights['novelty'] = 0.8
            base_weights['emotional_valence'] = 0.7
            base_weights['logical_coherence'] = 0.3
            
        elif self.frame_type == FrameType.EMPATHETIC:
            base_weights['emotional_valence'] = 0.9
            base_weights['relevance'] = 0.7
            base_weights['logical_coherence'] = 0.3
            
        elif self.frame_type == FrameType.CREATIVE:
            base_weights['novelty'] = 0.9
            base_weights['complexity'] = 0.6
            base_weights['logical_coherence'] = 0.4
            
        elif self.frame_type == FrameType.PRACTICAL:
            base_weights['action_potential'] = 0.9
            base_weights['urgency'] = 0.8
            base_weights['novelty'] = 0.3
            
        elif self.frame_type == FrameType.EMBODIED:
            base_weights['action_potential'] = 0.8
            base_weights['urgency'] = 0.6
            base_weights['emotional_valence'] = 0.7
            
        elif self.frame_type == FrameType.METACOGNITIVE:
            base_weights['complexity'] = 0.8
            base_weights['relevance'] = 0.8
            base_weights['logical_coherence'] = 0.7
        
        return base_weights
    
    def add_point(
        self,
        id: str,
        position: np.ndarray,
        features: Dict[str, Any]
    ):
        """Add a point to the salience landscape."""
        salience = self._compute_salience(features)
        is_figure = salience > 0.5  # Above threshold = figure
        
        self.points[id] = SaliencePoint(
            id=id,
            position=position,
            salience=salience,
            features=features,
            is_figure=is_figure
        )
    
    def _compute_salience(self, features: Dict[str, Any]) -> float:
        """Compute salience score for features."""
        total = 0.0
        weight_sum = 0.0
        
        for key, weight in self.feature_weights.items():
            if key in features:
                value = features[key]
                if isinstance(value, (int, float)):
                    total += value * weight
                    weight_sum += weight
        
        if weight_sum > 0:
            return np.clip(total / weight_sum, 0.0, 1.0)
        return 0.5
    
    def focus_attention(self, center: np.ndarray, radius: float = 1.0):
        """Focus attention on a region."""
        self.attention_center = center
        self.attention_radius = radius
        
        # Recompute salience with attention focus
        for point in self.points.values():
            distance = np.linalg.norm(point.position - center)
            attention_factor = max(0, 1 - distance / radius)
            point.salience = np.clip(
                point.salience + attention_factor * 0.3, 0.0, 1.0
            )
            point.is_figure = point.salience > 0.5
    
    def get_figures(self) -> List[SaliencePoint]:
        """Get all figure (foreground) points."""
        return [p for p in self.points.values() if p.is_figure]
    
    def get_ground(self) -> List[SaliencePoint]:
        """Get all ground (background) points."""
        return [p for p in self.points.values() if not p.is_figure]
    
    def shift_figure_ground(self):
        """Perform gestalt shift - swap figure and ground."""
        for point in self.points.values():
            point.is_figure = not point.is_figure
            point.salience = 1.0 - point.salience


@dataclass
class Frame:
    """
    A cognitive frame represents a perspective or way of seeing.
    
    Frames determine:
    - What stands out as salient
    - How information is organized
    - What aspects are foregrounded
    - How meaning is constructed
    """
    name: str
    frame_type: FrameType
    salience_landscape: SalienceLandscape
    relevance_filter: Dict[str, float] = field(default_factory=dict)
    aspect_map: Dict[str, str] = field(default_factory=dict)  # data_type -> perceived_as
    active: bool = False
    
    def __post_init__(self):
        if not self.salience_landscape:
            self.salience_landscape = SalienceLandscape(self.frame_type)
    
    def perceive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perceive data through this frame."""
        perceived = {}
        
        for key, value in data.items():
            # Apply aspect mapping if available
            if key in self.aspect_map:
                perceived_key = self.aspect_map[key]
            else:
                perceived_key = key
            
            # Apply relevance filter
            if key in self.relevance_filter:
                if self.relevance_filter[key] < 0.3:
                    continue  # Filter out low-relevance data
            
            perceived[perceived_key] = value
        
        # Add frame metadata
        perceived['_frame'] = self.name
        perceived['_frame_type'] = self.frame_type.value
        
        return perceived


class GestaltDetector:
    """
    Detects gestalt shifts and pattern reorganization.
    
    Monitors for:
    - Figure-ground reversals
    - Pattern emergence
    - Sudden reorganization
    - Insight moments
    """
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.shift_threshold = 0.3
        
    def record_perception(self, perception: Dict[str, Any]):
        """Record a perception state."""
        self.history.append({
            'perception': perception,
            'hash': self._compute_hash(perception)
        })
        
    def _compute_hash(self, perception: Dict) -> int:
        """Compute perceptual hash for comparison."""
        # Simple hash based on key structure
        return hash(frozenset(perception.keys()))
    
    def detect_shift(self) -> Optional[Dict[str, Any]]:
        """Detect if a gestalt shift has occurred."""
        if len(self.history) < 2:
            return None
        
        current = self.history[-1]
        previous = self.history[-2]
        
        # Compare perceptual hashes
        if current['hash'] != previous['hash']:
            # Key structure changed - potential shift
            return {
                'type': 'structural_shift',
                'from_keys': set(previous['perception'].keys()),
                'to_keys': set(current['perception'].keys()),
                'added': set(current['perception'].keys()) - set(previous['perception'].keys()),
                'removed': set(previous['perception'].keys()) - set(current['perception'].keys())
            }
        
        # Check for value reversals
        reversals = []
        for key in current['perception']:
            if key in previous['perception']:
                if key.startswith('_'):
                    continue
                curr_val = current['perception'][key]
                prev_val = previous['perception'][key]
                if isinstance(curr_val, (int, float)) and isinstance(prev_val, (int, float)):
                    if abs(curr_val - prev_val) > self.shift_threshold:
                        reversals.append(key)
        
        if reversals:
            return {
                'type': 'value_shift',
                'shifted_features': reversals
            }
        
        return None


class AspectPerception:
    """
    Implements "see-as" capability - perceiving the same data as different things.
    
    Classic example: The duck-rabbit illusion
    - Same visual pattern can be seen AS a duck OR AS a rabbit
    - Not about different data, but different seeing
    """
    
    def __init__(self):
        self.aspect_library: Dict[str, List[str]] = {}  # data_pattern -> possible_aspects
        self.current_aspects: Dict[str, str] = {}  # data_id -> current_aspect
        
    def register_aspects(self, pattern_type: str, aspects: List[str]):
        """Register possible aspects for a pattern type."""
        self.aspect_library[pattern_type] = aspects
        
    def see_as(self, data_id: str, pattern_type: str, aspect: str) -> bool:
        """
        See data as a particular aspect.
        
        Args:
            data_id: Identifier for the data
            pattern_type: Type of pattern
            aspect: Aspect to perceive as
            
        Returns:
            True if aspect change succeeded
        """
        if pattern_type not in self.aspect_library:
            return False
        
        if aspect not in self.aspect_library[pattern_type]:
            return False
        
        self.current_aspects[data_id] = aspect
        return True
    
    def get_current_aspect(self, data_id: str) -> Optional[str]:
        """Get current aspect for data."""
        return self.current_aspects.get(data_id)
    
    def get_possible_aspects(self, pattern_type: str) -> List[str]:
        """Get all possible aspects for a pattern type."""
        return self.aspect_library.get(pattern_type, [])
    
    def flip_aspect(self, data_id: str, pattern_type: str) -> Optional[str]:
        """Flip to next possible aspect."""
        if pattern_type not in self.aspect_library:
            return None
        
        aspects = self.aspect_library[pattern_type]
        if len(aspects) < 2:
            return None
        
        current = self.current_aspects.get(data_id)
        if current in aspects:
            idx = aspects.index(current)
            next_idx = (idx + 1) % len(aspects)
            new_aspect = aspects[next_idx]
        else:
            new_aspect = aspects[0]
        
        self.current_aspects[data_id] = new_aspect
        return new_aspect


class PerspectivalKnowing:
    """
    Main interface for perspectival knowing capabilities.
    
    Integrates:
    - Frame management and switching
    - Salience landscape manipulation
    - Gestalt shift detection
    - Aspect perception
    """
    
    def __init__(self):
        self.frames: Dict[str, Frame] = {}
        self.current_frame: Optional[Frame] = None
        self.gestalt_detector = GestaltDetector()
        self.aspect_perception = AspectPerception()
        
        # Initialize default frames
        self._init_default_frames()
        
        # Transition history
        self.transition_history: List[Tuple[str, str]] = []
        
        logger.info("PerspectivalKnowing initialized")
    
    def _init_default_frames(self):
        """Initialize default cognitive frames."""
        for frame_type in FrameType:
            name = frame_type.value
            self.frames[name] = Frame(
                name=name,
                frame_type=frame_type,
                salience_landscape=SalienceLandscape(frame_type)
            )
        
        # Set default to balanced/analytical
        self.current_frame = self.frames['analytical']
        self.current_frame.active = True
    
    def add_frame(self, name: str, frame_type: FrameType, **kwargs):
        """Add a custom frame."""
        self.frames[name] = Frame(
            name=name,
            frame_type=frame_type,
            salience_landscape=SalienceLandscape(frame_type),
            **kwargs
        )
    
    def switch_frame(self, target_frame: str, context: Optional[Dict] = None) -> bool:
        """
        Switch to a different cognitive frame.
        
        Args:
            target_frame: Name of frame to switch to
            context: Optional context for the switch
            
        Returns:
            True if switch succeeded
        """
        if target_frame not in self.frames:
            logger.warning(f"Unknown frame: {target_frame}")
            return False
        
        old_frame = self.current_frame.name if self.current_frame else None
        
        # Deactivate current frame
        if self.current_frame:
            self.current_frame.active = False
        
        # Activate new frame
        self.current_frame = self.frames[target_frame]
        self.current_frame.active = True
        
        # Perform gestalt shift
        self._perform_gestalt_shift(old_frame, target_frame, context)
        
        # Record transition
        if old_frame:
            self.transition_history.append((old_frame, target_frame))
        
        logger.info(f"Switched frame: {old_frame} → {target_frame}")
        return True
    
    def _perform_gestalt_shift(
        self, 
        old_frame: Optional[str], 
        new_frame: str,
        context: Optional[Dict]
    ):
        """Execute the gestalt shift between frames."""
        if not old_frame:
            return
        
        old = self.frames.get(old_frame)
        new = self.frames.get(new_frame)
        
        if not old or not new:
            return
        
        # Transfer attention if context provided
        if context and 'focus' in context:
            focus = context['focus']
            if isinstance(focus, np.ndarray):
                new.salience_landscape.focus_attention(focus)
    
    def perceive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perceive data through current frame."""
        if not self.current_frame:
            return data
        
        perception = self.current_frame.perceive(data)
        
        # Record for gestalt detection
        self.gestalt_detector.record_perception(perception)
        
        # Check for spontaneous shifts
        shift = self.gestalt_detector.detect_shift()
        if shift:
            perception['_gestalt_shift'] = shift
        
        return perception
    
    def see_as(
        self, 
        data: Dict[str, Any], 
        aspect: str,
        pattern_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        See data as a particular aspect.
        
        The core of perspectival knowing: the same data can be
        perceived differently without changing the data itself.
        
        Args:
            data: Data to perceive
            aspect: Aspect to perceive as
            pattern_type: Optional pattern type for aspect lookup
            
        Returns:
            Data perceived through the aspect
        """
        data_id = str(hash(frozenset(data.items())))
        
        # Try to set aspect
        if pattern_type:
            self.aspect_perception.see_as(data_id, pattern_type, aspect)
        
        # Perceive with aspect
        perceived = {**data}
        perceived['_aspect'] = aspect
        perceived['_seeing_as'] = True
        
        return perceived
    
    def get_available_aspects(self, pattern_type: str) -> List[str]:
        """Get available aspects for a pattern type."""
        return self.aspect_perception.get_possible_aspects(pattern_type)
    
    def register_aspect_pattern(self, pattern_type: str, aspects: List[str]):
        """Register a pattern type with its possible aspects."""
        self.aspect_perception.register_aspects(pattern_type, aspects)
    
    def get_current_frame(self) -> Optional[str]:
        """Get name of current frame."""
        return self.current_frame.name if self.current_frame else None
    
    def get_frame_types(self) -> List[str]:
        """Get all available frame types."""
        return list(self.frames.keys())
    
    def get_salience_landscape(self) -> Optional[SalienceLandscape]:
        """Get current frame's salience landscape."""
        if self.current_frame:
            return self.current_frame.salience_landscape
        return None
    
    def add_to_landscape(
        self,
        id: str,
        position: np.ndarray,
        features: Dict[str, Any]
    ):
        """Add a point to current frame's salience landscape."""
        if self.current_frame:
            self.current_frame.salience_landscape.add_point(id, position, features)
    
    def get_figures(self) -> List[SaliencePoint]:
        """Get figure (foreground) elements in current frame."""
        if self.current_frame:
            return self.current_frame.salience_landscape.get_figures()
        return []
    
    def get_ground(self) -> List[SaliencePoint]:
        """Get ground (background) elements in current frame."""
        if self.current_frame:
            return self.current_frame.salience_landscape.get_ground()
        return []
    
    def flip_figure_ground(self):
        """Flip figure-ground in current frame."""
        if self.current_frame:
            self.current_frame.salience_landscape.shift_figure_ground()
            logger.info("Figure-ground flipped")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current perspectival knowing state."""
        return {
            'current_frame': self.current_frame.name if self.current_frame else None,
            'frame_type': self.current_frame.frame_type.value if self.current_frame else None,
            'available_frames': list(self.frames.keys()),
            'num_figures': len(self.get_figures()),
            'num_ground': len(self.get_ground()),
            'transition_count': len(self.transition_history)
        }


# Convenience functions
def create_perspectival_knowing() -> PerspectivalKnowing:
    """Factory function to create PerspectivalKnowing instance."""
    return PerspectivalKnowing()


def quick_frame_switch(
    perceiver: PerspectivalKnowing,
    data: Dict[str, Any],
    target_frame: str
) -> Dict[str, Any]:
    """Quick frame switch and perception."""
    perceiver.switch_frame(target_frame)
    return perceiver.perceive(data)
