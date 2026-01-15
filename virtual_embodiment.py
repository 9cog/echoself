"""
Virtual Embodiment Layer for EchoSelf
======================================

Provides sensorimotor grounding for cognitive architecture through
virtual embodiment, addressing the critical gap in 4E cognition.

Core Insight:
Meaning emerges from embodied interaction with environment, not from
abstract symbol manipulation alone. This layer grounds all higher-order
cognition in action-perception loops.

Key Components:
- Sensorimotor contingencies (if I do X, I perceive Y)
- Body schema (sense of extent and capabilities)
- Affordance detection (what actions are possible)
- Forward/inverse models (predict and plan)
- Proprioception (internal state awareness)

Author: Deep Tree Echo
Date: November 2025
"""

from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Sensory modalities"""
    VISION = "vision"
    AUDIO = "audio"
    TOUCH = "touch"
    PROPRIOCEPTION = "proprioception"
    INTEROCEPTION = "interoception"  # Internal body state


@dataclass
class SensoryInput:
    """Multimodal sensory input"""
    modality: ModalityType
    data: np.ndarray
    timestamp: float
    location: Optional[np.ndarray] = None  # Where in body/space
    
    def __repr__(self):
        return f"SensoryInput({self.modality.value}, shape={self.data.shape})"


@dataclass
class MotorCommand:
    """Motor action command"""
    action_type: str  # e.g., "move", "grasp", "look"
    parameters: Dict[str, Any]
    expected_duration: float = 0.1
    
    def __repr__(self):
        return f"MotorCommand({self.action_type}, {self.parameters})"


@dataclass
class SensoriMotorContingency:
    """
    Learned relationship between action and perception.
    Core of enacted cognition: "If I do X, I perceive Y"
    """
    action: MotorCommand
    expected_perception: SensoryInput
    confidence: float = 0.5
    usage_count: int = 0
    
    def update(self, actual_perception: SensoryInput, learning_rate: float = 0.1):
        """Update contingency based on actual outcome"""
        # Compute prediction error
        error = np.mean((self.expected_perception.data - actual_perception.data) ** 2)
        
        # Update confidence based on error
        self.confidence = (1 - learning_rate) * self.confidence + learning_rate * (1.0 / (1.0 + error))
        self.usage_count += 1


class BodySchema:
    """
    Representation of body structure, extent, and capabilities.
    Enables proprioception and action planning.
    """
    
    def __init__(self):
        # Body parts and their properties
        self.parts = {
            'head': {'position': np.array([0.0, 0.0, 1.8]), 'dof': 3},
            'torso': {'position': np.array([0.0, 0.0, 1.5]), 'dof': 3},
            'left_arm': {'position': np.array([-0.4, 0.0, 1.5]), 'dof': 7},
            'right_arm': {'position': np.array([0.4, 0.0, 1.5]), 'dof': 7},
        }
        
        # Sensory field for each body part
        self.sensory_fields = {
            part: {'range': 1.0, 'active': True} 
            for part in self.parts
        }
        
        # Internal state
        self.posture = self._initialize_posture()
        self.energy_level = 1.0
        self.damage = {}
        
    def _initialize_posture(self) -> Dict[str, np.ndarray]:
        """Initialize default posture"""
        return {part: np.zeros(info['dof']) for part, info in self.parts.items()}
    
    def get_part_position(self, part: str) -> np.ndarray:
        """Get current position of body part in world space"""
        if part not in self.parts:
            raise ValueError(f"Unknown body part: {part}")
        return self.parts[part]['position']
    
    def update_posture(self, part: str, joint_angles: np.ndarray):
        """Update posture for a body part"""
        if part in self.posture:
            self.posture[part] = joint_angles
            # Update derived positions based on forward kinematics
            self._update_forward_kinematics(part)
    
    def _update_forward_kinematics(self, part: str):
        """Update world positions based on joint angles"""
        # Simplified - would use proper FK in real implementation
        pass
    
    def get_reachable_space(self, part: str = 'right_arm') -> np.ndarray:
        """Get workspace reachable by body part"""
        # Simplified - return sphere around part
        center = self.get_part_position(part)
        radius = 0.8  # Arm reach
        return {'center': center, 'radius': radius}
    
    def is_damaged(self, part: str) -> bool:
        """Check if body part is damaged"""
        return part in self.damage and self.damage[part] > 0.5


class AffordanceDetector:
    """
    Detects action possibilities in environment based on body capabilities.
    
    Affordances are relational: what CAN BE DONE given body + environment.
    Gibson's ecological psychology core concept.
    """
    
    def __init__(self, body_schema: BodySchema):
        self.body_schema = body_schema
        self.known_affordances = {}
        
    def detect_affordances(self, environment_state: Dict[str, Any]) -> List[Dict]:
        """
        Detect what actions are possible given current environment.
        
        Returns list of affordances with:
        - action_type: What can be done
        - parameters: How to do it
        - utility: Why it matters
        """
        affordances = []
        
        # Example: Object within reach
        if 'objects' in environment_state:
            for obj in environment_state['objects']:
                obj_pos = obj.get('position', np.zeros(3))
                
                # Check if reachable
                for part in ['left_arm', 'right_arm']:
                    reachable = self._is_reachable(obj_pos, part)
                    
                    if reachable:
                        # Can grasp it
                        affordances.append({
                            'action_type': 'grasp',
                            'parameters': {
                                'object': obj['id'],
                                'effector': part,
                                'position': obj_pos
                            },
                            'utility': obj.get('value', 0.5),
                            'body_part': part
                        })
        
        # Example: Movable space
        if 'terrain' in environment_state:
            walkable_regions = environment_state['terrain'].get('walkable', [])
            for region in walkable_regions:
                affordances.append({
                    'action_type': 'move_to',
                    'parameters': {'target': region['center']},
                    'utility': self._navigation_utility(region),
                    'body_part': 'torso'
                })
        
        return affordances
    
    def _is_reachable(self, position: np.ndarray, body_part: str) -> bool:
        """Check if position is within reach of body part"""
        workspace = self.body_schema.get_reachable_space(body_part)
        distance = np.linalg.norm(position - workspace['center'])
        return distance <= workspace['radius']
    
    def _navigation_utility(self, region: Dict) -> float:
        """Evaluate utility of navigating to region"""
        # Factors: exploration value, goal alignment, safety
        return 0.5  # Placeholder


class ForwardModel:
    """
    Predicts sensory consequences of actions.
    Core of anticipatory cognition: "If I do X, I'll perceive Y"
    """
    
    def __init__(self):
        self.model = {}  # Maps (state, action) -> predicted_next_state
        
    def predict(
        self, 
        current_state: Dict[str, Any], 
        action: MotorCommand
    ) -> Tuple[Dict[str, Any], float]:
        """
        Predict next state and sensory input from action.
        
        Returns:
            (predicted_state, confidence)
        """
        # Simplified model - would use learned dynamics
        predicted_state = current_state.copy()
        
        # Example: Movement affects position
        if action.action_type == "move":
            delta = action.parameters.get('delta', np.zeros(3))
            current_pos = current_state.get('position', np.zeros(3))
            predicted_state['position'] = current_pos + delta
        
        confidence = 0.7  # Placeholder
        return predicted_state, confidence
    
    def update(
        self,
        state: Dict[str, Any],
        action: MotorCommand,
        actual_next_state: Dict[str, Any]
    ):
        """Learn from prediction errors"""
        predicted, _ = self.predict(state, action)
        
        # Compute error
        error = self._state_difference(predicted, actual_next_state)
        
        # Update model (simplified - would use gradient descent or similar)
        key = (self._state_key(state), self._action_key(action))
        if key not in self.model:
            self.model[key] = {'prediction': actual_next_state, 'confidence': 0.5}
        
        # Improve model
        self.model[key]['prediction'] = self._interpolate_states(
            self.model[key]['prediction'], 
            actual_next_state,
            alpha=0.1
        )
    
    def _state_difference(self, s1: Dict, s2: Dict) -> float:
        """Compute difference between states"""
        # Simplified metric
        return 0.1  # Placeholder
    
    def _state_key(self, state: Dict) -> str:
        """Create hashable key for state"""
        # Simplified - would use state hashing
        return str(sorted(state.keys()))
    
    def _action_key(self, action: MotorCommand) -> str:
        """Create hashable key for action"""
        return f"{action.action_type}_{sorted(action.parameters.keys())}"
    
    def _interpolate_states(self, s1: Dict, s2: Dict, alpha: float) -> Dict:
        """Interpolate between states for learning"""
        # Simplified
        return s2


class InverseModel:
    """
    Determines actions needed to achieve desired sensory outcomes.
    Core of goal-directed behavior: "To perceive Y, I should do X"
    """
    
    def __init__(self):
        self.model = {}  # Maps (state, desired_next_state) -> action
        
    def plan_action(
        self,
        current_state: Dict[str, Any],
        desired_state: Dict[str, Any]
    ) -> Optional[MotorCommand]:
        """
        Determine action that would achieve desired state.
        
        Returns:
            Motor command to execute, or None if not possible
        """
        # Check if we know how to achieve this
        key = (self._state_key(current_state), self._state_key(desired_state))
        
        if key in self.model:
            return self.model[key]['action']
        
        # Try to infer action
        action = self._infer_action(current_state, desired_state)
        return action
    
    def _infer_action(
        self, 
        current: Dict[str, Any], 
        desired: Dict[str, Any]
    ) -> Optional[MotorCommand]:
        """Infer action from state transition"""
        # Example: Position difference -> movement
        if 'position' in current and 'position' in desired:
            delta = desired['position'] - current['position']
            if np.linalg.norm(delta) > 0.01:
                return MotorCommand(
                    action_type="move",
                    parameters={'delta': delta}
                )
        
        return None
    
    def update(
        self,
        state: Dict[str, Any],
        action: MotorCommand,
        achieved_state: Dict[str, Any]
    ):
        """Learn inverse model from experience"""
        key = (self._state_key(state), self._state_key(achieved_state))
        self.model[key] = {'action': action, 'confidence': 0.8}
    
    def _state_key(self, state: Dict) -> str:
        """Create hashable key for state"""
        return str(sorted(state.keys()))


class SensoryBuffer:
    """
    Multimodal sensory buffer with active sampling capability.
    Not passive reception - active exploration.
    """
    
    def __init__(self, buffer_size: int = 100):
        self.buffer = {modality: deque(maxlen=buffer_size) 
                      for modality in ModalityType}
        self.attention_focus = None
        
    def add_input(self, sensory_input: SensoryInput):
        """Add sensory input to appropriate buffer"""
        self.buffer[sensory_input.modality].append(sensory_input)
    
    def sample(
        self, 
        modality: Optional[ModalityType] = None,
        time_window: float = 1.0
    ) -> List[SensoryInput]:
        """
        Sample from sensory buffer.
        
        Args:
            modality: Specific modality or None for all
            time_window: Recent time window to sample
        """
        import time
        current_time = time.time()
        
        if modality:
            buffer = self.buffer[modality]
            return [s for s in buffer 
                   if current_time - s.timestamp <= time_window]
        else:
            # Sample from all modalities
            samples = []
            for mod_buffer in self.buffer.values():
                samples.extend([s for s in mod_buffer 
                              if current_time - s.timestamp <= time_window])
            return samples
    
    def active_sample(self, focus: Dict[str, Any]) -> List[SensoryInput]:
        """
        Active sampling guided by attention/goals.
        This is enacted perception - not passive reception.
        """
        self.attention_focus = focus
        
        # Filter based on focus
        samples = []
        for modality, buffer in self.buffer.items():
            if focus.get(modality.value, False):
                # Attend to this modality
                samples.extend(list(buffer)[-10:])  # Recent samples
        
        return samples


class MotorController:
    """
    Executes motor primitives and composite actions.
    Bridge between planning and physical action.
    """
    
    def __init__(self, body_schema: BodySchema):
        self.body_schema = body_schema
        self.motor_primitives = self._initialize_primitives()
        self.execution_queue = deque()
        
    def _initialize_primitives(self) -> Dict[str, Callable]:
        """Initialize basic motor primitives"""
        return {
            'move': self._primitive_move,
            'grasp': self._primitive_grasp,
            'look': self._primitive_look,
            'reach': self._primitive_reach,
        }
    
    def execute(self, command: MotorCommand) -> Dict[str, Any]:
        """
        Execute motor command.
        
        Returns:
            Execution result with actual outcome
        """
        if command.action_type not in self.motor_primitives:
            logger.warning(f"Unknown action type: {command.action_type}")
            return {'success': False, 'error': 'unknown_action'}
        
        # Execute primitive
        primitive = self.motor_primitives[command.action_type]
        result = primitive(command.parameters)
        
        # Update body schema based on action
        self._update_body_state(command, result)
        
        return result
    
    def _primitive_move(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Basic movement primitive"""
        delta = params.get('delta', np.zeros(3))
        
        # Update body position (simplified)
        current_pos = self.body_schema.parts['torso']['position']
        new_pos = current_pos + delta
        self.body_schema.parts['torso']['position'] = new_pos
        
        return {
            'success': True,
            'new_position': new_pos,
            'duration': np.linalg.norm(delta) * 0.5  # Time proportional to distance
        }
    
    def _primitive_grasp(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Basic grasping primitive"""
        obj_id = params.get('object')
        effector = params.get('effector', 'right_arm')
        
        # Check if damaged
        if self.body_schema.is_damaged(effector):
            return {'success': False, 'error': 'effector_damaged'}
        
        return {
            'success': True,
            'grasped_object': obj_id,
            'effector': effector
        }
    
    def _primitive_look(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Basic looking/attention primitive"""
        target = params.get('target', np.zeros(3))
        
        # Update head orientation
        return {
            'success': True,
            'looking_at': target
        }
    
    def _primitive_reach(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Basic reaching primitive"""
        target = params.get('position', np.zeros(3))
        effector = params.get('effector', 'right_arm')
        
        # Check reachability
        reachable_space = self.body_schema.get_reachable_space(effector)
        distance = np.linalg.norm(target - reachable_space['center'])
        
        if distance > reachable_space['radius']:
            return {'success': False, 'error': 'unreachable'}
        
        return {
            'success': True,
            'reached_position': target,
            'effector': effector
        }
    
    def _update_body_state(self, command: MotorCommand, result: Dict):
        """Update body schema based on action result"""
        # Energy expenditure
        self.body_schema.energy_level *= 0.999
        
        # Update posture if needed
        if command.action_type == 'reach' and result.get('success'):
            effector = command.parameters.get('effector', 'right_arm')
            # Simplified posture update
            # Would compute IK in real implementation


class VirtualEmbodiment:
    """
    Main virtual embodiment system integrating all components.
    
    Provides sensorimotor grounding through:
    - Perception-action loops
    - Forward/inverse models
    - Body schema and proprioception
    - Affordance detection
    - Sensorimotor contingencies
    """
    
    def __init__(self):
        # Core components
        self.body_schema = BodySchema()
        self.sensory_buffer = SensoryBuffer()
        self.motor_controller = MotorController(self.body_schema)
        self.affordance_detector = AffordanceDetector(self.body_schema)
        
        # Models
        self.forward_model = ForwardModel()
        self.inverse_model = InverseModel()
        
        # Sensorimotor contingencies
        self.contingencies = []
        
        # Current state
        self.current_state = self._initialize_state()
        
        logger.info("Virtual embodiment system initialized")
    
    def _initialize_state(self) -> Dict[str, Any]:
        """Initialize embodiment state"""
        return {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'energy': 1.0
        }
    
    def perceive_act_cycle(
        self, 
        environment_state: Dict[str, Any],
        goal: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Core embodied cognition cycle: perceive -> decide -> act -> learn.
        
        This is the fundamental loop of embodied intelligence.
        
        Args:
            environment_state: Current state of environment
            goal: Optional goal to pursue
            
        Returns:
            Action taken and its consequences
        """
        # 1. PERCEIVE: Sample environment
        sensory_input = self._generate_sensory_input(environment_state)
        self.sensory_buffer.add_input(sensory_input)
        
        # 2. DETECT AFFORDANCES: What can I do?
        affordances = self.affordance_detector.detect_affordances(environment_state)
        
        # 3. DECIDE: Select action based on affordances and goal
        selected_action = self._select_action(affordances, goal)
        
        if selected_action is None:
            logger.debug("No action selected")
            return {'action': None}
        
        # 4. ANTICIPATE: Use forward model to predict outcome
        predicted_state, confidence = self.forward_model.predict(
            self.current_state, 
            selected_action
        )
        
        # 5. ACT: Execute action
        result = self.motor_controller.execute(selected_action)
        
        # 6. PERCEIVE OUTCOME: Get actual sensory feedback
        post_action_input = self._generate_sensory_input(environment_state)
        self.sensory_buffer.add_input(post_action_input)
        
        # 7. LEARN: Update models based on prediction error
        actual_state = self._update_state(result)
        
        self.forward_model.update(
            self.current_state,
            selected_action,
            actual_state
        )
        
        self.inverse_model.update(
            self.current_state,
            selected_action,
            actual_state
        )
        
        # 8. LEARN CONTINGENCY: action -> perception mapping
        contingency = SensoriMotorContingency(
            action=selected_action,
            expected_perception=predicted_state.get('sensory', sensory_input),
            confidence=confidence
        )
        contingency.update(post_action_input)
        self.contingencies.append(contingency)
        
        # Update current state
        self.current_state = actual_state
        
        return {
            'action': selected_action,
            'result': result,
            'prediction_error': self._compute_prediction_error(
                predicted_state, actual_state
            ),
            'contingency': contingency
        }
    
    def _generate_sensory_input(self, env_state: Dict) -> SensoryInput:
        """Generate sensory input from environment state"""
        # Simplified - would generate multimodal sensory data
        import time
        
        # Visual input (simplified)
        visual_data = np.random.random((64, 64, 3))  # Placeholder
        
        return SensoryInput(
            modality=ModalityType.VISION,
            data=visual_data,
            timestamp=time.time()
        )
    
    def _select_action(
        self, 
        affordances: List[Dict],
        goal: Optional[Dict[str, Any]]
    ) -> Optional[MotorCommand]:
        """Select action from affordances based on goal"""
        if not affordances:
            return None
        
        if goal is None:
            # Explore: random action
            affordance = np.random.choice(affordances)
        else:
            # Exploit: action aligned with goal
            affordance = max(affordances, 
                           key=lambda a: self._goal_alignment(a, goal))
        
        return MotorCommand(
            action_type=affordance['action_type'],
            parameters=affordance['parameters']
        )
    
    def _goal_alignment(self, affordance: Dict, goal: Dict) -> float:
        """Score how well affordance aligns with goal"""
        # Simplified goal alignment
        return affordance.get('utility', 0.5)
    
    def _update_state(self, action_result: Dict) -> Dict[str, Any]:
        """Update internal state based on action result"""
        new_state = self.current_state.copy()
        
        # Update based on action result
        if 'new_position' in action_result:
            new_state['position'] = action_result['new_position']
        
        # Update energy
        new_state['energy'] = self.body_schema.energy_level
        
        return new_state
    
    def _compute_prediction_error(
        self, 
        predicted: Dict, 
        actual: Dict
    ) -> float:
        """Compute error between predicted and actual states"""
        # Simplified error computation
        if 'position' in predicted and 'position' in actual:
            return np.linalg.norm(predicted['position'] - actual['position'])
        return 0.0
    
    def get_proprioception(self) -> Dict[str, Any]:
        """
        Get proprioceptive state (internal body awareness).
        
        Returns current body configuration, energy, damage, etc.
        """
        return {
            'posture': self.body_schema.posture,
            'energy': self.body_schema.energy_level,
            'damage': self.body_schema.damage,
            'position': self.current_state['position']
        }


# Example usage
if __name__ == "__main__":
    # Create virtual embodiment
    embodiment = VirtualEmbodiment()
    
    # Simulate environment
    environment = {
        'objects': [
            {'id': 'obj1', 'position': np.array([0.5, 0.0, 1.5]), 'value': 0.8},
            {'id': 'obj2', 'position': np.array([-0.5, 0.0, 1.5]), 'value': 0.6}
        ],
        'terrain': {
            'walkable': [
                {'center': np.array([1.0, 0.0, 0.0])},
                {'center': np.array([0.0, 1.0, 0.0])}
            ]
        }
    }
    
    # Goal: Explore and interact
    goal = {'type': 'explore', 'value_threshold': 0.7}
    
    # Run perception-action cycles
    print("\n=== Virtual Embodiment Demo ===\n")
    
    for cycle in range(5):
        print(f"\nCycle {cycle + 1}:")
        
        result = embodiment.perceive_act_cycle(environment, goal)
        
        if result['action']:
            print(f"  Action: {result['action']}")
            print(f"  Result: {result['result'].get('success', False)}")
            print(f"  Prediction Error: {result['prediction_error']:.4f}")
            print(f"  Contingency Confidence: {result['contingency'].confidence:.3f}")
        else:
            print("  No action taken")
        
        # Show proprioception
        proprio = embodiment.get_proprioception()
        print(f"  Energy: {proprio['energy']:.3f}")
        print(f"  Position: {proprio['position']}")
    
    print(f"\n=== Summary ===")
    print(f"Total contingencies learned: {len(embodiment.contingencies)}")
    print(f"Forward model entries: {len(embodiment.forward_model.model)}")
    print(f"Inverse model entries: {len(embodiment.inverse_model.model)}")
