"""
AtomSpace Adapter for Echogenesis
==================================

CogPrime-compatible hypergraph interface for echogenesis.
Provides AtomSpace-style node types, truth values, and attention
values for neural-symbolic integration.

This adapter bridges Python echogenesis components with AtomSpace
concepts, enabling PLN-compatible inference and ECAN attention
allocation.

Author: Deep Tree Echo
Date: June 2026
"""

from typing import Dict, Any, Optional, List, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


# ==================== TRUTH VALUES ====================

@dataclass
class TruthValue:
    """
    CogPrime-compatible truth value.
    
    Implements simple probabilistic truth value with:
    - strength: Probability/confidence in truth
    - confidence: Meta-confidence in the strength estimate
    """
    strength: float = 0.5
    confidence: float = 0.5
    
    def __post_init__(self):
        self.strength = np.clip(self.strength, 0.0, 1.0)
        self.confidence = np.clip(self.confidence, 0.0, 1.0)
    
    def to_vector(self) -> np.ndarray:
        """Convert to vector representation."""
        return np.array([self.strength, self.confidence])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'TruthValue':
        """Create from vector."""
        return cls(strength=vec[0], confidence=vec[1])
    
    def merge(self, other: 'TruthValue') -> 'TruthValue':
        """Merge two truth values (PLN revision rule)."""
        # Weighted average based on confidence
        total_conf = self.confidence + other.confidence
        if total_conf == 0:
            return TruthValue()
        
        new_strength = (
            self.strength * self.confidence + 
            other.strength * other.confidence
        ) / total_conf
        
        # Confidence increases with more evidence
        new_confidence = min(1.0, total_conf / 2)
        
        return TruthValue(new_strength, new_confidence)
    
    def negate(self) -> 'TruthValue':
        """Logical negation."""
        return TruthValue(1.0 - self.strength, self.confidence)
    
    def __repr__(self) -> str:
        return f"TV({self.strength:.3f}, {self.confidence:.3f})"


@dataclass
class IndefiniteTruthValue(TruthValue):
    """
    Indefinite truth value with probability distribution.
    
    Extends simple truth value with:
    - lower: Lower bound of credible interval
    - upper: Upper bound of credible interval
    """
    lower: float = 0.0
    upper: float = 1.0
    
    def width(self) -> float:
        """Width of credible interval."""
        return self.upper - self.lower
    
    def sample(self) -> float:
        """Sample from distribution."""
        return np.random.beta(
            self.strength * 10 + 1,
            (1 - self.strength) * 10 + 1
        )


# ==================== ATTENTION VALUES ====================

@dataclass
class AttentionValue:
    """
    ECAN-compatible attention value.
    
    Implements:
    - STI: Short-Term Importance (for working memory)
    - LTI: Long-Term Importance (for persistent memory)
    - VLTI: Very Long-Term Importance flag
    """
    sti: float = 0.0  # Short-Term Importance
    lti: float = 0.0  # Long-Term Importance
    vlti: bool = False  # Very Long-Term Importance flag
    
    # Threshold constants
    ATTENTION_FOCUS_BOUNDARY: float = 100.0
    AF_SIZE: int = 50  # Attentional focus size
    
    def in_attention_focus(self) -> bool:
        """Check if in attentional focus."""
        return self.sti >= self.ATTENTION_FOCUS_BOUNDARY
    
    def spread_attention(
        self,
        target: 'AttentionValue',
        weight: float = 0.1
    ) -> 'AttentionValue':
        """Spread attention to target."""
        transfer = self.sti * weight
        self.sti -= transfer
        target.sti += transfer
        return target
    
    def decay(self, rate: float = 0.95):
        """Apply attention decay."""
        self.sti *= rate
        if not self.vlti:
            self.lti *= rate
    
    def stimulate(self, amount: float):
        """Increase STI (excitation)."""
        self.sti += amount
    
    def inhibit(self, amount: float):
        """Decrease STI (inhibition)."""
        self.sti = max(0, self.sti - amount)
    
    def to_vector(self) -> np.ndarray:
        """Convert to vector."""
        return np.array([self.sti, self.lti, float(self.vlti)])
    
    def __repr__(self) -> str:
        vlti_str = " VLTI" if self.vlti else ""
        return f"AV(STI:{self.sti:.1f}, LTI:{self.lti:.1f}{vlti_str})"


# ==================== NODE TYPES ====================

class NodeType(Enum):
    """CogPrime-compatible node types."""
    CONCEPT = auto()
    PREDICATE = auto()
    SCHEMA = auto()
    GROUNDED_SCHEMA = auto()
    PROCEDURE = auto()
    GOAL = auto()
    NUMBER = auto()
    WORD = auto()
    SENTENCE = auto()
    CONTEXT = auto()
    ANCHOR = auto()
    VARIABLE = auto()
    PATTERN = auto()
    MODEL = auto()


class LinkType(Enum):
    """CogPrime-compatible link types."""
    # Logical links
    INHERITANCE = auto()
    SIMILARITY = auto()
    IMPLICATION = auto()
    EQUIVALENCE = auto()
    
    # List links
    LIST = auto()
    SET = auto()
    
    # Evaluation links
    EVALUATION = auto()
    EXECUTION = auto()
    
    # Context links
    CONTEXT = auto()
    MEMBER = auto()
    
    # Pattern links
    BIND = auto()
    QUOTE = auto()
    
    # Cognitive schema links
    PROCEDURE_GOAL = auto()
    CONTEXT_PROCEDURE = auto()
    SCHEMA_GOAL = auto()
    
    # Attention links
    HEBBIAN = auto()
    ASYMMETRIC_HEBBIAN = auto()


# ==================== ATOMS ====================

@dataclass
class Atom(ABC):
    """
    Base atom class for AtomSpace.
    
    All atoms have:
    - Unique identifier
    - Truth value
    - Attention value
    - Optional embedding vector
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    truth_value: TruthValue = field(default_factory=TruthValue)
    attention_value: AttentionValue = field(default_factory=AttentionValue)
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    @abstractmethod
    def get_type(self) -> Union[NodeType, LinkType]:
        """Get atom type."""
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """String representation for display."""
        pass
    
    def get_embedding(self, dimension: int = 768) -> np.ndarray:
        """Get or compute embedding."""
        if self.embedding is None:
            # Generate hash-based embedding as fallback
            hash_val = hash(self.id)
            np.random.seed(hash_val % (2**32))
            self.embedding = np.random.randn(dimension)
            self.embedding /= np.linalg.norm(self.embedding)
        return self.embedding
    
    def set_embedding(self, embedding: np.ndarray):
        """Set embedding vector."""
        self.embedding = embedding


@dataclass
class Node(Atom):
    """
    AtomSpace node (leaf atom).
    """
    name: str = ""
    node_type: NodeType = NodeType.CONCEPT
    
    def get_type(self) -> NodeType:
        return self.node_type
    
    def to_string(self) -> str:
        return f"({self.node_type.name} \"{self.name}\")"
    
    def __hash__(self):
        return hash((self.node_type, self.name))
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.node_type == other.node_type and self.name == other.name


@dataclass
class Link(Atom):
    """
    AtomSpace link (connecting atoms).
    """
    link_type: LinkType = LinkType.LIST
    outgoing: List[Atom] = field(default_factory=list)
    
    def get_type(self) -> LinkType:
        return self.link_type
    
    def to_string(self) -> str:
        outgoing_strs = " ".join(a.to_string() for a in self.outgoing)
        return f"({self.link_type.name} {outgoing_strs})"
    
    def arity(self) -> int:
        """Number of outgoing atoms."""
        return len(self.outgoing)
    
    def get_outgoing(self, index: int) -> Optional[Atom]:
        """Get outgoing atom by index."""
        if 0 <= index < len(self.outgoing):
            return self.outgoing[index]
        return None
    
    def __hash__(self):
        return hash((self.link_type, tuple(a.id for a in self.outgoing)))


# ==================== COGNITIVE SCHEMATICS ====================

@dataclass
class CognitiveSchematic:
    """
    CogPrime cognitive schematic: Context → Procedure → Goal
    
    Represents procedural knowledge linking contexts to
    goal-achieving procedures.
    """
    context: Node
    procedure: Node
    goal: Node
    truth_value: TruthValue = field(default_factory=TruthValue)
    attention_value: AttentionValue = field(default_factory=AttentionValue)
    activation_count: int = 0
    last_activation: Optional[datetime] = None
    
    def to_implication_link(self) -> Link:
        """Convert to PLN-compatible implication link."""
        # Context ∧ Procedure → Goal
        context_proc = Link(
            link_type=LinkType.LIST,
            outgoing=[self.context, self.procedure],
            truth_value=TruthValue(1.0, 1.0)
        )
        
        return Link(
            link_type=LinkType.IMPLICATION,
            outgoing=[context_proc, self.goal],
            truth_value=self.truth_value
        )
    
    def activate(self):
        """Record schematic activation."""
        self.activation_count += 1
        self.last_activation = datetime.now()
        self.attention_value.stimulate(10.0)
    
    def matches_context(self, context: Dict[str, Any]) -> float:
        """Check if context matches schematic context."""
        # Embedding-based matching
        if self.context.embedding is not None:
            context_vec = context.get('embedding')
            if context_vec is not None:
                similarity = np.dot(
                    self.context.embedding, 
                    context_vec
                ) / (
                    np.linalg.norm(self.context.embedding) * 
                    np.linalg.norm(context_vec)
                )
                return max(0, similarity)
        return 0.5  # Default


# ==================== ATOMSPACE ====================

class AtomSpace:
    """
    AtomSpace implementation for echogenesis.
    
    Provides:
    - Atom storage and retrieval
    - Type-based queries
    - Pattern matching
    - Attention allocation
    - Embedding management
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.atoms: Dict[str, Atom] = {}
        self.by_type: Dict[Union[NodeType, LinkType], Set[str]] = {}
        self.by_name: Dict[str, Set[str]] = {}  # For nodes
        self.embedding_dim = embedding_dim
        
        # Attention economy
        self.attention_bank: float = 1000.0
        self.stimulation_rate: float = 0.1
        self.decay_rate: float = 0.95
        
        # Cognitive schematics
        self.schematics: List[CognitiveSchematic] = []
        
        logger.info(f"AtomSpace initialized with embedding_dim={embedding_dim}")
    
    def add_atom(self, atom: Atom) -> Atom:
        """Add atom to space."""
        self.atoms[atom.id] = atom
        
        # Index by type
        atom_type = atom.get_type()
        if atom_type not in self.by_type:
            self.by_type[atom_type] = set()
        self.by_type[atom_type].add(atom.id)
        
        # Index by name (for nodes)
        if isinstance(atom, Node):
            if atom.name not in self.by_name:
                self.by_name[atom.name] = set()
            self.by_name[atom.name].add(atom.id)
        
        return atom
    
    def add_node(
        self,
        node_type: NodeType,
        name: str,
        truth_value: Optional[TruthValue] = None,
        embedding: Optional[np.ndarray] = None
    ) -> Node:
        """Create and add a node."""
        node = Node(
            name=name,
            node_type=node_type,
            truth_value=truth_value or TruthValue(),
            embedding=embedding
        )
        return self.add_atom(node)
    
    def add_link(
        self,
        link_type: LinkType,
        outgoing: List[Atom],
        truth_value: Optional[TruthValue] = None
    ) -> Link:
        """Create and add a link."""
        link = Link(
            link_type=link_type,
            outgoing=outgoing,
            truth_value=truth_value or TruthValue()
        )
        return self.add_atom(link)
    
    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Get atom by ID."""
        return self.atoms.get(atom_id)
    
    def get_by_type(
        self,
        atom_type: Union[NodeType, LinkType]
    ) -> List[Atom]:
        """Get all atoms of a type."""
        ids = self.by_type.get(atom_type, set())
        return [self.atoms[id] for id in ids if id in self.atoms]
    
    def get_by_name(self, name: str) -> List[Node]:
        """Get nodes by name."""
        ids = self.by_name.get(name, set())
        return [
            self.atoms[id] for id in ids 
            if id in self.atoms and isinstance(self.atoms[id], Node)
        ]
    
    def find_incoming(self, atom: Atom) -> List[Link]:
        """Find all links pointing to atom."""
        incoming = []
        for a in self.atoms.values():
            if isinstance(a, Link):
                if any(o.id == atom.id for o in a.outgoing):
                    incoming.append(a)
        return incoming
    
    def add_schematic(
        self,
        context_name: str,
        procedure_name: str,
        goal_name: str,
        strength: float = 0.5,
        confidence: float = 0.5
    ) -> CognitiveSchematic:
        """Add a cognitive schematic."""
        context = self.add_node(NodeType.CONTEXT, context_name)
        procedure = self.add_node(NodeType.PROCEDURE, procedure_name)
        goal = self.add_node(NodeType.GOAL, goal_name)
        
        schematic = CognitiveSchematic(
            context=context,
            procedure=procedure,
            goal=goal,
            truth_value=TruthValue(strength, confidence)
        )
        
        self.schematics.append(schematic)
        
        # Also add as link
        self.add_atom(schematic.to_implication_link())
        
        return schematic
    
    def find_applicable_schematics(
        self,
        context: Dict[str, Any],
        goal: Optional[str] = None,
        threshold: float = 0.5
    ) -> List[Tuple[CognitiveSchematic, float]]:
        """Find schematics applicable in context."""
        results = []
        
        for schematic in self.schematics:
            match_score = schematic.matches_context(context)
            
            if goal and schematic.goal.name != goal:
                continue
            
            if match_score >= threshold:
                results.append((schematic, match_score))
        
        return sorted(results, key=lambda x: -x[1])
    
    # Attention allocation
    def spread_attention(self):
        """Spread attention through hebbian links."""
        for atom in self.atoms.values():
            if isinstance(atom, Link) and atom.link_type in [
                LinkType.HEBBIAN, LinkType.ASYMMETRIC_HEBBIAN
            ]:
                source = atom.outgoing[0]
                target = atom.outgoing[1]
                
                # Spread proportional to link strength
                weight = atom.truth_value.strength * self.stimulation_rate
                source.attention_value.spread_attention(
                    target.attention_value, weight
                )
    
    def decay_attention(self):
        """Apply attention decay to all atoms."""
        for atom in self.atoms.values():
            atom.attention_value.decay(self.decay_rate)
    
    def get_attention_focus(self, size: Optional[int] = None) -> List[Atom]:
        """Get atoms in attentional focus."""
        if size is None:
            size = AttentionValue.AF_SIZE
        
        atoms_by_sti = sorted(
            self.atoms.values(),
            key=lambda a: -a.attention_value.sti
        )
        
        return atoms_by_sti[:size]
    
    def stimulate(self, atom_id: str, amount: float):
        """Stimulate atom attention."""
        if atom_id in self.atoms:
            self.atoms[atom_id].attention_value.stimulate(amount)
    
    # Embedding operations
    def compute_similarity(self, atom1: Atom, atom2: Atom) -> float:
        """Compute embedding similarity."""
        emb1 = atom1.get_embedding(self.embedding_dim)
        emb2 = atom2.get_embedding(self.embedding_dim)
        
        return float(np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        ))
    
    def find_similar(
        self,
        atom: Atom,
        k: int = 10,
        atom_type: Optional[Union[NodeType, LinkType]] = None
    ) -> List[Tuple[Atom, float]]:
        """Find k most similar atoms."""
        candidates = (
            self.get_by_type(atom_type) if atom_type else list(self.atoms.values())
        )
        
        similarities = []
        for candidate in candidates:
            if candidate.id == atom.id:
                continue
            sim = self.compute_similarity(atom, candidate)
            similarities.append((candidate, sim))
        
        return sorted(similarities, key=lambda x: -x[1])[:k]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            'atoms': {
                id: {
                    'type': str(atom.get_type()),
                    'string': atom.to_string(),
                    'truth_value': {
                        's': atom.truth_value.strength,
                        'c': atom.truth_value.confidence
                    },
                    'attention_value': {
                        'sti': atom.attention_value.sti,
                        'lti': atom.attention_value.lti
                    }
                }
                for id, atom in self.atoms.items()
            },
            'schematics': len(self.schematics),
            'embedding_dim': self.embedding_dim
        }
    
    def __len__(self) -> int:
        return len(self.atoms)
    
    def __repr__(self) -> str:
        return f"AtomSpace({len(self)} atoms, {len(self.schematics)} schematics)"


# ==================== ECHOGENESIS INTEGRATION ====================

class EchogenesisAtomSpaceAdapter:
    """
    Adapter connecting echogenesis to AtomSpace.
    
    Maps echogenesis cognitive states to hypergraph patterns
    and retrieves relevant patterns for cognitive processing.
    """
    
    def __init__(self, atomspace: Optional[AtomSpace] = None):
        self.atomspace = atomspace or AtomSpace()
        
        # Concept embedding cache
        self.concept_embeddings: Dict[str, np.ndarray] = {}
    
    def encode_cognitive_state(
        self,
        state: Dict[str, Any],
        truth_value: Optional[TruthValue] = None
    ) -> Node:
        """Encode cognitive state as concept node."""
        state_name = state.get('name', f"state_{uuid.uuid4().hex[:8]}")
        
        # Create embedding from state
        if 'embedding' in state:
            embedding = np.array(state['embedding'])
        else:
            # Generate from state content
            embedding = self._state_to_embedding(state)
        
        node = self.atomspace.add_node(
            NodeType.CONCEPT,
            state_name,
            truth_value or TruthValue(0.8, 0.7),
            embedding
        )
        
        return node
    
    def _state_to_embedding(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert state to embedding vector."""
        # Simple hash-based embedding
        state_str = str(sorted(state.items()))
        np.random.seed(hash(state_str) % (2**32))
        embedding = np.random.randn(self.atomspace.embedding_dim)
        embedding /= np.linalg.norm(embedding)
        return embedding
    
    def encode_procedure(
        self,
        name: str,
        steps: List[str],
        truth_value: Optional[TruthValue] = None
    ) -> Node:
        """Encode procedure as schema node."""
        node = self.atomspace.add_node(
            NodeType.PROCEDURE,
            name,
            truth_value or TruthValue(0.9, 0.8)
        )
        
        # Add steps as linked concepts
        for i, step in enumerate(steps):
            step_node = self.atomspace.add_node(
                NodeType.SCHEMA,
                f"{name}_step_{i}"
            )
            step_node.metadata['step_text'] = step
            
            self.atomspace.add_link(
                LinkType.MEMBER,
                [step_node, node]
            )
        
        return node
    
    def encode_goal(
        self,
        name: str,
        target_state: Optional[Dict] = None,
        importance: float = 0.5
    ) -> Node:
        """Encode goal node."""
        node = self.atomspace.add_node(
            NodeType.GOAL,
            name,
            TruthValue(importance, 0.9)
        )
        
        if target_state:
            target_node = self.encode_cognitive_state(target_state)
            self.atomspace.add_link(
                LinkType.EVALUATION,
                [node, target_node]
            )
        
        return node
    
    def create_cognitive_schematic(
        self,
        context: Dict[str, Any],
        procedure: str,
        goal: str,
        truth_value: Optional[TruthValue] = None
    ) -> CognitiveSchematic:
        """Create full cognitive schematic."""
        context_name = context.get('name', f"context_{uuid.uuid4().hex[:8]}")
        
        schematic = self.atomspace.add_schematic(
            context_name,
            procedure,
            goal,
            truth_value.strength if truth_value else 0.7,
            truth_value.confidence if truth_value else 0.8
        )
        
        # Add context embedding
        if 'embedding' in context:
            schematic.context.embedding = np.array(context['embedding'])
        
        return schematic
    
    def find_relevant_patterns(
        self,
        state: Dict[str, Any],
        k: int = 10
    ) -> List[Tuple[Atom, float]]:
        """Find patterns relevant to current state."""
        state_embedding = self._state_to_embedding(state)
        
        # Create temporary atom for comparison
        query_atom = Node(
            name="query",
            node_type=NodeType.CONCEPT,
            embedding=state_embedding
        )
        
        return self.atomspace.find_similar(query_atom, k)
    
    def select_procedure(
        self,
        context: Dict[str, Any],
        goal: str,
        threshold: float = 0.5
    ) -> Optional[CognitiveSchematic]:
        """Select best procedure for context and goal."""
        applicable = self.atomspace.find_applicable_schematics(
            context, goal, threshold
        )
        
        if applicable:
            best, score = applicable[0]
            best.activate()
            return best
        
        return None
    
    def update_schematic_truth(
        self,
        schematic: CognitiveSchematic,
        success: bool,
        weight: float = 0.1
    ):
        """Update schematic truth value based on outcome."""
        current = schematic.truth_value
        
        if success:
            new_strength = current.strength + weight * (1 - current.strength)
        else:
            new_strength = current.strength - weight * current.strength
        
        # Confidence increases with experience
        new_confidence = current.confidence + weight * (1 - current.confidence)
        
        schematic.truth_value = TruthValue(new_strength, new_confidence)


# ==================== INITIALIZATION ====================

def create_atomspace_adapter(
    embedding_dim: int = 768
) -> EchogenesisAtomSpaceAdapter:
    """Create initialized AtomSpace adapter."""
    atomspace = AtomSpace(embedding_dim)
    adapter = EchogenesisAtomSpaceAdapter(atomspace)
    
    logger.info("AtomSpace adapter created")
    
    return adapter


# Module exports
__all__ = [
    'TruthValue',
    'IndefiniteTruthValue',
    'AttentionValue',
    'NodeType',
    'LinkType',
    'Atom',
    'Node',
    'Link',
    'CognitiveSchematic',
    'AtomSpace',
    'EchogenesisAtomSpaceAdapter',
    'create_atomspace_adapter'
]
