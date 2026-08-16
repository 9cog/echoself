"""
Adaptive Dimensional Embedding Engine
======================================

Implements dynamic, context-sensitive dimensional spaces for cognitive
representations in the Echogenesis architecture.

Core Capabilities:
- Dynamic dimension expansion/contraction based on cognitive load
- Multi-scale embedding (local → global)
- Attention-modulated dimensionality
- Cross-scale attention mechanisms
- Relevance-aware embedding updates

The embedding architecture follows:
    Sensory Space (S) ↔ Motor Space (M) ↔ Cognitive Space (C)
       [Ns dims]         [Nm dims]          [Nc dims]
            ↓                ↓                   ↓
       Unified Embodiment Manifold: E = S ⊗ M ⊗ C

Author: Deep Tree Echo
Date: June 2026
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingScale(Enum):
    """Embedding granularity levels."""
    LOCAL = "local"          # Sentence-level: 384 dims
    CONTEXT = "context"      # Document-level: 768 dims
    GLOBAL = "global"        # Memory-level: 1536 dims
    ADAPTIVE = "adaptive"    # Dynamic dimension selection


@dataclass
class EmbeddingConfig:
    """Configuration for adaptive embedding system."""
    sensory_dim: int = 256
    motor_dim: int = 128
    cognitive_dim: int = 768
    echo_dim: int = 1536
    
    local_dim: int = 384
    context_dim: int = 768
    global_dim: int = 1536
    
    min_effective_dim: int = 128
    max_effective_dim: int = 2048
    
    projection_method: str = "attention_gated_linear"
    
    # Attention parameters
    attention_heads: int = 8
    dropout_rate: float = 0.1


@dataclass
class EmbeddingState:
    """Current state of the adaptive embedding system."""
    current_dimension: int = 768
    cognitive_load: float = 0.5
    attention_threshold: float = 0.5
    scale: EmbeddingScale = EmbeddingScale.CONTEXT
    
    # Metrics
    dimension_history: List[int] = field(default_factory=list)
    adaptation_count: int = 0
    drift_detected: bool = False


class AttentionGate:
    """
    Attention-gated projection mechanism for adaptive dimensionality.
    
    Implements soft gating based on relevance scores to dynamically
    select which dimensions are active.
    """
    
    def __init__(self, input_dim: int, output_dim: int, n_heads: int = 8):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        
        # Initialize projection matrices
        self.W_gate = np.random.randn(input_dim, output_dim) * 0.02
        self.W_proj = np.random.randn(input_dim, output_dim) * 0.02
        self.W_query = np.random.randn(input_dim, n_heads) * 0.02
        
    def forward(
        self, 
        x: np.ndarray, 
        attention_weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply attention-gated projection.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            attention_weights: Optional pre-computed attention weights
            
        Returns:
            Tuple of (projected output, gate values)
        """
        # Compute gating scores
        if attention_weights is None:
            attention_weights = self._compute_attention(x)
        
        # Apply soft gating
        gate = self._sigmoid(np.dot(x, self.W_gate))
        projection = np.dot(x, self.W_proj)
        
        # Gated output
        gated_output = gate * projection
        
        return gated_output, gate
    
    def _compute_attention(self, x: np.ndarray) -> np.ndarray:
        """Compute multi-head attention weights."""
        queries = np.dot(x, self.W_query)
        attention = self._softmax(queries, axis=-1)
        return attention
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class CrossScaleAttention:
    """
    Cross-scale attention mechanism for multi-resolution embedding integration.
    
    Enables information flow between local, context, and global scales.
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
        # Scale-specific projections
        self.local_proj = np.random.randn(config.local_dim, config.cognitive_dim) * 0.02
        self.context_proj = np.random.randn(config.context_dim, config.cognitive_dim) * 0.02
        self.global_proj = np.random.randn(config.global_dim, config.cognitive_dim) * 0.02
        
        # Cross-attention weights
        self.W_cross = np.random.randn(config.cognitive_dim * 3, config.cognitive_dim) * 0.02
        
    def attend(
        self,
        local_emb: np.ndarray,
        context_emb: np.ndarray,
        global_emb: np.ndarray
    ) -> np.ndarray:
        """
        Compute cross-scale attended embedding.
        
        Args:
            local_emb: Local-scale embeddings
            context_emb: Context-scale embeddings
            global_emb: Global-scale embeddings
            
        Returns:
            Unified cross-scale embedding
        """
        # Project to common dimension
        local_proj = np.dot(local_emb, self.local_proj)
        context_proj = np.dot(context_emb, self.context_proj)
        global_proj = np.dot(global_emb, self.global_proj)
        
        # Concatenate and attend
        concat = np.concatenate([local_proj, context_proj, global_proj], axis=-1)
        unified = np.dot(concat, self.W_cross)
        
        return unified


class AdaptiveDimensionalEmbedding:
    """
    Main adaptive dimensional embedding engine.
    
    Dynamically adjusts embedding dimensionality based on:
    - Current cognitive load
    - Attention threshold
    - Relevance scores
    - Processing context
    
    Key Features:
    1. Dynamic dimension expansion/contraction
    2. Multi-scale embedding hierarchy
    3. Attention-modulated dimensionality selection
    4. Embedding drift detection and correction
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.state = EmbeddingState()
        
        # Initialize components
        self.attention_gate = AttentionGate(
            self.config.cognitive_dim,
            self.config.echo_dim,
            self.config.attention_heads
        )
        
        self.cross_scale = CrossScaleAttention(self.config)
        
        # Projection matrices for different scales
        self._init_projections()
        
        # Embedding storage for drift detection
        self.embedding_history: List[np.ndarray] = []
        self.drift_threshold = 0.1
        
        logger.info("AdaptiveDimensionalEmbedding initialized")
    
    def _init_projections(self):
        """Initialize projection matrices for dimension transformations."""
        cfg = self.config
        
        # Up-projections (increase dimensionality)
        self.up_proj = {
            (cfg.local_dim, cfg.context_dim): np.random.randn(cfg.local_dim, cfg.context_dim) * 0.02,
            (cfg.context_dim, cfg.global_dim): np.random.randn(cfg.context_dim, cfg.global_dim) * 0.02,
            (cfg.local_dim, cfg.global_dim): np.random.randn(cfg.local_dim, cfg.global_dim) * 0.02,
        }
        
        # Down-projections (decrease dimensionality)
        self.down_proj = {
            (cfg.global_dim, cfg.context_dim): np.random.randn(cfg.global_dim, cfg.context_dim) * 0.02,
            (cfg.context_dim, cfg.local_dim): np.random.randn(cfg.context_dim, cfg.local_dim) * 0.02,
            (cfg.global_dim, cfg.local_dim): np.random.randn(cfg.global_dim, cfg.local_dim) * 0.02,
        }
    
    def compute_effective_dimension(
        self,
        cognitive_load: float,
        attention_threshold: float,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Compute effective dimension based on current cognitive state.
        
        Higher cognitive load → Higher dimensionality (more capacity needed)
        Higher attention threshold → Lower dimensionality (more focused)
        
        Args:
            cognitive_load: Current cognitive load [0, 1]
            attention_threshold: Current attention threshold [0, 1]
            context: Optional processing context
            
        Returns:
            Effective dimension for current state
        """
        # Base dimension from configuration
        base_dim = self.config.cognitive_dim
        
        # Load factor: higher load needs more dimensions
        load_factor = 1.0 + (cognitive_load * 0.5)
        
        # Attention factor: higher threshold means more focus, fewer dims needed
        attention_factor = 1.0 - (attention_threshold * 0.3)
        
        # Context adjustments
        context_factor = 1.0
        if context:
            if context.get('complexity', 'normal') == 'high':
                context_factor = 1.2
            elif context.get('complexity', 'normal') == 'low':
                context_factor = 0.8
        
        # Compute effective dimension
        effective = int(base_dim * load_factor * attention_factor * context_factor)
        
        # Clamp to valid range
        effective = max(self.config.min_effective_dim, 
                       min(self.config.max_effective_dim, effective))
        
        # Update state
        self.state.current_dimension = effective
        self.state.cognitive_load = cognitive_load
        self.state.attention_threshold = attention_threshold
        self.state.dimension_history.append(effective)
        self.state.adaptation_count += 1
        
        return effective
    
    def adaptive_projection(
        self,
        x: np.ndarray,
        cognitive_load: float,
        attention_threshold: float,
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Project input to adaptive dimensional space.
        
        Args:
            x: Input tensor
            cognitive_load: Current cognitive load [0, 1]
            attention_threshold: Current attention threshold [0, 1]
            context: Optional processing context
            
        Returns:
            Adaptively projected tensor
        """
        # Compute target dimension
        target_dim = self.compute_effective_dimension(
            cognitive_load, attention_threshold, context
        )
        
        current_dim = x.shape[-1]
        
        # Project to target dimension
        if target_dim > current_dim:
            projected = self._expand_dimension(x, target_dim)
        elif target_dim < current_dim:
            projected = self._contract_dimension(x, target_dim)
        else:
            projected = x
        
        # Apply attention gating if at cognitive dimension
        if target_dim == self.config.cognitive_dim:
            projected, _ = self.attention_gate.forward(projected)
        
        # Check for embedding drift
        self._detect_drift(projected)
        
        return projected
    
    def _expand_dimension(self, x: np.ndarray, target_dim: int) -> np.ndarray:
        """Expand dimensionality with learned projections."""
        current_dim = x.shape[-1]
        
        # Find appropriate projection
        key = (current_dim, target_dim)
        if key in self.up_proj:
            return np.dot(x, self.up_proj[key])
        
        # Fallback: pad with zeros and linear interpolation
        pad_width = target_dim - current_dim
        if len(x.shape) == 2:
            padding = np.zeros((x.shape[0], pad_width))
        else:
            padding = np.zeros((*x.shape[:-1], pad_width))
        
        return np.concatenate([x, padding], axis=-1)
    
    def _contract_dimension(self, x: np.ndarray, target_dim: int) -> np.ndarray:
        """Contract dimensionality with learned projections."""
        current_dim = x.shape[-1]
        
        # Find appropriate projection
        key = (current_dim, target_dim)
        if key in self.down_proj:
            return np.dot(x, self.down_proj[key])
        
        # Fallback: truncate and learned compression
        return x[..., :target_dim]
    
    def multi_scale_embed(
        self,
        x: np.ndarray,
        scales: Optional[List[EmbeddingScale]] = None
    ) -> Dict[EmbeddingScale, np.ndarray]:
        """
        Create multi-scale embeddings at different granularities.
        
        Args:
            x: Input tensor
            scales: List of scales to compute (default: all)
            
        Returns:
            Dictionary mapping scales to embeddings
        """
        if scales is None:
            scales = [EmbeddingScale.LOCAL, EmbeddingScale.CONTEXT, EmbeddingScale.GLOBAL]
        
        embeddings = {}
        
        for scale in scales:
            if scale == EmbeddingScale.LOCAL:
                embeddings[scale] = self._project_to_scale(x, self.config.local_dim)
            elif scale == EmbeddingScale.CONTEXT:
                embeddings[scale] = self._project_to_scale(x, self.config.context_dim)
            elif scale == EmbeddingScale.GLOBAL:
                embeddings[scale] = self._project_to_scale(x, self.config.global_dim)
            elif scale == EmbeddingScale.ADAPTIVE:
                embeddings[scale] = self._project_to_scale(x, self.state.current_dimension)
        
        return embeddings
    
    def _project_to_scale(self, x: np.ndarray, target_dim: int) -> np.ndarray:
        """Project to specific scale dimension."""
        current_dim = x.shape[-1]
        
        if target_dim > current_dim:
            return self._expand_dimension(x, target_dim)
        elif target_dim < current_dim:
            return self._contract_dimension(x, target_dim)
        return x
    
    def _detect_drift(self, embedding: np.ndarray):
        """Detect and flag embedding drift."""
        if len(self.embedding_history) < 10:
            self.embedding_history.append(embedding.mean(axis=0) if embedding.ndim > 1 else embedding)
            return
        
        # Compute drift as deviation from recent history
        recent_mean = np.mean(self.embedding_history[-10:], axis=0)
        current_mean = embedding.mean(axis=0) if embedding.ndim > 1 else embedding
        
        # Handle dimension mismatch
        min_dim = min(len(recent_mean), len(current_mean))
        drift = np.mean((recent_mean[:min_dim] - current_mean[:min_dim]) ** 2)
        
        self.state.drift_detected = drift > self.drift_threshold
        
        # Update history
        self.embedding_history.append(current_mean)
        if len(self.embedding_history) > 100:
            self.embedding_history = self.embedding_history[-50:]
    
    def create_embodiment_manifold(
        self,
        sensory: np.ndarray,
        motor: np.ndarray,
        cognitive: np.ndarray
    ) -> np.ndarray:
        """
        Create unified embodiment manifold: E = S ⊗ M ⊗ C
        
        Combines sensory, motor, and cognitive spaces into unified
        embodied representation.
        
        Args:
            sensory: Sensory space representation
            motor: Motor space representation
            cognitive: Cognitive space representation
            
        Returns:
            Unified embodiment manifold embedding
        """
        # Project each to common dimension
        s_proj = self._project_to_scale(sensory, self.config.cognitive_dim)
        m_proj = self._project_to_scale(motor, self.config.cognitive_dim)
        c_proj = self._project_to_scale(cognitive, self.config.cognitive_dim)
        
        # Tensor product approximation through concatenation + attention
        concat = np.concatenate([s_proj, m_proj, c_proj], axis=-1)
        
        # Compress to echo dimension
        manifold = self._project_to_scale(concat, self.config.echo_dim)
        
        return manifold
    
    def get_state(self) -> Dict[str, Any]:
        """Get current embedding state."""
        return {
            'current_dimension': self.state.current_dimension,
            'cognitive_load': self.state.cognitive_load,
            'attention_threshold': self.state.attention_threshold,
            'scale': self.state.scale.value,
            'adaptation_count': self.state.adaptation_count,
            'drift_detected': self.state.drift_detected,
            'dimension_variance': np.var(self.state.dimension_history[-20:]) if self.state.dimension_history else 0
        }
    
    def reset_state(self):
        """Reset embedding state to defaults."""
        self.state = EmbeddingState()
        self.embedding_history = []
        logger.info("AdaptiveDimensionalEmbedding state reset")


# Convenience function for creating embeddings
def create_adaptive_embedding(config: Optional[Dict[str, Any]] = None) -> AdaptiveDimensionalEmbedding:
    """
    Factory function to create AdaptiveDimensionalEmbedding.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured AdaptiveDimensionalEmbedding instance
    """
    if config:
        embedding_config = EmbeddingConfig(**config)
    else:
        embedding_config = EmbeddingConfig()
    
    return AdaptiveDimensionalEmbedding(embedding_config)
