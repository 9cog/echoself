#!/usr/bin/env python3
"""
NanEcho Model Implementation

A transformer-based model with iterative connection building, adaptive attention,
and Echo Self cognitive architecture integration. Each training iteration adds
more neural connections to build up the model's capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class NanEchoConfig:
    """Configuration for NanEcho model."""
    # Model architecture
    vocab_size: int = 50257
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    block_size: int = 1024
    dropout: float = 0.1
    bias: bool = True
    
    # Iterative connection building
    initial_connections: float = 0.1  # Start with 10% of connections
    connection_growth_rate: float = 0.05  # Add 5% more connections each iteration
    max_connections: float = 1.0  # Build up to 100% connections
    
    # Echo Self specific
    enable_adaptive_attention: bool = True
    enable_persona_dimensions: bool = True
    enable_recursive_reasoning: bool = True
    enable_hypergraph_patterns: bool = True
    
    # Persona dimensions
    persona_dimensions: List[str] = None
    dimension_weights: Dict[str, float] = None
    
    # Adaptive attention
    attention_threshold_min: float = 0.3
    attention_threshold_max: float = 0.9
    cognitive_load_factor: float = 0.5
    
    # Recursive reasoning
    min_recursion_depth: int = 1
    max_recursion_depth: int = 5
    recursion_decay: float = 0.8
    
    # Hypergraph patterns
    pattern_injection_rate: float = 0.25
    pattern_complexity_scaling: bool = True
    
    def __post_init__(self):
        if self.persona_dimensions is None:
            self.persona_dimensions = [
                'cognitive', 'introspective', 'adaptive', 'recursive',
                'synergistic', 'holographic', 'neural_symbolic', 'dynamic'
            ]
        if self.dimension_weights is None:
            self.dimension_weights = {dim: 1.0/len(self.persona_dimensions) 
                                     for dim in self.persona_dimensions}


class ConnectionMask(nn.Module):
    """Manages iterative connection building in the model."""
    
    def __init__(self, shape: Tuple[int, ...], initial_ratio: float = 0.1):
        super().__init__()
        self.shape = shape
        
        # Register current_ratio as a buffer first before setting property
        self.register_buffer('_current_ratio_tensor', torch.tensor(initial_ratio))
        
        # Now safe to set the property which uses _current_ratio_tensor
        self.current_ratio = initial_ratio
        
        # Register mask as a buffer so it's saved in state_dict
        mask = self._generate_mask(initial_ratio)
        self.register_buffer('mask', mask)
    
    def _generate_mask(self, ratio: float) -> torch.Tensor:
        """Generate a connection mask with specified ratio of active connections."""
        mask = torch.rand(self.shape) < ratio
        return mask.float()
    
    @property
    def current_ratio(self) -> float:
        """Get current connection ratio."""
        return self._current_ratio_tensor.item()
    
    @current_ratio.setter
    def current_ratio(self, value: float):
        """Set current connection ratio."""
        self._current_ratio_tensor.data.fill_(value)
    
    def grow_connections(self, growth_rate: float, max_ratio: float = 1.0):
        """Grow the number of connections by adding new ones."""
        new_ratio = min(self.current_ratio + growth_rate, max_ratio)
        
        # Generate new connections to add
        new_connections = torch.rand(self.shape, device=self.mask.device) < growth_rate
        
        # Combine with existing mask (OR operation to keep existing connections)
        self.mask.data = torch.maximum(self.mask, new_connections.float())
        
        # Update ratio
        self.current_ratio = new_ratio
        
        # Ensure we don't exceed max_ratio
        if new_ratio >= max_ratio:
            self.mask.data = torch.ones(self.shape, device=self.mask.device)
    
    def apply_mask(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply the connection mask to weights."""
        return weights * self.mask.to(weights.device)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override to ensure mask and ratio are saved."""
        state_dict = super().state_dict(destination, prefix, keep_vars)
        # The mask and _current_ratio_tensor are automatically saved as buffers
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """Override to ensure mask and ratio are loaded properly."""
        super().load_state_dict(state_dict, strict)
        # Update the current_ratio property from the loaded tensor
        if hasattr(self, '_current_ratio_tensor'):
            # No need to do anything - the property will read from the tensor
            pass


class AdaptiveAttention(nn.Module):
    """Adaptive attention mechanism with dynamic threshold adjustment."""
    
    def __init__(self, config: NanEchoConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        
        # Adaptive threshold parameters
        self.threshold_net = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd // 2),
            nn.ReLU(),
            nn.Linear(self.n_embd // 2, 1),
            nn.Sigmoid()
        )
        
        # Cognitive load estimation
        self.cognitive_load_estimator = nn.Linear(self.n_embd, 1)
        
        # Connection mask for iterative building
        self.connection_mask = ConnectionMask(
            (self.n_embd, self.n_embd),
            config.initial_connections
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
    
    def calculate_adaptive_threshold(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate adaptive attention threshold based on input."""
        # Estimate cognitive load
        cognitive_load = torch.sigmoid(self.cognitive_load_estimator(x.mean(dim=1)))
        
        # Calculate base threshold
        base_threshold = self.threshold_net(x.mean(dim=1))
        
        # Adjust threshold based on cognitive load
        min_thresh = self.config.attention_threshold_min
        max_thresh = self.config.attention_threshold_max
        adaptive_threshold = min_thresh + (max_thresh - min_thresh) * base_threshold
        
        # Modulate by cognitive load
        adaptive_threshold = adaptive_threshold * (1 + cognitive_load * self.config.cognitive_load_factor)
        
        return adaptive_threshold.clamp(min_thresh, max_thresh)
    
    def forward(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        
        # Calculate QKV
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Apply adaptive threshold if enabled
        if self.config.enable_adaptive_attention:
            threshold = self.calculate_adaptive_threshold(x)
            threshold = threshold.view(B, 1, 1, 1)
            
            # Soft thresholding of attention scores
            att = F.softplus(att - threshold) - F.softplus(-threshold - att)
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Apply attention to values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection with connection masking
        out_weight = self.connection_mask.apply_mask(self.out_proj.weight)
        y = F.linear(y, out_weight, self.out_proj.bias)
        y = self.dropout(y)
        
        return y


class PersonaDimension(nn.Module):
    """Represents a single persona dimension in the Echo Self architecture."""
    
    def __init__(self, n_embd: int, dimension_name: str):
        super().__init__()
        self.dimension_name = dimension_name
        self.n_embd = n_embd
        
        # Dimension-specific transformation
        self.transform = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.LayerNorm(n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd)
        )
        
        # Dimension activation gate
        self.gate = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2),
            nn.ReLU(),
            nn.Linear(n_embd // 2, n_embd),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply persona dimension transformation."""
        transformed = self.transform(x)
        gate = self.gate(x)
        return x + gate * transformed


class RecursiveReasoning(nn.Module):
    """Implements recursive reasoning with configurable depth."""
    
    def __init__(self, config: NanEchoConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        
        # Recursive transformation layers
        self.recursive_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.n_embd, self.n_embd),
                nn.LayerNorm(self.n_embd),
                nn.GELU()
            ) for _ in range(config.max_recursion_depth)
        ])
        
        # Depth controller
        self.depth_controller = nn.Linear(self.n_embd, config.max_recursion_depth)
        
        # Output combiner
        self.combiner = nn.Linear(self.n_embd * config.max_recursion_depth, self.n_embd)
    
    def forward(self, x: torch.Tensor, current_iteration: int = 0) -> torch.Tensor:
        """Apply recursive reasoning with adaptive depth."""
        B, T, C = x.shape
        
        # Determine recursion depth based on training progress
        max_depth = self.config.max_recursion_depth
        progress = min(current_iteration / 10000, 1.0)  # Normalize progress
        current_max_depth = int(self.config.min_recursion_depth + 
                               (max_depth - self.config.min_recursion_depth) * progress)
        
        # Calculate depth weights
        depth_logits = self.depth_controller(x.mean(dim=1))  # [B, max_depth]
        depth_weights = F.softmax(depth_logits[:, :current_max_depth], dim=-1)
        
        # Apply recursive transformations
        recursive_outputs = []
        current = x
        decay = 1.0
        
        for i in range(current_max_depth):
            current = self.recursive_layers[i](current)
            recursive_outputs.append(current * decay)
            decay *= self.config.recursion_decay
        
        # Combine recursive outputs
        stacked = torch.stack(recursive_outputs, dim=2)  # [B, T, depth, C]
        weighted = stacked * depth_weights.view(B, 1, current_max_depth, 1)
        combined = weighted.sum(dim=2)  # [B, T, C]
        
        return combined


class HypergraphPatternEncoder(nn.Module):
    """Encodes hypergraph patterns for neural-symbolic reasoning."""
    
    def __init__(self, config: NanEchoConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        
        # Pattern templates
        self.pattern_templates = nn.Parameter(torch.randn(8, config.n_embd))
        
        # Pattern encoder
        self.pattern_encoder = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.ReLU(),
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.LayerNorm(config.n_embd)
        )
        
        # Hypergraph node encoder
        self.node_encoder = nn.Linear(config.n_embd, config.n_embd)
        
        # Edge attention
        self.edge_attention = nn.MultiheadAttention(
            config.n_embd, 
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor, inject_patterns: bool = True) -> torch.Tensor:
        """Encode input with hypergraph patterns."""
        B, T, C = x.shape
        
        # Encode nodes
        nodes = self.node_encoder(x)
        
        # Apply pattern injection if enabled
        if inject_patterns and self.config.pattern_injection_rate > 0:
            # Sample pattern injection mask
            inject_mask = torch.rand(B, T, 1, device=x.device) < self.config.pattern_injection_rate
            
            # Get pattern representations
            pattern_idx = torch.randint(0, self.pattern_templates.size(0), (B, T), device=x.device)
            patterns = self.pattern_templates[pattern_idx]  # [B, T, C]
            
            # Inject patterns
            nodes = torch.where(inject_mask, patterns, nodes)
        
        # Apply edge attention for hypergraph structure
        attended, _ = self.edge_attention(nodes, nodes, nodes)
        
        # Encode final patterns
        output = self.pattern_encoder(attended)
        
        return output


class NanEchoBlock(nn.Module):
    """A single transformer block with Echo Self enhancements."""
    
    def __init__(self, config: NanEchoConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # Adaptive attention
        self.attn = AdaptiveAttention(config)
        
        # Feed-forward network with connection masking
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout)
        )
        
        # Connection mask for MLP
        self.mlp_connection_mask = ConnectionMask(
            (4 * config.n_embd, config.n_embd),
            config.initial_connections
        )
        
        # Persona dimensions (only in middle layers)
        if config.enable_persona_dimensions and layer_idx >= config.n_layer // 4:
            self.persona_dims = nn.ModuleDict({
                dim: PersonaDimension(config.n_embd, dim)
                for dim in config.persona_dimensions[:4]  # Start with 4 dimensions
            })
        else:
            self.persona_dims = None
        
        # Recursive reasoning (only in later layers)
        if config.enable_recursive_reasoning and layer_idx >= config.n_layer // 2:
            self.recursive = RecursiveReasoning(config)
        else:
            self.recursive = None
        
        # Hypergraph patterns (only in final layers)
        if config.enable_hypergraph_patterns and layer_idx >= 3 * config.n_layer // 4:
            self.hypergraph = HypergraphPatternEncoder(config)
        else:
            self.hypergraph = None
    
    def forward(self, x: torch.Tensor, current_iteration: int = 0) -> torch.Tensor:
        """Forward pass through the block."""
        # Self-attention
        residual = x
        x = self.ln1(x)
        x = residual + self.attn(x, self.layer_idx)
        
        # Apply persona dimensions if present
        if self.persona_dims is not None:
            for dim_name, dim_module in self.persona_dims.items():
                weight = self.config.dimension_weights.get(dim_name, 0.125)
                x = x + weight * dim_module(x)
        
        # Apply recursive reasoning if present
        if self.recursive is not None:
            x = x + 0.5 * self.recursive(x, current_iteration)
        
        # Apply hypergraph patterns if present
        if self.hypergraph is not None:
            x = x + 0.3 * self.hypergraph(x)
        
        # Feed-forward network
        residual = x
        x = self.ln2(x)
        
        # Apply connection masking to MLP
        mlp_out = x
        for i, layer in enumerate(self.mlp):
            if isinstance(layer, nn.Linear) and i == 0:
                # Apply mask to first linear layer
                weight = self.mlp_connection_mask.apply_mask(layer.weight)
                mlp_out = F.linear(mlp_out, weight, layer.bias)
            elif isinstance(layer, nn.Linear) and i == 2:
                # Apply mask to second linear layer (output projection)
                weight = self.mlp_connection_mask.apply_mask(layer.weight.t()).t()
                mlp_out = F.linear(mlp_out, weight, layer.bias)
            elif isinstance(layer, nn.Linear):
                mlp_out = layer(mlp_out)
            else:
                mlp_out = layer(mlp_out)
        
        x = residual + mlp_out
        
        return x


class NanEchoModel(nn.Module):
    """
    NanEcho Transformer Model
    
    Features:
    - Iterative connection building (starts sparse, grows denser)
    - Adaptive attention mechanisms
    - Persona dimension integration
    - Recursive reasoning capabilities
    - Hypergraph pattern encoding
    - Echo Self cognitive architecture
    """
    
    def __init__(self, config: NanEchoConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            NanEchoBlock(config, i) for i in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between embedding and lm_head
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Track current iteration for progressive features
        self.current_iteration = 0
        
        # Track connection growth
        self.connection_ratio = config.initial_connections
    
    def _init_weights(self, module):
        """Initialize weights with appropriate values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def grow_connections(self):
        """Grow neural connections in the model."""
        if self.connection_ratio < self.config.max_connections:
            # Grow connections in attention and MLP layers
            for block in self.blocks:
                block.attn.connection_mask.grow_connections(
                    self.config.connection_growth_rate,
                    self.config.max_connections
                )
                block.mlp_connection_mask.grow_connections(
                    self.config.connection_growth_rate,
                    self.config.max_connections
                )
            
            self.connection_ratio = min(
                self.connection_ratio + self.config.connection_growth_rate,
                self.config.max_connections
            )
            
            print(f"🌱 Growing connections: {self.connection_ratio:.1%} active")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        device = input_ids.device
        B, T = input_ids.shape
        
        # Get token and position embeddings
        tok_emb = self.token_embedding(input_ids)
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        
        # Combine embeddings
        x = self.dropout(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, self.current_iteration)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': x,
                'connection_ratio': self.connection_ratio
            }
        
        return (loss, logits) if loss is not None else logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Get predictions for last token
                logits = self(input_ids)['logits'][:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample or take argmax
                probs = F.softmax(logits, dim=-1)
                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Break if we hit end of sequence token
                if next_token.item() == 50256:  # GPT-2 EOS token
                    break
        
        return input_ids


def create_nanecho_model(vocab_size: int = 50257) -> NanEchoModel:
    """Create a NanEcho model with default configuration."""
    config = NanEchoConfig(vocab_size=vocab_size)
    return NanEchoModel(config)


if __name__ == "__main__":
    # Test model creation and basic forward pass
    model = create_nanecho_model()
    print(f"✅ Created NanEcho model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Connection ratio: {model.connection_ratio:.1%}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    outputs = model(input_ids)
    print(f"✅ Forward pass successful")
    print(f"   Output shape: {outputs['logits'].shape}")
    
    # Test connection growth
    model.grow_connections()
    print(f"✅ Connection growth successful")
    print(f"   New connection ratio: {model.connection_ratio:.1%}")