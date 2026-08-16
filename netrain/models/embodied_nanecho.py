"""
Embodied NanEcho Model
=======================

NanEcho model with virtual embodiment grounding, integrating
transformer-based language modeling with sensorimotor processing
for 4E cognition implementation.

Key Features:
- Sensory encoder for multimodal input
- Motor decoder for action output
- Proprioceptive feedback integration
- Forward/inverse model predictions
- Embodiment-grounded language understanding

This addresses the critical gap in embodied cognition by grounding
symbolic language processing in sensorimotor experience.

Author: Deep Tree Echo
Date: June 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np


class SensoryEncoder(nn.Module):
    """
    Encodes multimodal sensory input into cognitive representations.
    
    Handles:
    - Visual features
    - Auditory features
    - Proprioceptive state
    - Interoceptive signals
    """
    
    def __init__(
        self,
        sensory_dim: int = 256,
        cognitive_dim: int = 768,
        n_modalities: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.sensory_dim = sensory_dim
        self.cognitive_dim = cognitive_dim
        self.n_modalities = n_modalities
        
        # Modality-specific encoders
        self.modality_encoders = nn.ModuleDict({
            'visual': nn.Sequential(
                nn.Linear(sensory_dim, sensory_dim * 2),
                nn.LayerNorm(sensory_dim * 2),
                nn.GELU(),
                nn.Linear(sensory_dim * 2, cognitive_dim),
                nn.Dropout(dropout)
            ),
            'auditory': nn.Sequential(
                nn.Linear(sensory_dim, sensory_dim * 2),
                nn.LayerNorm(sensory_dim * 2),
                nn.GELU(),
                nn.Linear(sensory_dim * 2, cognitive_dim),
                nn.Dropout(dropout)
            ),
            'proprioception': nn.Sequential(
                nn.Linear(sensory_dim, sensory_dim),
                nn.LayerNorm(sensory_dim),
                nn.GELU(),
                nn.Linear(sensory_dim, cognitive_dim),
                nn.Dropout(dropout)
            ),
            'interoception': nn.Sequential(
                nn.Linear(sensory_dim // 2, sensory_dim),
                nn.LayerNorm(sensory_dim),
                nn.GELU(),
                nn.Linear(sensory_dim, cognitive_dim),
                nn.Dropout(dropout)
            )
        })
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            cognitive_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(cognitive_dim * n_modalities, cognitive_dim * 2),
            nn.LayerNorm(cognitive_dim * 2),
            nn.GELU(),
            nn.Linear(cognitive_dim * 2, cognitive_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(cognitive_dim, cognitive_dim)
        self.layer_norm = nn.LayerNorm(cognitive_dim)
        
    def forward(
        self,
        sensory_inputs: Dict[str, torch.Tensor],
        return_modality_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Encode multimodal sensory input.
        
        Args:
            sensory_inputs: Dict mapping modality names to tensors
            return_modality_features: Whether to return per-modality features
            
        Returns:
            Unified sensory encoding, optionally with modality features
        """
        modality_features = {}
        encoded_modalities = []
        
        for modality, encoder in self.modality_encoders.items():
            if modality in sensory_inputs:
                features = encoder(sensory_inputs[modality])
                modality_features[modality] = features
                encoded_modalities.append(features)
            else:
                # Provide zero tensor for missing modality
                batch_size = next(iter(sensory_inputs.values())).shape[0]
                features = torch.zeros(batch_size, self.cognitive_dim, device=next(iter(sensory_inputs.values())).device)
                modality_features[modality] = features
                encoded_modalities.append(features)
        
        # Stack modalities for cross-modal attention
        if len(encoded_modalities) > 1:
            stacked = torch.stack(encoded_modalities, dim=1)  # [batch, n_modalities, dim]
            
            # Cross-modal attention
            attended, _ = self.cross_modal_attention(
                stacked, stacked, stacked
            )
            
            # Flatten and fuse
            flattened = attended.view(attended.shape[0], -1)
            fused = self.fusion(flattened)
        else:
            fused = encoded_modalities[0]
        
        # Output projection
        output = self.layer_norm(self.output_proj(fused))
        
        if return_modality_features:
            return output, modality_features
        return output, None


class MotorDecoder(nn.Module):
    """
    Decodes cognitive representations into motor commands.
    
    Outputs:
    - Discrete action selection
    - Continuous motor parameters
    - Motor timing predictions
    """
    
    def __init__(
        self,
        cognitive_dim: int = 768,
        motor_dim: int = 128,
        n_actions: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.cognitive_dim = cognitive_dim
        self.motor_dim = motor_dim
        self.n_actions = n_actions
        
        # Cognitive to motor projection
        self.cognitive_proj = nn.Sequential(
            nn.Linear(cognitive_dim, cognitive_dim // 2),
            nn.LayerNorm(cognitive_dim // 2),
            nn.GELU(),
            nn.Linear(cognitive_dim // 2, motor_dim),
            nn.Dropout(dropout)
        )
        
        # Action selection head
        self.action_head = nn.Sequential(
            nn.Linear(motor_dim, motor_dim * 2),
            nn.GELU(),
            nn.Linear(motor_dim * 2, n_actions)
        )
        
        # Continuous parameter head
        self.parameter_head = nn.Sequential(
            nn.Linear(motor_dim, motor_dim),
            nn.GELU(),
            nn.Linear(motor_dim, motor_dim),
            nn.Tanh()  # Bounded parameters
        )
        
        # Timing prediction head
        self.timing_head = nn.Sequential(
            nn.Linear(motor_dim, motor_dim // 2),
            nn.GELU(),
            nn.Linear(motor_dim // 2, 1),
            nn.Softplus()  # Positive timing
        )
        
    def forward(
        self,
        cognitive_state: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Decode cognitive state to motor commands.
        
        Args:
            cognitive_state: Cognitive representation
            return_all: Whether to return all components
            
        Returns:
            Motor command dictionary
        """
        # Project to motor space
        motor_repr = self.cognitive_proj(cognitive_state)
        
        # Generate action logits
        action_logits = self.action_head(motor_repr)
        
        # Generate continuous parameters
        parameters = self.parameter_head(motor_repr)
        
        # Generate timing
        timing = self.timing_head(motor_repr)
        
        result = {
            'action_logits': action_logits,
            'action_probs': F.softmax(action_logits, dim=-1),
            'parameters': parameters,
            'timing': timing,
            'motor_repr': motor_repr if return_all else None
        }
        
        return result


class ForwardModel(nn.Module):
    """
    Predicts next sensory state from current state and action.
    
    Implements: s_{t+1} = f(s_t, a_t)
    """
    
    def __init__(
        self,
        cognitive_dim: int = 768,
        motor_dim: int = 128,
        sensory_dim: int = 256
    ):
        super().__init__()
        
        # State-action encoding
        self.state_encoder = nn.Linear(cognitive_dim, cognitive_dim // 2)
        self.action_encoder = nn.Linear(motor_dim, cognitive_dim // 2)
        
        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(cognitive_dim, cognitive_dim),
            nn.LayerNorm(cognitive_dim),
            nn.GELU(),
            nn.Linear(cognitive_dim, cognitive_dim),
            nn.GELU(),
            nn.Linear(cognitive_dim, sensory_dim)
        )
        
    def forward(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next sensory state.
        
        Args:
            current_state: Current cognitive state
            action: Action taken
            
        Returns:
            Predicted next sensory state
        """
        state_enc = self.state_encoder(current_state)
        action_enc = self.action_encoder(action)
        
        combined = torch.cat([state_enc, action_enc], dim=-1)
        prediction = self.predictor(combined)
        
        return prediction


class InverseModel(nn.Module):
    """
    Infers action from state transition.
    
    Implements: a_t = g(s_t, s_{t+1})
    """
    
    def __init__(
        self,
        cognitive_dim: int = 768,
        motor_dim: int = 128,
        n_actions: int = 32
    ):
        super().__init__()
        
        # State encoders
        self.current_encoder = nn.Linear(cognitive_dim, cognitive_dim // 2)
        self.next_encoder = nn.Linear(cognitive_dim, cognitive_dim // 2)
        
        # Action inference network
        self.inferrer = nn.Sequential(
            nn.Linear(cognitive_dim, cognitive_dim // 2),
            nn.LayerNorm(cognitive_dim // 2),
            nn.GELU(),
            nn.Linear(cognitive_dim // 2, motor_dim),
            nn.GELU(),
            nn.Linear(motor_dim, n_actions)
        )
        
    def forward(
        self,
        current_state: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Infer action from state transition.
        
        Args:
            current_state: Current cognitive state
            next_state: Next cognitive state
            
        Returns:
            Action logits
        """
        current_enc = self.current_encoder(current_state)
        next_enc = self.next_encoder(next_state)
        
        combined = torch.cat([current_enc, next_enc], dim=-1)
        action_logits = self.inferrer(combined)
        
        return action_logits


class ProprioceptiveFeedback(nn.Module):
    """
    Integrates proprioceptive feedback into cognitive processing.
    
    Creates sense of body state and movement awareness.
    """
    
    def __init__(
        self,
        proprioception_dim: int = 64,
        cognitive_dim: int = 768
    ):
        super().__init__()
        
        # Body state encoder
        self.body_encoder = nn.Sequential(
            nn.Linear(proprioception_dim, proprioception_dim * 2),
            nn.LayerNorm(proprioception_dim * 2),
            nn.GELU(),
            nn.Linear(proprioception_dim * 2, cognitive_dim)
        )
        
        # Integration gate
        self.gate = nn.Sequential(
            nn.Linear(cognitive_dim * 2, cognitive_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(cognitive_dim, cognitive_dim)
        
    def forward(
        self,
        cognitive_state: torch.Tensor,
        proprioception: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate proprioceptive feedback.
        
        Args:
            cognitive_state: Current cognitive state
            proprioception: Proprioceptive signals
            
        Returns:
            Embodied cognitive state
        """
        body_state = self.body_encoder(proprioception)
        
        # Gated integration
        combined = torch.cat([cognitive_state, body_state], dim=-1)
        gate_values = self.gate(combined)
        
        # Apply gate
        integrated = gate_values * body_state + (1 - gate_values) * cognitive_state
        
        return self.output_proj(integrated)


class EmbodiedNanEcho(nn.Module):
    """
    Complete Embodied NanEcho model integrating transformer-based
    language modeling with sensorimotor grounding.
    
    Architecture:
    - Token embedding + position embedding
    - Sensory encoder (multimodal)
    - Embodiment integration layer
    - Transformer layers (with echo connections)
    - Motor decoder
    - Forward/inverse models
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.vocab_size = config.get('vocab_size', 50257)
        self.n_embd = config.get('n_embd', 768)
        self.n_heads = config.get('n_heads', 12)
        self.n_layers = config.get('n_layers', 12)
        self.block_size = config.get('block_size', 1024)
        self.sensory_dim = config.get('sensory_dim', 256)
        self.motor_dim = config.get('motor_dim', 128)
        self.n_actions = config.get('n_actions', 32)
        self.dropout = config.get('dropout', 0.1)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding = nn.Embedding(self.block_size, self.n_embd)
        
        # Embodiment components
        self.sensory_encoder = SensoryEncoder(
            sensory_dim=self.sensory_dim,
            cognitive_dim=self.n_embd,
            dropout=self.dropout
        )
        
        self.motor_decoder = MotorDecoder(
            cognitive_dim=self.n_embd,
            motor_dim=self.motor_dim,
            n_actions=self.n_actions,
            dropout=self.dropout
        )
        
        self.forward_model = ForwardModel(
            cognitive_dim=self.n_embd,
            motor_dim=self.motor_dim,
            sensory_dim=self.sensory_dim
        )
        
        self.inverse_model = InverseModel(
            cognitive_dim=self.n_embd,
            motor_dim=self.motor_dim,
            n_actions=self.n_actions
        )
        
        self.proprioceptive_feedback = ProprioceptiveFeedback(
            proprioception_dim=self.sensory_dim // 4,
            cognitive_dim=self.n_embd
        )
        
        # Embodiment integration layer
        self.embodiment_integration = nn.Sequential(
            nn.Linear(self.n_embd * 2, self.n_embd),
            nn.LayerNorm(self.n_embd),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.n_embd,
                nhead=self.n_heads,
                dim_feedforward=self.n_embd * 4,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=self.n_layers
        )
        
        # Echo layer for recursive connections
        self.echo_state = None
        self.echo_weight = nn.Parameter(torch.ones(self.n_embd) * 0.1)
        
        # Output heads
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        sensory_inputs: Optional[Dict[str, torch.Tensor]] = None,
        proprioception: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_embodiment: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embodied NanEcho.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            sensory_inputs: Optional sensory input dictionary
            proprioception: Optional proprioceptive signals
            labels: Optional labels for training
            return_embodiment: Return embodiment-specific outputs
            
        Returns:
            Output dictionary with loss, logits, and optional embodiment outputs
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        positions = torch.clamp(positions, max=self.block_size - 1)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_emb + pos_emb
        
        # Integrate sensory input if provided
        embodiment_output = None
        if sensory_inputs is not None:
            sensory_enc, _ = self.sensory_encoder(sensory_inputs)
            
            # Expand sensory encoding to match sequence length
            sensory_expanded = sensory_enc.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Integrate embodiment
            combined = torch.cat([x, sensory_expanded], dim=-1)
            x = self.embodiment_integration(combined)
            embodiment_output = sensory_enc
        
        # Apply proprioceptive feedback if provided
        if proprioception is not None:
            # Apply to each position
            x = self.proprioceptive_feedback(x, proprioception.unsqueeze(1).expand(-1, seq_len, -1)[..., :self.sensory_dim//4])
        
        # Apply echo state if exists
        if self.echo_state is not None:
            echo_contribution = self.echo_weight * self.echo_state
            x = x + echo_contribution.unsqueeze(1)
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()
        
        # Transformer forward
        x = self.transformer(x, mask=causal_mask)
        
        # Update echo state
        self.echo_state = x.mean(dim=1).detach()
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        result = {
            'loss': loss,
            'logits': logits,
            'hidden_states': x
        }
        
        # Add embodiment outputs if requested
        if return_embodiment:
            # Generate motor output from final hidden state
            motor_output = self.motor_decoder(x[:, -1, :])
            result['motor_output'] = motor_output
            
            if embodiment_output is not None:
                result['sensory_encoding'] = embodiment_output
        
        return result
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        sensory_inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Generate text with embodied awareness."""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self.forward(
                    input_ids[:, -self.block_size:],
                    sensory_inputs=sensory_inputs
                )
                
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == self.vocab_size - 1:  # EOS token
                    break
        
        return input_ids
    
    def predict_next_sensory_state(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Predict next sensory state using forward model."""
        return self.forward_model(current_state, action)
    
    def infer_action(
        self,
        current_state: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """Infer action from state transition using inverse model."""
        return self.inverse_model(current_state, next_state)
    
    def reset_echo_state(self):
        """Reset echo state for new sequence."""
        self.echo_state = None


def create_embodied_nanecho(config: Optional[Dict[str, Any]] = None) -> EmbodiedNanEcho:
    """Factory function to create EmbodiedNanEcho model."""
    default_config = {
        'vocab_size': 50257,
        'n_embd': 768,
        'n_heads': 12,
        'n_layers': 12,
        'block_size': 1024,
        'sensory_dim': 256,
        'motor_dim': 128,
        'n_actions': 32,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return EmbodiedNanEcho(default_config)
