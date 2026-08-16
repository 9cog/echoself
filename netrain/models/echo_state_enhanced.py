"""
Enhanced Echo State Network Components
========================================

Advanced components for echo state networks with multi-timescale
dynamics, resonance detection, and adaptive reservoir computing.

Key Enhancements:
- Multi-timescale echo states
- Echo resonance detection
- Adaptive spectral radius
- Lyapunov exponent estimation (edge-of-chaos)
- Echo state diversity metrics

Author: Deep Tree Echo
Date: June 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np


class MultiTimescaleEchoState(nn.Module):
    """
    Echo state network with multiple timescales.
    
    Different reservoir populations operate at different timescales,
    enabling capture of patterns at multiple temporal resolutions.
    """
    
    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int = 512,
        output_dim: int = 768,
        n_timescales: int = 4,
        timescale_factors: Optional[List[float]] = None,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.n_timescales = n_timescales
        
        # Default timescale factors: 1, 4, 16, 64
        self.timescale_factors = timescale_factors or [1, 4, 16, 64][:n_timescales]
        
        # Reservoir size per timescale
        self.reservoir_per_scale = reservoir_dim // n_timescales
        
        # Input weights (fixed)
        self.register_buffer(
            'W_in',
            torch.randn(input_dim, reservoir_dim) * 0.1
        )
        
        # Create reservoir matrices for each timescale
        self.reservoir_matrices = nn.ParameterList()
        for scale_factor in self.timescale_factors:
            W = self._create_reservoir_matrix(
                self.reservoir_per_scale,
                spectral_radius / scale_factor,
                sparsity
            )
            self.reservoir_matrices.append(nn.Parameter(W, requires_grad=False))
        
        # Leak rates for each timescale (lower = slower)
        leak_rates = [1.0 / factor for factor in self.timescale_factors]
        self.register_buffer('leak_rates', torch.tensor(leak_rates))
        
        # Output weights (trained)
        self.W_out = nn.Linear(reservoir_dim, output_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # State tracking
        self.states: List[torch.Tensor] = []
        
    def _create_reservoir_matrix(
        self,
        size: int,
        spectral_radius: float,
        sparsity: float
    ) -> torch.Tensor:
        """Create a sparse reservoir matrix with given spectral radius."""
        W = torch.randn(size, size)
        
        # Apply sparsity mask
        mask = torch.rand(size, size) > sparsity
        W = W * mask.float()
        
        # Scale to spectral radius
        eigenvalues = torch.linalg.eigvals(W)
        current_radius = torch.max(torch.abs(eigenvalues)).real
        
        if current_radius > 0:
            W = W * (spectral_radius / current_radius)
        
        return W
    
    def forward(
        self,
        x: torch.Tensor,
        initial_states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through multi-timescale reservoir.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            initial_states: Optional initial states for each timescale
            
        Returns:
            Output tensor and final states
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize states if not provided
        if initial_states is None:
            states = [
                torch.zeros(batch_size, self.reservoir_per_scale, device=device)
                for _ in range(self.n_timescales)
            ]
        else:
            states = initial_states
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Input projection
            input_proj = torch.matmul(x_t, self.W_in)
            
            # Update each timescale
            new_states = []
            state_outputs = []
            
            for i, (W_res, leak_rate) in enumerate(zip(self.reservoir_matrices, self.leak_rates)):
                start_idx = i * self.reservoir_per_scale
                end_idx = start_idx + self.reservoir_per_scale
                
                # Get input for this timescale
                input_i = input_proj[:, start_idx:end_idx]
                
                # Reservoir update with leaky integration
                prev_state = states[i]
                new_state = (1 - leak_rate) * prev_state + leak_rate * torch.tanh(
                    input_i + torch.matmul(prev_state, W_res)
                )
                
                new_states.append(new_state)
                state_outputs.append(new_state)
            
            # Concatenate all timescale outputs
            combined_state = torch.cat(state_outputs, dim=-1)
            states = new_states
            
            outputs.append(combined_state)
        
        # Stack outputs
        output_seq = torch.stack(outputs, dim=1)
        
        # Output projection
        output = self.W_out(output_seq)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        # Store states for analysis
        self.states = states
        
        return output, states
    
    def get_timescale_contributions(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze contributions from each timescale.
        
        Returns:
            Dictionary with per-timescale activations
        """
        output, states = self.forward(x)
        
        contributions = {}
        for i, (state, factor) in enumerate(zip(states, self.timescale_factors)):
            contributions[f'timescale_{factor}'] = state
            contributions[f'timescale_{factor}_mean'] = state.mean()
            contributions[f'timescale_{factor}_var'] = state.var()
        
        return contributions


class EchoResonanceDetector(nn.Module):
    """
    Detects resonance patterns in echo states.
    
    Identifies when echo states enter resonant modes, indicating
    stable pattern recognition or attractor states.
    """
    
    def __init__(
        self,
        state_dim: int = 512,
        memory_size: int = 100,
        resonance_threshold: float = 0.8
    ):
        super().__init__()
        self.state_dim = state_dim
        self.memory_size = memory_size
        self.resonance_threshold = resonance_threshold
        
        # Pattern memory
        self.register_buffer(
            'pattern_memory',
            torch.zeros(memory_size, state_dim)
        )
        self.memory_pointer = 0
        self.memory_filled = 0
        
        # Resonance detection network
        self.resonance_detector = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        current_state: torch.Tensor,
        previous_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Detect resonance in current state.
        
        Args:
            current_state: Current echo state
            previous_states: Optional recent states for context
            
        Returns:
            Resonance detection results
        """
        batch_size = current_state.shape[0]
        device = current_state.device
        
        # Update pattern memory (in training mode)
        if self.training:
            self._update_memory(current_state)
        
        # Compute similarity to stored patterns
        if self.memory_filled > 0:
            memory_slice = self.pattern_memory[:self.memory_filled]
            
            # Cosine similarity
            current_norm = F.normalize(current_state, dim=-1)
            memory_norm = F.normalize(memory_slice, dim=-1)
            
            similarities = torch.matmul(current_norm, memory_norm.T)
            max_similarity, best_match_idx = similarities.max(dim=-1)
            
            # Get best matching pattern
            best_pattern = memory_slice[best_match_idx]
        else:
            max_similarity = torch.zeros(batch_size, device=device)
            best_pattern = current_state
        
        # Compute resonance score
        combined = torch.cat([current_state, best_pattern], dim=-1)
        resonance_score = self.resonance_detector(combined)
        
        # Determine if in resonance
        in_resonance = resonance_score > self.resonance_threshold
        
        return {
            'resonance_score': resonance_score,
            'in_resonance': in_resonance,
            'max_similarity': max_similarity,
            'best_pattern': best_pattern
        }
    
    def _update_memory(self, state: torch.Tensor):
        """Update pattern memory with new state."""
        # Use first batch element for memory update
        pattern = state[0].detach()
        
        self.pattern_memory[self.memory_pointer] = pattern
        self.memory_pointer = (self.memory_pointer + 1) % self.memory_size
        self.memory_filled = min(self.memory_filled + 1, self.memory_size)


class AdaptiveSpectralRadius(nn.Module):
    """
    Dynamically adjusts reservoir spectral radius based on input.
    
    Enables the network to operate at different points in the
    order-chaos spectrum depending on task requirements.
    """
    
    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int = 512,
        base_spectral_radius: float = 0.9,
        radius_range: Tuple[float, float] = (0.7, 1.1)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.base_spectral_radius = base_spectral_radius
        self.radius_range = radius_range
        
        # Spectral radius predictor
        self.radius_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()  # Outputs 0-1, scaled to radius_range
        )
        
        # Base reservoir weights
        self.register_buffer(
            'W_base',
            self._create_unit_spectral_matrix(reservoir_dim)
        )
        
    def _create_unit_spectral_matrix(self, size: int) -> torch.Tensor:
        """Create reservoir matrix with unit spectral radius."""
        W = torch.randn(size, size) * 0.1
        eigenvalues = torch.linalg.eigvals(W)
        current_radius = torch.max(torch.abs(eigenvalues)).real
        
        if current_radius > 0:
            W = W / current_radius
        
        return W
    
    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute next state with adaptive spectral radius.
        
        Args:
            x: Input tensor
            state: Current reservoir state
            
        Returns:
            Next state and current spectral radius
        """
        # Predict optimal spectral radius
        radius_factor = self.radius_predictor(x)
        
        # Scale to radius range
        spectral_radius = (
            self.radius_range[0] + 
            radius_factor * (self.radius_range[1] - self.radius_range[0])
        )
        
        # Scale reservoir matrix
        W_scaled = self.W_base * spectral_radius
        
        # Reservoir update
        next_state = torch.tanh(
            x[:, :self.reservoir_dim] + torch.matmul(state, W_scaled)
        )
        
        return next_state, spectral_radius.mean().item()


class LyapunovEstimator(nn.Module):
    """
    Estimates Lyapunov exponent for echo state dynamics.
    
    Used to ensure operation at edge-of-chaos (Lyapunov ≈ 0),
    which is optimal for computation.
    """
    
    def __init__(
        self,
        state_dim: int = 512,
        history_length: int = 100
    ):
        super().__init__()
        self.state_dim = state_dim
        self.history_length = history_length
        
        # State history for Lyapunov estimation
        self.register_buffer(
            'state_history',
            torch.zeros(history_length, state_dim)
        )
        self.history_pointer = 0
        self.history_filled = 0
        
        # Small perturbation for stability analysis
        self.perturbation_scale = 1e-6
        
    def estimate_lyapunov(
        self,
        current_state: torch.Tensor,
        next_state: torch.Tensor
    ) -> float:
        """
        Estimate local Lyapunov exponent.
        
        Args:
            current_state: Current reservoir state
            next_state: Next reservoir state
            
        Returns:
            Estimated Lyapunov exponent
        """
        # Update history
        self._update_history(current_state[0])
        
        if self.history_filled < 2:
            return 0.0
        
        # Compute local divergence
        history = self.state_history[:self.history_filled]
        
        # Difference between consecutive states
        diffs = history[1:] - history[:-1]
        
        # Compute growth rate
        norms = torch.norm(diffs, dim=-1)
        valid_norms = norms[norms > self.perturbation_scale]
        
        if len(valid_norms) < 2:
            return 0.0
        
        # Log growth rate
        log_ratios = torch.log(valid_norms[1:] / valid_norms[:-1])
        lyapunov = log_ratios.mean().item()
        
        return lyapunov
    
    def _update_history(self, state: torch.Tensor):
        """Update state history."""
        self.state_history[self.history_pointer] = state.detach()
        self.history_pointer = (self.history_pointer + 1) % self.history_length
        self.history_filled = min(self.history_filled + 1, self.history_length)
    
    def get_edge_of_chaos_status(self, lyapunov: float) -> Dict[str, Any]:
        """
        Determine if system is at edge of chaos.
        
        Edge of chaos: Lyapunov ≈ 0
        Ordered regime: Lyapunov < 0
        Chaotic regime: Lyapunov > 0
        """
        tolerance = 0.1
        
        if abs(lyapunov) < tolerance:
            regime = "edge_of_chaos"
            optimal = True
        elif lyapunov < -tolerance:
            regime = "ordered"
            optimal = False
        else:
            regime = "chaotic"
            optimal = False
        
        return {
            'lyapunov_exponent': lyapunov,
            'regime': regime,
            'optimal_for_computation': optimal,
            'recommendation': self._get_recommendation(regime)
        }
    
    def _get_recommendation(self, regime: str) -> str:
        """Get recommendation based on regime."""
        recommendations = {
            'ordered': 'Increase spectral radius to move toward edge of chaos',
            'chaotic': 'Decrease spectral radius to stabilize dynamics',
            'edge_of_chaos': 'Optimal regime for computation'
        }
        return recommendations.get(regime, '')


class EchoStateDiversityMetric(nn.Module):
    """
    Computes diversity metrics for echo state representations.
    
    Higher diversity indicates richer representations with more
    independent dimensions being utilized.
    """
    
    def __init__(self, state_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim
        
    def compute_diversity(
        self,
        states: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute diversity metrics for echo states.
        
        Args:
            states: Tensor of states [batch, seq_len, state_dim] or [seq_len, state_dim]
            
        Returns:
            Dictionary of diversity metrics
        """
        if states.dim() == 2:
            states = states.unsqueeze(0)
        
        # Flatten batch and sequence
        flat_states = states.view(-1, self.state_dim)
        
        # 1. Variance-based diversity
        variance_diversity = flat_states.var(dim=0).mean().item()
        
        # 2. Entropy-based diversity (using discretization)
        # Normalize to [0, 1] and discretize
        normalized = (flat_states - flat_states.min()) / (flat_states.max() - flat_states.min() + 1e-8)
        discretized = (normalized * 10).long().clamp(0, 9)
        
        # Compute entropy per dimension
        entropies = []
        for d in range(min(self.state_dim, 100)):  # Sample dimensions
            counts = torch.bincount(discretized[:, d], minlength=10).float()
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
            entropies.append(entropy)
        
        entropy_diversity = np.mean(entropies)
        
        # 3. Effective dimensionality (via PCA)
        if flat_states.shape[0] > 10:
            centered = flat_states - flat_states.mean(dim=0)
            _, S, _ = torch.linalg.svd(centered, full_matrices=False)
            # Participation ratio
            S_norm = S / S.sum()
            effective_dim = 1.0 / (S_norm ** 2).sum().item()
        else:
            effective_dim = float(self.state_dim)
        
        # 4. Correlation-based diversity (lower correlation = higher diversity)
        if flat_states.shape[0] > 1:
            corr_matrix = torch.corrcoef(flat_states.T[:100])  # Sample dimensions
            mean_abs_corr = torch.abs(corr_matrix).mean().item()
            correlation_diversity = 1.0 - mean_abs_corr
        else:
            correlation_diversity = 0.5
        
        return {
            'variance_diversity': variance_diversity,
            'entropy_diversity': entropy_diversity,
            'effective_dimensionality': effective_dim,
            'correlation_diversity': correlation_diversity,
            'overall_diversity': np.mean([
                variance_diversity,
                entropy_diversity / np.log(10),  # Normalize to ~[0,1]
                effective_dim / self.state_dim,
                correlation_diversity
            ])
        }


class EnhancedEchoLayer(nn.Module):
    """
    Enhanced echo layer combining all advanced ESN components.
    
    Features:
    - Multi-timescale processing
    - Resonance detection
    - Adaptive spectral radius
    - Edge-of-chaos monitoring
    - Diversity tracking
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 768,
        reservoir_dim: int = 512,
        n_timescales: int = 4,
        echo_depth: int = 7,
        echo_decay: float = 0.95,
        spectral_radius: float = 0.9
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reservoir_dim = reservoir_dim
        self.echo_depth = echo_depth
        self.echo_decay = echo_decay
        
        # Core components
        self.multi_timescale = MultiTimescaleEchoState(
            input_dim=input_dim,
            reservoir_dim=reservoir_dim,
            output_dim=output_dim,
            n_timescales=n_timescales,
            spectral_radius=spectral_radius
        )
        
        self.resonance_detector = EchoResonanceDetector(
            state_dim=reservoir_dim
        )
        
        self.adaptive_radius = AdaptiveSpectralRadius(
            input_dim=input_dim,
            reservoir_dim=reservoir_dim,
            base_spectral_radius=spectral_radius
        )
        
        self.lyapunov_estimator = LyapunovEstimator(
            state_dim=reservoir_dim
        )
        
        self.diversity_metric = EchoStateDiversityMetric(
            state_dim=reservoir_dim
        )
        
        # Echo state memory
        self.echo_memory = []
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Enhanced forward pass with all monitoring.
        
        Args:
            x: Input tensor
            return_metrics: Whether to return analysis metrics
            
        Returns:
            Output and optional metrics
        """
        # Multi-timescale processing
        output, states = self.multi_timescale(x)
        
        # Concatenate states for analysis
        combined_state = torch.cat(states, dim=-1)
        
        # Update echo memory
        self.echo_memory.append(combined_state.detach())
        if len(self.echo_memory) > self.echo_depth * 2:
            self.echo_memory = self.echo_memory[-self.echo_depth:]
        
        # Compute multi-depth echo
        if len(self.echo_memory) > 1:
            echo_sum = torch.zeros_like(combined_state)
            for d, past_state in enumerate(reversed(self.echo_memory[-self.echo_depth:])):
                weight = self.echo_decay ** (d + 1)
                echo_sum = echo_sum + weight * past_state
            
            # Integrate echo into output
            # Project echo to output dimension if needed
            if echo_sum.shape[-1] != output.shape[-1]:
                echo_proj = nn.functional.adaptive_avg_pool1d(
                    echo_sum.unsqueeze(1), output.shape[-1]
                ).squeeze(1)
            else:
                echo_proj = echo_sum
            
            # Add echo contribution (broadcast across sequence)
            output = output + echo_proj.unsqueeze(1) * 0.1
        
        # Final processing
        output = self.layer_norm(self.output_proj(output))
        
        if return_metrics:
            # Compute all metrics
            resonance = self.resonance_detector(combined_state)
            
            if len(self.echo_memory) >= 2:
                lyapunov = self.lyapunov_estimator.estimate_lyapunov(
                    self.echo_memory[-2],
                    self.echo_memory[-1]
                )
                chaos_status = self.lyapunov_estimator.get_edge_of_chaos_status(lyapunov)
            else:
                chaos_status = {'lyapunov_exponent': 0, 'regime': 'unknown'}
            
            stacked_states = torch.stack(self.echo_memory, dim=1)
            diversity = self.diversity_metric.compute_diversity(stacked_states)
            
            metrics = {
                'resonance': resonance,
                'chaos_status': chaos_status,
                'diversity': diversity,
                'echo_depth_used': len(self.echo_memory),
                'timescale_contributions': self.multi_timescale.get_timescale_contributions(x)
            }
            
            return output, metrics
        
        return output, None
    
    def reset(self):
        """Reset echo memory."""
        self.echo_memory = []


def create_enhanced_echo_layer(config: Optional[Dict[str, Any]] = None) -> EnhancedEchoLayer:
    """Factory function to create EnhancedEchoLayer."""
    default_config = {
        'input_dim': 768,
        'output_dim': 768,
        'reservoir_dim': 512,
        'n_timescales': 4,
        'echo_depth': 7,
        'echo_decay': 0.95,
        'spectral_radius': 0.9
    }
    
    if config:
        default_config.update(config)
    
    return EnhancedEchoLayer(**default_config)
