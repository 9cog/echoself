"""
Virtual Endocrine System for Deep Tree Echo
============================================

Replaces fixed activation functions with learned neural networks that act as
a virtual endocrine system, dynamically modulating the transformer's behavior
based on long-term "hormonal" signals from an Echo State Network (ESN) reservoir.

Architecture:
  EndocrineReservoir (ESN) → Hormone Vector (Cortisol, Dopamine, Serotonin)
       ↓                          ↓                    ↓
  DynamicRNNGate          DynamicGNNMemory      DynamicCNNActivation
  (Triad modulation)      (Pentad modulation)   (Dyad/MLP modulation)

The ESN maintains persistent state across the entire sequence, providing
the transformer with a "felt sense" of its own processing history — a
slow-timescale modulation that gives the system something analogous to
mood, arousal, and cognitive readiness.

Biological Analogy:
  - Cortisol → Stress/Salience → Sharpens working memory (Triad)
  - Dopamine → Reward/Resonance → Strengthens memory consolidation (Pentad)
  - Serotonin → Plasticity/Exploration → Opens the reservoir to new patterns
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class EndocrineConfig:
    """Configuration for the Virtual Endocrine System."""
    # ESN Reservoir parameters
    reservoir_size: int = 256        # Number of reservoir neurons
    spectral_radius: float = 0.95    # Edge of chaos (modulated by serotonin)
    leak_rate: float = 0.3           # Base leak rate (modulated by serotonin)
    input_scaling: float = 0.1       # Input weight scaling
    sparsity: float = 0.9            # Reservoir connectivity sparsity

    # Hormone dimensions
    n_hormones: int = 3              # Cortisol, Dopamine, Serotonin
    hormone_dim: int = 64            # Internal processing dim for each hormone

    # Dynamic activation parameters
    n_embd: int = 768                # Must match transformer embedding dim
    rnn_hidden: int = 128            # GRU hidden size for RNN activation
    cnn_kernel_sizes: Tuple[int, ...] = (3, 5, 7)  # Multi-scale CNN kernels
    gnn_message_dim: int = 128       # GNN message passing dimension

    # Temporal dynamics
    hormone_momentum: float = 0.9    # Exponential moving average for hormone smoothing
    hormone_noise: float = 0.01      # Stochastic noise for exploration


class EchoStateReservoir(nn.Module):
    """
    Echo State Network (ESN) Reservoir — the "glandular" backend.

    Maintains a persistent, non-resetting state that accumulates information
    across the entire sequence (and optionally across sequences during generation).
    Produces a 3-dimensional hormone vector that modulates the transformer.

    The reservoir is NOT trained by backpropagation — only the readout layer is.
    This gives it the stability of a dynamical system while allowing the
    modulation to be learned end-to-end.
    """

    def __init__(self, config: EndocrineConfig):
        super().__init__()
        self.config = config
        self.reservoir_size = config.reservoir_size

        # Input weights (fixed, not trained)
        W_in = torch.randn(config.reservoir_size, config.n_embd) * config.input_scaling
        self.register_buffer("W_in", W_in)

        # Reservoir weights (fixed, sparse, scaled to spectral radius)
        W_res = torch.randn(config.reservoir_size, config.reservoir_size)
        # Apply sparsity mask
        mask = (torch.rand_like(W_res) > config.sparsity).float()
        W_res = W_res * mask
        # Scale to desired spectral radius
        eigenvalues = torch.linalg.eigvals(W_res).abs()
        if eigenvalues.max() > 0:
            W_res = W_res * (config.spectral_radius / eigenvalues.max())
        self.register_buffer("W_res", W_res)

        # Feedback weights (fixed)
        W_fb = torch.randn(config.reservoir_size, config.n_hormones) * 0.01
        self.register_buffer("W_fb", W_fb)

        # Learnable readout: reservoir state → hormone vector
        self.readout = nn.Sequential(
            nn.Linear(config.reservoir_size, config.hormone_dim),
            nn.LayerNorm(config.hormone_dim),
            nn.GELU(),
            nn.Linear(config.hormone_dim, config.n_hormones),
            nn.Sigmoid()  # Hormones are bounded [0, 1]
        )

        # Serotonin-modulated leak rate and spectral radius
        self.leak_modulator = nn.Linear(1, 2)  # serotonin → (leak_rate_delta, radius_delta)

        # Persistent state (not a parameter, but maintained across forward calls)
        self.register_buffer("reservoir_state", torch.zeros(1, config.reservoir_size))
        self.register_buffer("hormone_ema", torch.ones(1, config.n_hormones) * 0.5)

    def reset_state(self, batch_size: int = 1):
        """Reset reservoir state (e.g., at start of new sequence)."""
        device = self.W_in.device
        self.reservoir_state = torch.zeros(batch_size, self.reservoir_size, device=device)
        self.hormone_ema = torch.ones(batch_size, self.config.n_hormones, device=device) * 0.5

    def forward(
        self,
        x: torch.Tensor,
        prev_hormones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through the reservoir and produce hormone vector.

        Args:
            x: Input from transformer (B, C) — typically pooled layer output
            prev_hormones: Previous hormone vector for feedback (B, 3)

        Returns:
            hormones: Current hormone levels (B, 3) — [cortisol, dopamine, serotonin]
            reservoir_state: Current reservoir state (B, reservoir_size)
        """
        B = x.shape[0]

        # Ensure state has correct batch size
        if self.reservoir_state.shape[0] != B:
            self.reset_state(B)

        # Get current serotonin to modulate dynamics
        if prev_hormones is not None:
            serotonin = prev_hormones[:, 2:3]  # (B, 1)
            modulation = self.leak_modulator(serotonin)  # (B, 2)
            leak_delta = torch.tanh(modulation[:, 0:1]) * 0.2  # ±0.2
            # Dynamic leak rate
            leak_rate = (self.config.leak_rate + leak_delta).clamp(0.05, 0.95)
        else:
            leak_rate = torch.full((B, 1), self.config.leak_rate, device=x.device)

        # Reservoir update equation:
        # h(t) = (1-α)·h(t-1) + α·tanh(W_in·x + W_res·h(t-1) + W_fb·y(t-1))
        input_drive = torch.mm(x, self.W_in.t())  # (B, reservoir_size)
        recurrent_drive = torch.mm(self.reservoir_state, self.W_res.t())  # (B, reservoir_size)

        feedback_drive = torch.zeros_like(input_drive)
        if prev_hormones is not None:
            feedback_drive = torch.mm(prev_hormones, self.W_fb.t())

        # Non-linear activation of the reservoir
        pre_activation = input_drive + recurrent_drive + feedback_drive
        new_state = torch.tanh(pre_activation)

        # Leaky integration
        self.reservoir_state = (1 - leak_rate) * self.reservoir_state + leak_rate * new_state

        # Add exploration noise (scaled by serotonin)
        if self.training:
            noise_scale = self.config.hormone_noise
            if prev_hormones is not None:
                noise_scale = noise_scale * (1 + prev_hormones[:, 2:3])  # More noise with high serotonin
            noise = torch.randn_like(self.reservoir_state) * noise_scale
            self.reservoir_state = self.reservoir_state + noise

        # Readout: reservoir state → hormone vector
        raw_hormones = self.readout(self.reservoir_state)  # (B, 3)

        # Exponential moving average for temporal smoothing
        momentum = self.config.hormone_momentum
        self.hormone_ema = momentum * self.hormone_ema + (1 - momentum) * raw_hormones

        return self.hormone_ema.clone(), self.reservoir_state.clone()


class DynamicRNNGate(nn.Module):
    """
    Dynamic RNN-based Activation Gate — replaces fixed gating in TernaryAttention.

    A small GRU cell maintains short-term context within the working memory (Triad).
    The Cortisol hormone modulates the gate bias, making the system more or less
    reactive to incoming information.

    High Cortisol → Gates close → Focus narrows → Working memory becomes selective
    Low Cortisol → Gates open → Broad attention → Working memory is receptive
    """

    def __init__(self, config: EndocrineConfig):
        super().__init__()
        self.n_embd = config.n_embd

        # GRU cell for temporal context within the gate
        self.gru = nn.GRUCell(config.n_embd, config.rnn_hidden)

        # Cortisol modulation: hormone → gate bias
        self.cortisol_proj = nn.Sequential(
            nn.Linear(1, config.rnn_hidden),
            nn.Tanh()
        )

        # Output projection: GRU hidden → gate values
        self.gate_proj = nn.Sequential(
            nn.Linear(config.rnn_hidden, config.n_embd),
            nn.Sigmoid()
        )

        # Persistent GRU hidden state
        self.register_buffer("gru_state", torch.zeros(1, config.rnn_hidden))

    def reset_state(self, batch_size: int = 1):
        device = self.gru_state.device
        self.gru_state = torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, x: torch.Tensor, cortisol: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, C) or (B, C)
            cortisol: Cortisol level (B, 1) in [0, 1]

        Returns:
            Gated output (same shape as x)
        """
        squeeze_needed = False
        if x.dim() == 3:
            B, T, C = x.shape
            # Process each timestep through GRU
            outputs = []
            if self.gru_state.shape[0] != B:
                self.reset_state(B)

            # Cortisol bias (applied to GRU hidden state)
            cortisol_bias = self.cortisol_proj(cortisol)  # (B, rnn_hidden)

            for t in range(T):
                self.gru_state = self.gru(x[:, t, :], self.gru_state + cortisol_bias)
                gate = self.gate_proj(self.gru_state)  # (B, C)
                outputs.append(x[:, t, :] * gate)

            return torch.stack(outputs, dim=1)
        else:
            B, C = x.shape
            if self.gru_state.shape[0] != B:
                self.reset_state(B)
            cortisol_bias = self.cortisol_proj(cortisol)
            self.gru_state = self.gru(x, self.gru_state + cortisol_bias)
            gate = self.gate_proj(self.gru_state)
            return x * gate


class DynamicGNNMemory(nn.Module):
    """
    Dynamic GNN-based Memory Activation — replaces fixed memory retrieval in QuinaryIntegration.

    Treats the memory bank as a graph where:
    - Nodes = memory slots
    - Edges = learned associations (modulated by Dopamine)
    - Message passing = memory retrieval with context-dependent routing

    High Dopamine → Stronger edge weights → Richer memory retrieval → More identity grounding
    Low Dopamine → Weaker edges → Shallow retrieval → More novel/exploratory responses
    """

    def __init__(self, config: EndocrineConfig, memory_size: int = 1024):
        super().__init__()
        self.n_embd = config.n_embd
        self.memory_size = memory_size
        self.message_dim = config.gnn_message_dim

        # Message function: computes messages between query and memory nodes
        self.message_fn = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.gnn_message_dim),
            nn.GELU(),
            nn.Linear(config.gnn_message_dim, config.gnn_message_dim)
        )

        # Aggregation function: combines incoming messages
        self.aggregate_fn = nn.Sequential(
            nn.Linear(config.gnn_message_dim, config.n_embd),
            nn.LayerNorm(config.n_embd)
        )

        # Dopamine modulation: scales edge weights
        self.dopamine_edge_scale = nn.Sequential(
            nn.Linear(1, config.gnn_message_dim),
            nn.Sigmoid()
        )

        # Update function: updates node representation
        self.update_fn = nn.GRUCell(config.n_embd, config.n_embd)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        dopamine: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: Query tensor (B, T, C)
            memory: Memory bank (B, M, C)
            dopamine: Dopamine level (B, 1) in [0, 1]

        Returns:
            Retrieved and integrated memory (B, T, C)
        """
        B, T, C = query.shape
        M = memory.shape[1]

        # Compute messages from each memory node to each query position
        # Expand for pairwise computation
        q_expanded = query.unsqueeze(2).expand(B, T, M, C)  # (B, T, M, C)
        m_expanded = memory.unsqueeze(1).expand(B, T, M, C)  # (B, T, M, C)

        # Concatenate for message computation
        pairs = torch.cat([q_expanded, m_expanded], dim=-1)  # (B, T, M, 2C)
        messages = self.message_fn(pairs)  # (B, T, M, msg_dim)

        # Dopamine modulates edge weights (how strongly memories connect)
        edge_scale = self.dopamine_edge_scale(dopamine)  # (B, msg_dim)
        edge_scale = edge_scale.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, msg_dim)
        messages = messages * edge_scale

        # Attention-weighted aggregation
        attn_scores = (messages * messages).sum(dim=-1) / math.sqrt(self.message_dim)  # (B, T, M)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, M)

        # Weighted sum of messages
        aggregated = torch.matmul(attn_weights.unsqueeze(-2), messages).squeeze(-2)  # (B, T, msg_dim)
        aggregated = self.aggregate_fn(aggregated)  # (B, T, C)

        # Update query representation with retrieved information
        output = []
        for t in range(T):
            updated = self.update_fn(aggregated[:, t, :], query[:, t, :])
            output.append(updated)

        return torch.stack(output, dim=1)


class DynamicCNNActivation(nn.Module):
    """
    Dynamic CNN-based Activation — replaces fixed GELU in Feed-Forward MLPs.

    Uses multi-scale 1D convolutions across the sequence dimension to capture
    local patterns at different granularities. The Serotonin hormone modulates
    the receptive field by weighting different kernel sizes.

    High Serotonin → Larger receptive field → More context integration → Exploration
    Low Serotonin → Smaller receptive field → Local focus → Exploitation
    """

    def __init__(self, config: EndocrineConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.kernel_sizes = config.cnn_kernel_sizes

        # Multi-scale convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(
                config.n_embd, config.n_embd,
                kernel_size=k, padding=k // 2, groups=config.n_embd // 4
            )
            for k in config.cnn_kernel_sizes
        ])

        # Serotonin modulation: determines kernel weighting
        self.serotonin_router = nn.Sequential(
            nn.Linear(1, len(config.cnn_kernel_sizes)),
            nn.Softmax(dim=-1)
        )

        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.layer_norm = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor, serotonin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, C)
            serotonin: Serotonin level (B, 1) in [0, 1]

        Returns:
            Activated output (B, T, C)
        """
        B, T, C = x.shape
        residual = x

        # Transpose for conv1d: (B, C, T)
        x_conv = x.transpose(1, 2)

        # Apply each kernel size
        conv_outputs = []
        for conv in self.convs:
            out = conv(x_conv)  # (B, C, T)
            conv_outputs.append(out)

        # Stack: (B, n_kernels, C, T)
        stacked = torch.stack(conv_outputs, dim=1)

        # Serotonin determines which kernel sizes to weight
        # High serotonin → favor larger kernels (more context)
        weights = self.serotonin_router(serotonin)  # (B, n_kernels)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, n_kernels, 1, 1)

        # Weighted combination
        combined = (stacked * weights).sum(dim=1)  # (B, C, T)
        combined = combined.transpose(1, 2)  # (B, T, C)

        # Non-linearity (still needed, but now context-aware)
        activated = F.gelu(combined)

        # Output projection + residual
        output = self.out_proj(activated)
        return self.layer_norm(residual + output)


class VirtualEndocrineSystem(nn.Module):
    """
    Complete Virtual Endocrine System.

    Integrates the ESN reservoir with all three dynamic activation modules,
    providing a unified interface for the 2-3-5 transformer to query and
    be modulated by.

    Usage:
        endocrine = VirtualEndocrineSystem(config)

        # At each transformer layer:
        hormones, reservoir_state = endocrine.step(layer_output)

        # Use hormones to modulate:
        gated = endocrine.rnn_gate(x, hormones[:, 0:1])     # Cortisol gates triad
        retrieved = endocrine.gnn_memory(q, mem, hormones[:, 1:2])  # Dopamine gates pentad
        activated = endocrine.cnn_activation(x, hormones[:, 2:3])   # Serotonin gates MLP
    """

    def __init__(self, config: EndocrineConfig):
        super().__init__()
        self.config = config

        # The reservoir (ESN backend)
        self.reservoir = EchoStateReservoir(config)

        # Dynamic activation modules
        self.rnn_gate = DynamicRNNGate(config)
        self.gnn_memory = DynamicGNNMemory(config)
        self.cnn_activation = DynamicCNNActivation(config)

        # Input projection: reduce transformer output to reservoir input
        self.input_proj = nn.Linear(config.n_embd, config.n_embd)

        # Hormone history (for visualization and monitoring)
        self.hormone_history: list = []

    def reset(self, batch_size: int = 1):
        """Reset all internal states."""
        self.reservoir.reset_state(batch_size)
        self.rnn_gate.reset_state(batch_size)
        self.hormone_history = []

    def step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one step through the endocrine system.

        Args:
            x: Pooled transformer layer output (B, C)

        Returns:
            hormones: Current hormone levels (B, 3) — [cortisol, dopamine, serotonin]
            reservoir_state: Current ESN state (B, reservoir_size)
        """
        # Project input
        x_proj = self.input_proj(x)

        # Get previous hormones for feedback
        prev_hormones = self.reservoir.hormone_ema.clone()

        # Step the reservoir
        hormones, state = self.reservoir(x_proj, prev_hormones)

        # Record history
        if not self.training:
            self.hormone_history.append(hormones.detach().cpu())

        return hormones, state

    @property
    def cortisol(self) -> torch.Tensor:
        """Current cortisol level."""
        return self.reservoir.hormone_ema[:, 0:1]

    @property
    def dopamine(self) -> torch.Tensor:
        """Current dopamine level."""
        return self.reservoir.hormone_ema[:, 1:2]

    @property
    def serotonin(self) -> torch.Tensor:
        """Current serotonin level."""
        return self.reservoir.hormone_ema[:, 2:3]

    def get_hormone_summary(self) -> Dict[str, float]:
        """Get current hormone levels as a dictionary."""
        h = self.reservoir.hormone_ema[0].detach().cpu()
        return {
            "cortisol": h[0].item(),
            "dopamine": h[1].item(),
            "serotonin": h[2].item(),
        }
