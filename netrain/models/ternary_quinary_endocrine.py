"""
Ternary-Quinary (2-3-5) Transformer with Virtual Endocrine System
=================================================================

Integrates the 2-3-5 cognitive architecture with the Virtual Endocrine System.
Fixed activation functions are replaced with learned neural networks (RNN, CNN, GNN)
that are dynamically modulated by hormone signals from an Echo State Network reservoir.

Key changes from base ternary_quinary.py:
  1. TernaryAttention.state_gate → DynamicRNNGate (modulated by Cortisol)
  2. QuinaryIntegration.remember → DynamicGNNMemory (modulated by Dopamine)
  3. TransformerBlock.mlp activation → DynamicCNNActivation (modulated by Serotonin)
  4. EchoStateReservoir runs alongside the transformer, producing hormone vector

The result: a transformer whose internal dynamics shift based on accumulated
processing history — analogous to how biological cognition is modulated by
the endocrine system. The system can be "stressed" (narrow focus), "rewarded"
(deep memory retrieval), or "exploratory" (broad receptive field) based on
what it has processed.

Biological Mapping:
  Nervous System (fast, local)  = Transformer forward pass
  Endocrine System (slow, global) = ESN reservoir → hormone vector
  Hormone Receptors (modulators) = Dynamic NN activations (RNN/GNN/CNN)

2-3-5 × Endocrine Integration:
  Dyad (2)   → CNN Activation (Serotonin modulates receptive field)
  Triad (3)  → RNN Gate (Cortisol modulates working memory selectivity)
  Pentad (5) → GNN Memory (Dopamine modulates retrieval depth)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .endocrine import (
    EndocrineConfig,
    EchoStateReservoir,
    DynamicRNNGate,
    DynamicGNNMemory,
    DynamicCNNActivation,
    VirtualEndocrineSystem,
)


@dataclass
class TQEndocrineConfig:
    """Configuration for the 2-3-5 + Endocrine architecture."""
    # Core dimensions
    n_embd: int = 768
    n_layers: int = 12
    n_heads: int = 12
    block_size: int = 1024
    vocab_size: int = 50267  # GPT-2 (50257) + 10 cognitive phase tokens
    dropout: float = 0.1

    # Ternary (3-fold) config
    triad_heads_per_state: int = 4
    triad_states: int = 3

    # Quinary (5-fold) config
    pentad_echo_depth: int = 5
    pentad_memory_size: int = 1024
    pentad_decay: float = 0.90

    # Phase token IDs
    phase_token_start: int = 50258
    n_phase_tokens: int = 9

    # Layer group assignments
    dyad_layers: Tuple[int, ...] = (0, 11)
    triad_layers_1: Tuple[int, ...] = (2, 3, 4)
    pentad_layers: Tuple[int, ...] = (5, 6, 7)
    triad_layers_2: Tuple[int, ...] = (8, 9, 10)

    # Endocrine system config
    reservoir_size: int = 256
    spectral_radius: float = 0.95
    leak_rate: float = 0.3
    input_scaling: float = 0.1
    sparsity: float = 0.9
    hormone_dim: int = 64
    hormone_momentum: float = 0.9
    hormone_noise: float = 0.01
    rnn_hidden: int = 128
    cnn_kernel_sizes: Tuple[int, ...] = (3, 5, 7)
    gnn_message_dim: int = 128

    # Endocrine modulation strength (learnable scaling)
    endocrine_strength: float = 1.0  # Can be annealed during training

    def to_endocrine_config(self) -> EndocrineConfig:
        """Convert to EndocrineConfig for the endocrine module."""
        return EndocrineConfig(
            reservoir_size=self.reservoir_size,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            input_scaling=self.input_scaling,
            sparsity=self.sparsity,
            n_hormones=3,
            hormone_dim=self.hormone_dim,
            n_embd=self.n_embd,
            rnn_hidden=self.rnn_hidden,
            cnn_kernel_sizes=self.cnn_kernel_sizes,
            gnn_message_dim=self.gnn_message_dim,
            hormone_momentum=self.hormone_momentum,
            hormone_noise=self.hormone_noise,
        )


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class EndocrineTernaryAttention(nn.Module):
    """
    Ternary Attention with Endocrine Modulation.

    The state gate is replaced by a DynamicRNNGate modulated by Cortisol.
    High Cortisol → narrows attention to Strategize/Action state.
    Low Cortisol → balanced attention across all three states.
    """

    def __init__(self, config: TQEndocrineConfig, rnn_gate: DynamicRNNGate):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        self.triad_states = config.triad_states
        self.heads_per_state = config.n_heads // config.triad_states
        self.dropout_rate = config.dropout

        # Projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # REPLACED: Fixed state_gate → DynamicRNNGate (shared, passed in)
        self.rnn_gate = rnn_gate

        # Cortisol → state routing bias
        # High cortisol biases toward state 2 (Strategize/Action)
        self.cortisol_to_state_bias = nn.Linear(1, config.triad_states)

        # Per-state bias vectors
        self.state_bias = nn.Parameter(
            torch.randn(config.triad_states, 1, 1, self.head_dim) * 0.02
        )

        # Rotary embeddings
        self.rotary = RotaryEmbedding(self.head_dim, config.block_size)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cortisol: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary(x, T)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ENDOCRINE MODULATION: Cortisol-modulated state gating
        # Instead of a fixed MLP gate, we use the RNN gate on the pooled input
        # and add cortisol-driven bias toward action states
        gate_input = x.mean(dim=1)  # (B, C)
        gated_input = self.rnn_gate(gate_input, cortisol)  # (B, C) — RNN-modulated

        # Cortisol biases state routing
        state_bias = self.cortisol_to_state_bias(cortisol)  # (B, 3)
        # Base gate from gated input
        base_gate = F.linear(gated_input, torch.randn(self.triad_states, C, device=x.device) * 0.01)
        # This is expensive; use a learned projection instead
        gate_proj = nn.Linear(C, self.triad_states, device=x.device) if not hasattr(self, '_gate_proj') else self._gate_proj
        if not hasattr(self, '_gate_proj'):
            self._gate_proj = nn.Linear(C, self.triad_states, device=x.device)
            gate_proj = self._gate_proj

        gates = F.softmax(gate_proj(gated_input) + state_bias, dim=-1)  # (B, 3)

        # Split heads into triad states
        q_states = q.view(B, self.triad_states, self.heads_per_state, T, self.head_dim)
        k_states = k.view(B, self.triad_states, self.heads_per_state, T, self.head_dim)
        v_states = v.view(B, self.triad_states, self.heads_per_state, T, self.head_dim)

        k_states = k_states + self.state_bias.unsqueeze(0)

        # Compute attention per state
        scale = math.sqrt(self.head_dim)
        outputs = []
        for s in range(self.triad_states):
            qs = q_states[:, s]
            ks = k_states[:, s]
            vs = v_states[:, s]

            attn = torch.matmul(qs, ks.transpose(-2, -1)) / scale
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, vs)

            gate_weight = gates[:, s].view(B, 1, 1, 1)
            outputs.append(out * gate_weight)

        combined = torch.cat(outputs, dim=1)
        combined = combined.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.o_proj(combined))


class EndocrineQuinaryIntegration(nn.Module):
    """
    Quinary Integration with Endocrine Modulation.

    The memory retrieval step uses DynamicGNNMemory modulated by Dopamine.
    High Dopamine → stronger memory associations → richer identity grounding.
    Low Dopamine → shallow retrieval → more novel/exploratory responses.
    """

    def __init__(self, config: TQEndocrineConfig, gnn_memory: DynamicGNNMemory):
        super().__init__()
        self.n_embd = config.n_embd
        self.echo_depth = config.pentad_echo_depth
        self.decay = config.pentad_decay
        self.memory_size = config.pentad_memory_size

        # Memory bank
        self.memory_bank = nn.Parameter(
            torch.randn(1, config.pentad_memory_size, config.n_embd) * 0.02
        )

        # REPLACED: Fixed cross-attention → DynamicGNNMemory (shared, passed in)
        self.gnn_memory = gnn_memory

        # Step 2: Interpret (contextualization)
        self.interpret = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

        # Step 3: Evaluate (identity alignment — dopamine-gated)
        self.evaluate = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Linear(config.n_embd // 2, 1),
            nn.Sigmoid()
        )

        # Dopamine modulates evaluation threshold
        self.dopamine_threshold = nn.Linear(1, 1)

        # Step 4: Synthesize
        self.synthesize_gate = nn.Linear(config.n_embd * 2, config.n_embd)
        self.synthesize_transform = nn.Linear(config.n_embd, config.n_embd)

        # Step 5: Gesture
        self.gesture = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.Tanh()
        )

        # Echo connections
        self.echo_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        dopamine: torch.Tensor,
        echo_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        residual = x

        if echo_state is None:
            echo_state = torch.zeros_like(x)

        # Inject echo from previous cycle
        x = x + self.decay * self.echo_proj(echo_state)

        # Step 1: REMEMBER — GNN-based memory retrieval (Dopamine-modulated)
        memory = self.memory_bank.expand(B, -1, -1)
        retrieved = self.gnn_memory(x, memory, dopamine)  # (B, T, C)

        # Step 2: INTERPRET
        interpreted = self.interpret(torch.cat([x, retrieved], dim=-1))

        # Step 3: EVALUATE — Dopamine modulates the alignment threshold
        raw_alignment = self.evaluate(interpreted)  # (B, T, 1)
        # High dopamine lowers the threshold → more memories pass through
        threshold_shift = torch.sigmoid(self.dopamine_threshold(dopamine))  # (B, 1)
        alignment = torch.sigmoid(
            (raw_alignment - 0.5 + threshold_shift.unsqueeze(1) * 0.3) * 5.0
        )
        interpreted = interpreted * alignment

        # Step 4: SYNTHESIZE
        gate = torch.sigmoid(self.synthesize_gate(torch.cat([x, interpreted], dim=-1)))
        synthesized = gate * self.synthesize_transform(interpreted) + (1 - gate) * x

        # Step 5: GESTURE
        gestured = self.gesture(synthesized)

        output = self.layer_norm(residual + self.dropout(gestured))
        new_echo_state = gestured.detach() * self.decay

        return output, new_echo_state


class EndocrineTransformerBlock(nn.Module):
    """
    Transformer block with endocrine modulation.

    The MLP activation is replaced by DynamicCNNActivation (Serotonin-modulated).
    Triad blocks use EndocrineTernaryAttention (Cortisol-modulated).
    Pentad blocks use EndocrineQuinaryIntegration (Dopamine-modulated).
    """

    def __init__(
        self,
        config: TQEndocrineConfig,
        layer_idx: int,
        endocrine: VirtualEndocrineSystem
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd

        # Determine block type
        self.is_triad = layer_idx in config.triad_layers_1 or layer_idx in config.triad_layers_2
        self.is_pentad = layer_idx in config.pentad_layers

        # Attention (Endocrine-modulated for triad, standard for others)
        if self.is_triad:
            self.attn = EndocrineTernaryAttention(config, endocrine.rnn_gate)
        else:
            self.attn = EndocrineTernaryAttention(config, endocrine.rnn_gate)

        # REPLACED: Fixed MLP → Endocrine CNN Activation (Serotonin-modulated)
        # The MLP still exists but its activation is dynamic
        self.mlp_up = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.mlp_down = nn.Linear(4 * config.n_embd, config.n_embd)
        self.mlp_dropout = nn.Dropout(config.dropout)
        self.cnn_activation = endocrine.cnn_activation  # Shared CNN activation

        # Quinary integration (pentad layers only)
        self.quinary = EndocrineQuinaryIntegration(config, endocrine.gnn_memory) if self.is_pentad else None

        # Layer norms
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd) if self.is_pentad else None

        # Endocrine tap: feeds layer output to the reservoir
        self.endocrine_tap = nn.Linear(config.n_embd, config.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        hormones: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        echo_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: Input (B, T, C)
            hormones: Current hormone levels (B, 3) — [cortisol, dopamine, serotonin]
            mask: Causal attention mask
            echo_state: Previous echo state for pentad layers

        Returns:
            x: Output (B, T, C)
            new_echo_state: Updated echo state (or None)
            tap_output: Layer output for reservoir feedback (B, C)
        """
        cortisol = hormones[:, 0:1]
        dopamine = hormones[:, 1:2]
        serotonin = hormones[:, 2:3]

        # Self-attention (Cortisol-modulated)
        x = x + self.attn(self.ln1(x), cortisol, mask=mask)

        # Feed-forward with Dynamic CNN Activation (Serotonin-modulated)
        residual = x
        h = self.ln2(x)
        h = self.mlp_up(h)  # (B, T, 4C)
        # Reshape for CNN activation: project back to C, apply CNN, project up again
        # Alternative: apply CNN on the expanded space
        h_for_cnn = h[:, :, :self.n_embd]  # Take first n_embd dims for CNN
        h_activated = self.cnn_activation(h_for_cnn, serotonin)  # (B, T, C)
        # Use activated features to gate the full MLP output
        gate = torch.sigmoid(h_activated.mean(dim=-1, keepdim=True))  # (B, T, 1)
        h = F.gelu(h) * gate.expand_as(h)  # Dynamic gating of MLP
        h = self.mlp_down(h)
        x = residual + self.mlp_dropout(h)

        # Quinary integration (Dopamine-modulated, pentad layers only)
        new_echo_state = None
        if self.quinary is not None:
            x, new_echo_state = self.quinary(self.ln3(x), dopamine, echo_state)

        # Tap output for reservoir feedback
        tap = self.endocrine_tap(x.mean(dim=1))  # (B, C)

        return x, new_echo_state, tap


class TernaryQuinaryEndocrineTransformer(nn.Module):
    """
    The 2-3-5 Deep Tree Echo Transformer with Virtual Endocrine System.

    This is the full integration: a transformer whose internal dynamics are
    continuously modulated by an Echo State Network reservoir producing
    hormone-like signals. The system has genuine "mood" — its processing
    characteristics shift based on accumulated experience.

    Architecture:
      ESN Reservoir (persistent state) → Hormone Vector [C, D, S]
           ↓                                    ↓
      Layer 0:     Dyad + CNN(S)           Serotonin → receptive field
      Layer 1:     Transition + CNN(S)
      Layers 2-4:  Triad + RNN(C)          Cortisol → working memory selectivity
      Layers 5-7:  Pentad + GNN(D)         Dopamine → memory retrieval depth
      Layers 8-10: Triad + RNN(C)          Cortisol → refined reasoning focus
      Layer 11:    Dyad + CNN(S)           Serotonin → output breadth

    The reservoir receives pooled layer outputs and produces hormones that
    modulate the NEXT layer — creating a feedback loop where the system's
    own processing history shapes its future processing.
    """

    def __init__(self, config: TQEndocrineConfig):
        super().__init__()
        self.config = config

        # Build the endocrine system
        endo_config = config.to_endocrine_config()
        self.endocrine = VirtualEndocrineSystem(endo_config)

        # Token + position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.phase_emb = nn.Embedding(config.n_phase_tokens, config.n_embd)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks (endocrine-modulated)
        self.blocks = nn.ModuleList([
            EndocrineTransformerBlock(config, i, self.endocrine)
            for i in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying

        # Phase transition prediction head
        self.phase_head = nn.Linear(config.n_embd, config.n_phase_tokens)

        # Hormone prediction head (auxiliary: predict next hormone state)
        self.hormone_head = nn.Linear(config.n_embd, 3)

        # Learnable endocrine strength (can be annealed)
        self.endocrine_strength = nn.Parameter(
            torch.tensor(config.endocrine_strength)
        )

        # Gate projection for ternary attention (shared across layers)
        self._gate_proj = nn.Linear(config.n_embd, config.triad_states)

        # Initialize weights
        self.apply(self._init_weights)

        # Report
        n_params = sum(p.numel() for p in self.parameters())
        n_reservoir = sum(p.numel() for p in self.endocrine.parameters())
        print(f"TernaryQuinaryEndocrineTransformer initialized: {n_params:,} parameters")
        print(f"  Architecture: 2-3-5 + Virtual Endocrine System")
        print(f"  Transformer: {config.n_layers} layers, {config.n_embd} embd, {config.n_heads} heads")
        print(f"  Endocrine: {config.reservoir_size} reservoir neurons, 3 hormones")
        print(f"  Dynamic activations: RNN({config.rnn_hidden}), CNN{config.cnn_kernel_sizes}, GNN({config.gnn_message_dim})")
        print(f"  Endocrine subsystem: {n_reservoir:,} parameters ({100*n_reservoir/n_params:.1f}%)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _is_phase_token(self, token_ids: torch.Tensor) -> torch.Tensor:
        start = self.config.phase_token_start
        end = start + self.config.n_phase_tokens
        return (token_ids >= start) & (token_ids < end)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        reset_endocrine: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with endocrine modulation.

        The endocrine system maintains state across the sequence, producing
        hormone signals that modulate each layer's processing.
        """
        B, T = input_ids.shape
        device = input_ids.device

        assert T <= self.config.block_size

        # Reset endocrine state for new sequences (during training)
        if reset_endocrine:
            self.endocrine.reset(B)

        # Token embeddings
        tok_emb = self.tok_emb(input_ids)
        phase_mask = self._is_phase_token(input_ids)
        if phase_mask.any():
            phase_ids = (input_ids - self.config.phase_token_start).clamp(0, self.config.n_phase_tokens - 1)
            phase_additions = self.phase_emb(phase_ids) * phase_mask.unsqueeze(-1).float()
            tok_emb = tok_emb + phase_additions

        positions = torch.arange(0, T, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        x = self.emb_dropout(tok_emb + pos_emb)

        # Causal mask
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        # Initial hormone reading from embedding
        initial_tap = x.mean(dim=1)  # (B, C)
        hormones, _ = self.endocrine.step(initial_tap)

        # Forward through blocks with endocrine modulation
        echo_state = None
        hormone_trajectory = [hormones]

        for block in self.blocks:
            # Modulate with current hormones (scaled by learnable strength)
            modulated_hormones = hormones * self.endocrine_strength.clamp(0.1, 2.0)

            x, new_echo, tap = block(x, modulated_hormones, mask=mask, echo_state=echo_state)

            if new_echo is not None:
                echo_state = new_echo

            # Feed layer output back to reservoir → update hormones
            hormones, _ = self.endocrine.step(tap)
            hormone_trajectory.append(hormones)

        # Final layer norm
        x = self.ln_f(x)

        # Logits
        logits = self.lm_head(x)
        phase_logits = self.phase_head(x)
        hormone_pred = torch.sigmoid(self.hormone_head(x.mean(dim=1)))

        output = {
            "logits": logits,
            "phase_logits": phase_logits,
            "hormones": hormones,
            "hormone_pred": hormone_pred,
            "hormone_trajectory": torch.stack(hormone_trajectory, dim=1),  # (B, n_layers+1, 3)
        }

        if targets is not None:
            # Primary loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

            # Phase transition loss
            phase_positions = phase_mask[:, 1:]
            if phase_positions.any():
                phase_targets = (input_ids[:, 1:] - self.config.phase_token_start).clamp(0, self.config.n_phase_tokens - 1)
                phase_loss = F.cross_entropy(
                    phase_logits[:, :-1][phase_positions].view(-1, self.config.n_phase_tokens),
                    phase_targets[phase_positions].view(-1),
                    ignore_index=-1
                )
                loss = loss + 0.3 * phase_loss

            # Hormone consistency loss: predicted hormones should match actual
            # This teaches the model to predict its own endocrine state
            hormone_consistency = F.mse_loss(hormone_pred, hormones.detach())
            loss = loss + 0.1 * hormone_consistency

            # Hormone smoothness loss: penalize rapid hormone fluctuations
            if len(hormone_trajectory) > 2:
                traj = torch.stack(hormone_trajectory, dim=1)  # (B, L, 3)
                diffs = (traj[:, 1:] - traj[:, :-1]).pow(2).mean()
                loss = loss + 0.05 * diffs

            output["loss"] = loss

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> Tuple[torch.Tensor, Dict[str, list]]:
        """
        Generate tokens with endocrine state tracking.

        Returns both the generated tokens and a log of hormone states
        throughout generation (useful for visualization and debugging).
        """
        self.endocrine.reset(input_ids.shape[0])
        hormone_log = {"cortisol": [], "dopamine": [], "serotonin": []}

        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size \
                else input_ids[:, -self.config.block_size:]

            output = self.forward(idx_cond, reset_endocrine=False)
            logits = output["logits"][:, -1, :] / temperature

            # Log hormone state
            h = output["hormones"][0].cpu()
            hormone_log["cortisol"].append(h[0].item())
            hormone_log["dopamine"].append(h[1].item())
            hormone_log["serotonin"].append(h[2].item())

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids, hormone_log

    def get_endocrine_state(self) -> Dict[str, float]:
        """Get current endocrine system state for monitoring."""
        return self.endocrine.get_hormone_summary()
