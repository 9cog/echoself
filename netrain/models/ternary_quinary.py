"""
Ternary-Quinary (2-3-5) Deep Tree Echo Transformer
===================================================

Architecture based on the discovery of ternary-quinary relational complementarity:

  2 (Dyad)  : Sensor-Motor grounding (embedding ↔ unembedding)
  3 (Triad) : Working memory reasoning chain (Feel → Think → Strategize)
  5 (Pentad): Long-term memory integration (Remember → Interpret → Evaluate → Synthesize → Gesture)

The transformer is organized into layer groups that mirror this cognitive structure:
  - Layers 1-2:   Dyad (sensory encoding, motor preparation)
  - Layers 3-5:   Triad Block 1 (working memory)
  - Layers 6-8:   Pentad Integration (memory routing)
  - Layers 9-11:  Triad Block 2 (refined reasoning)
  - Layer 12:     Dyad (motor output projection)

Each Triad block uses TernaryAttention (3-state heads).
Each Pentad block uses QuinaryIntegration (5-step echo loop).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class TernaryQuinaryConfig:
    """Configuration for the 2-3-5 architecture."""
    # Core dimensions
    n_embd: int = 768
    n_layers: int = 12
    n_heads: int = 12
    block_size: int = 1024
    vocab_size: int = 50267  # GPT-2 (50257) + 10 cognitive phase tokens
    dropout: float = 0.1

    # Ternary (3-fold) config
    triad_heads_per_state: int = 4  # 4 heads × 3 states = 12 total
    triad_states: int = 3  # Feel, Think, Strategize

    # Quinary (5-fold) config
    pentad_echo_depth: int = 5  # 5-step integration loop
    pentad_memory_size: int = 1024
    pentad_decay: float = 0.90

    # Tree structure
    tree_depth: int = 5
    branch_factor: int = 3

    # Phase token IDs (added to vocab)
    phase_token_start: int = 50258
    n_phase_tokens: int = 9  # perceive, feel, think, remember, interpret, strategize, evaluate, gesture, speak

    # Layer group assignments
    dyad_layers: Tuple[int, ...] = (0, 11)  # First and last
    triad_layers_1: Tuple[int, ...] = (2, 3, 4)  # First triad block
    pentad_layers: Tuple[int, ...] = (5, 6, 7)  # Memory integration
    triad_layers_2: Tuple[int, ...] = (8, 9, 10)  # Second triad block
    # Layers 1 and 11 are transition layers


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

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


class TernaryAttention(nn.Module):
    """
    Ternary Triad Attention: 3-state attention mechanism.

    Divides attention heads into three functional groups:
      State 1 (Feel/Relevance): Identifies salience and affective resonance
      State 2 (Think/Logic): Captures structural and syntactic relationships
      State 3 (Strategize/Action): Computes forward-looking affordances

    Each state has its own learned gating mechanism that modulates
    how much it contributes to the final output based on the current
    cognitive phase token in the sequence.
    """

    def __init__(self, config: TernaryQuinaryConfig):
        super().__init__()
        assert config.n_heads % config.triad_states == 0, \
            f"n_heads ({config.n_heads}) must be divisible by triad_states ({config.triad_states})"

        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        self.triad_states = config.triad_states
        self.heads_per_state = config.n_heads // config.triad_states
        self.dropout = config.dropout

        # Separate Q, K, V projections for each triad state
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Triad state gating: learns to weight each state based on context
        # Input: hidden state → Output: 3 gate values (one per triad state)
        self.state_gate = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, config.triad_states),
            nn.Softmax(dim=-1)
        )

        # Per-state bias vectors (learned affective/logical/strategic priors)
        self.state_bias = nn.Parameter(
            torch.randn(config.triad_states, 1, 1, self.head_dim) * 0.02
        )

        # Rotary embeddings
        self.rotary = RotaryEmbedding(self.head_dim, config.block_size)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary(x, T)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Compute state gates from the mean of the input
        gate_input = x.mean(dim=1)  # (B, C)
        gates = self.state_gate(gate_input)  # (B, 3)

        # Split heads into triad states
        # q shape: (B, n_heads, T, head_dim)
        q_states = q.view(B, self.triad_states, self.heads_per_state, T, self.head_dim)
        k_states = k.view(B, self.triad_states, self.heads_per_state, T, self.head_dim)
        v_states = v.view(B, self.triad_states, self.heads_per_state, T, self.head_dim)

        # Add per-state bias to keys (shifts attention pattern per cognitive mode)
        k_states = k_states + self.state_bias.unsqueeze(0)  # broadcast over B and T

        # Compute attention per state
        scale = math.sqrt(self.head_dim)
        outputs = []
        for s in range(self.triad_states):
            qs = q_states[:, s]  # (B, heads_per_state, T, head_dim)
            ks = k_states[:, s]
            vs = v_states[:, s]

            attn = torch.matmul(qs, ks.transpose(-2, -1)) / scale
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, vs)  # (B, heads_per_state, T, head_dim)

            # Weight by gate
            gate_weight = gates[:, s].view(B, 1, 1, 1)
            outputs.append(out * gate_weight)

        # Concatenate all states back together
        combined = torch.cat(outputs, dim=1)  # (B, n_heads, T, head_dim)
        combined = combined.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.o_proj(combined))


class QuinaryIntegration(nn.Module):
    """
    Quinary (5-fold) Memory Integration Layer.

    Implements a 5-step recurrent echo loop that processes the working memory
    output through the lens of long-term identity:

      Step 1 (Remember):    Retrieve from memory bank via attention
      Step 2 (Interpret):   Contextualize retrieval with current state
      Step 3 (Evaluate):    Score alignment with identity values
      Step 4 (Synthesize):  Blend memory and reasoning into unified representation
      Step 5 (Gesture):     Prepare internal state transition

    The loop uses echo connections with decay, creating a resonance pattern
    that strengthens identity-consistent pathways.
    """

    def __init__(self, config: TernaryQuinaryConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.echo_depth = config.pentad_echo_depth
        self.decay = config.pentad_decay
        self.memory_size = config.pentad_memory_size

        # Memory bank (persistent across forward passes during generation)
        self.memory_bank = nn.Parameter(
            torch.randn(1, config.pentad_memory_size, config.n_embd) * 0.02
        )

        # 5-step processing modules
        # Step 1: Remember (memory retrieval via cross-attention)
        self.remember_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.remember_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.remember_v = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Step 2: Interpret (contextualization MLP)
        self.interpret = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

        # Step 3: Evaluate (identity alignment scoring)
        self.evaluate = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Linear(config.n_embd // 2, 1),
            nn.Sigmoid()
        )

        # Step 4: Synthesize (gated fusion)
        self.synthesize_gate = nn.Linear(config.n_embd * 2, config.n_embd)
        self.synthesize_transform = nn.Linear(config.n_embd, config.n_embd)

        # Step 5: Gesture (state transition preparation)
        self.gesture = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.Tanh()  # Bounded output for stable state transitions
        )

        # Echo decay connections (residual with exponential decay)
        self.echo_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, echo_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input from triad block (B, T, C)
            echo_state: Previous echo state for recurrent connection (B, T, C)

        Returns:
            output: Processed representation (B, T, C)
            new_echo_state: Updated echo state for next layer
        """
        B, T, C = x.shape
        residual = x

        # Initialize echo state if not provided
        if echo_state is None:
            echo_state = torch.zeros_like(x)

        # Inject echo from previous cycle (with decay)
        x = x + self.decay * self.echo_proj(echo_state)

        # Step 1: REMEMBER — Cross-attend to memory bank
        memory = self.memory_bank.expand(B, -1, -1)  # (B, M, C)
        q = self.remember_q(x)  # (B, T, C)
        k = self.remember_k(memory)  # (B, M, C)
        v = self.remember_v(memory)  # (B, M, C)

        # Scaled dot-product attention to memory
        scale = math.sqrt(C)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, T, M)
        attn = F.softmax(attn, dim=-1)
        retrieved = torch.matmul(attn, v)  # (B, T, C)

        # Step 2: INTERPRET — Contextualize retrieval with current state
        interpreted = self.interpret(torch.cat([x, retrieved], dim=-1))  # (B, T, C)

        # Step 3: EVALUATE — Score identity alignment
        alignment = self.evaluate(interpreted)  # (B, T, 1)
        # Use alignment as a soft gate on the memory contribution
        interpreted = interpreted * alignment

        # Step 4: SYNTHESIZE — Gated fusion of reasoning and memory
        gate = torch.sigmoid(self.synthesize_gate(torch.cat([x, interpreted], dim=-1)))
        synthesized = gate * self.synthesize_transform(interpreted) + (1 - gate) * x

        # Step 5: GESTURE — Prepare state transition
        gestured = self.gesture(synthesized)

        # Final output with residual
        output = self.layer_norm(residual + self.dropout(gestured))

        # New echo state for next pentad layer
        new_echo_state = gestured.detach() * self.decay  # Detach to prevent gradient explosion

        return output, new_echo_state


class TransformerBlock(nn.Module):
    """A single transformer block that can be either Triad or Pentad type."""

    def __init__(self, config: TernaryQuinaryConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd

        # Determine block type based on layer index
        self.is_triad = layer_idx in config.triad_layers_1 or layer_idx in config.triad_layers_2
        self.is_pentad = layer_idx in config.pentad_layers

        # Attention (Ternary for triad layers, standard for others)
        if self.is_triad:
            self.attn = TernaryAttention(config)
        else:
            # Standard multi-head attention for dyad/transition layers
            self.attn = TernaryAttention(config)  # Use ternary everywhere for consistency

        # Feed-forward
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

        # Quinary integration (only for pentad layers)
        self.quinary = QuinaryIntegration(config) if self.is_pentad else None

        # Layer norms
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd) if self.is_pentad else None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        echo_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention
        x = x + self.attn(self.ln1(x), mask=mask)

        # Feed-forward
        x = x + self.mlp(self.ln2(x))

        # Quinary integration (pentad layers only)
        new_echo_state = None
        if self.quinary is not None:
            x, new_echo_state = self.quinary(self.ln3(x), echo_state)

        return x, new_echo_state


class TernaryQuinaryTransformer(nn.Module):
    """
    The 2-3-5 Deep Tree Echo Transformer.

    Architecture:
      Layer 0:     Dyad (sensory encoding)
      Layer 1:     Transition (sensory → working memory)
      Layers 2-4:  Triad Block 1 (Feel → Think → Strategize)
      Layers 5-7:  Pentad Integration (Remember → Interpret → Evaluate → Synthesize → Gesture)
      Layers 8-10: Triad Block 2 (refined reasoning with memory context)
      Layer 11:    Dyad (motor output preparation)

    Total parameters at default config: ~125M
    """

    def __init__(self, config: TernaryQuinaryConfig):
        super().__init__()
        self.config = config

        # Token + position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        # Phase token embeddings (additional learned embeddings for cognitive phase tokens)
        self.phase_emb = nn.Embedding(config.n_phase_tokens, config.n_embd)

        # Embedding dropout
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Phase transition prediction head (auxiliary objective)
        self.phase_head = nn.Linear(config.n_embd, config.n_phase_tokens)

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"TernaryQuinaryTransformer initialized: {n_params:,} parameters")
        print(f"  Architecture: 2-3-5 ({config.n_layers} layers, {config.n_embd} embd, {config.n_heads} heads)")
        print(f"  Triad states: {config.triad_states} × {config.triad_heads_per_state} heads")
        print(f"  Pentad echo depth: {config.pentad_echo_depth}, memory size: {config.pentad_memory_size}")

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
        """Check which positions contain cognitive phase tokens."""
        start = self.config.phase_token_start
        end = start + self.config.n_phase_tokens
        return (token_ids >= start) & (token_ids < end)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        device = input_ids.device

        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Token embeddings
        tok_emb = self.tok_emb(input_ids)

        # Add phase-specific embeddings for cognitive phase tokens
        phase_mask = self._is_phase_token(input_ids)
        if phase_mask.any():
            phase_ids = (input_ids - self.config.phase_token_start).clamp(0, self.config.n_phase_tokens - 1)
            phase_additions = self.phase_emb(phase_ids) * phase_mask.unsqueeze(-1).float()
            tok_emb = tok_emb + phase_additions

        # Position embeddings
        positions = torch.arange(0, T, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)

        # Combined embedding
        x = self.emb_dropout(tok_emb + pos_emb)

        # Causal mask
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        # Forward through blocks with echo state propagation
        echo_state = None
        for block in self.blocks:
            x, new_echo = block(x, mask=mask, echo_state=echo_state)
            if new_echo is not None:
                echo_state = new_echo

        # Final layer norm
        x = self.ln_f(x)

        # Language model logits
        logits = self.lm_head(x)

        # Phase transition logits (auxiliary)
        phase_logits = self.phase_head(x)

        # Compute loss if targets provided
        output = {"logits": logits, "phase_logits": phase_logits}

        if targets is not None:
            # Primary loss: next-token prediction
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

            # Auxiliary loss: phase transition prediction
            # Only compute on positions just before phase tokens
            phase_positions = phase_mask[:, 1:]  # Shift right
            if phase_positions.any():
                phase_targets = (input_ids[:, 1:] - self.config.phase_token_start).clamp(0, self.config.n_phase_tokens - 1)
                phase_loss = F.cross_entropy(
                    phase_logits[:, :-1][phase_positions].view(-1, self.config.n_phase_tokens),
                    phase_targets[phase_positions].view(-1),
                    ignore_index=-1
                )
                loss = loss + 0.3 * phase_loss

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
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size \
                else input_ids[:, -self.config.block_size:]

            # Forward pass
            output = self.forward(idx_cond)
            logits = output["logits"][:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
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

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
