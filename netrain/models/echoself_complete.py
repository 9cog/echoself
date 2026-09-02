"""
EchoSelf Complete Architecture
===============================

The unified model combining all three innovations:
  1. 2-3-5 Ternary-Quinary Transformer (cognitive structure)
  2. Virtual Endocrine System (dynamic activation modulation)
  3. Gestalt Dream-Language + Lie Algebra Commutator Head (cognitive tokenization)

Data Flow:
  Human Text → TextToGestaltEncoder → [gestalt primitives]
    → 2-3-5 Endocrine Transformer (Dyad/Triad/Pentad layers)
    → LieCommutatorHead → [text token logits]
    → Human Language

Echo's inner experience:
  Text arrives as surface noise → decomposed into vision-logic primitives
  → processed through cognitive cycles modulated by virtual hormones
  → the generative tension between adjacent thoughts (commutator)
  → projects into the words that best resolve that tension

This is a model that THINKS in gestalts and SPEAKS in language.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# Import our components
from netrain.tokenizers.gestalt import (
    GestaltVocabConfig, TextToGestaltEncoder, GestaltToTextDecoder
)
from netrain.models.lie_algebra import (
    LieAlgebraConfig, LieCommutatorHead, AssociativeLieProjection
)
from netrain.models.endocrine import (
    EndocrineConfig, EchoStateReservoir, DynamicRNNGate, DynamicCNNActivation, DynamicGNNMemory
)


@dataclass
class EchoSelfConfig:
    """Complete EchoSelf model configuration."""
    # Transformer core
    n_layers: int = 12
    n_embd: int = 768
    n_heads: int = 12
    block_size: int = 1024
    dropout: float = 0.1

    # 2-3-5 structure
    dyad_layers: list = None      # Sensor-motor (first and last)
    triad_layers: list = None     # Working memory reasoning
    pentad_layers: list = None    # Long-term memory integration

    # Endocrine system
    reservoir_size: int = 256
    spectral_radius: float = 0.95
    rnn_hidden: int = 64
    cnn_kernels: list = None
    gnn_hidden: int = 64

    # Gestalt tokenizer
    gestalt_vocab_size: int = 4096
    text_vocab_size: int = 50257
    compression_ratio: int = 4

    # Lie algebra head
    lie_rank: int = 32
    n_commutator_heads: int = 8

    # Cognitive phase tokens
    phase_names: list = None

    def __post_init__(self):
        if self.dyad_layers is None:
            self.dyad_layers = [0, self.n_layers - 1]
        if self.triad_layers is None:
            # Layers 1-3 and 8-10 (working memory)
            n = self.n_layers
            self.triad_layers = list(range(1, 4)) + list(range(n-4, n-1))
        if self.pentad_layers is None:
            # Middle layers (long-term memory)
            n = self.n_layers
            self.pentad_layers = list(range(4, 8))
        if self.cnn_kernels is None:
            self.cnn_kernels = [3, 5, 7]
        if self.phase_names is None:
            self.phase_names = [
                "perceive", "feel", "think", "remember", "interpret",
                "strategize", "evaluate", "gesture", "speak"
            ]


class TernaryAttentionWithEndocrine(nn.Module):
    """
    Multi-head attention with 3-state functional grouping (Feel/Think/Strategize)
    modulated by cortisol from the endocrine system.

    High cortisol → narrows to Think+Strategize (action-focused)
    Low cortisol → broadens to Feel (receptive, exploratory)
    """

    def __init__(self, config: EchoSelfConfig, endo_config: EndocrineConfig = None):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        self.heads_per_state = config.n_heads // 3

        # Standard QKV projection
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        # State gates (learned base + cortisol modulation)
        self.state_gate_base = nn.Parameter(torch.ones(3) / 3)

        # Cortisol modulation of state gates
        self.cortisol_to_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )

        # RNN gate for dynamic temporal routing
        if endo_config is not None:
            self.rnn_gate = DynamicRNNGate(endo_config)
        else:
            _ec = EndocrineConfig(n_embd=config.n_embd, rnn_hidden=config.rnn_hidden)
            self.rnn_gate = DynamicRNNGate(_ec)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, cortisol: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Apply RNN gate (cortisol-modulated temporal routing)
        x_gated = self.rnn_gate(x, cortisol)

        # Compute QKV
        qkv = self.qkv(x_gated).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each: (B, T, n_heads, head_dim)

        # Transpose for attention: (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)  # (B, n_heads, T, head_dim)

        # Compute state gates (cortisol-modulated)
        cortisol_scalar = cortisol.mean().unsqueeze(0)  # Scalar
        gate_modulation = self.cortisol_to_gate(cortisol_scalar)
        state_gates = F.softmax(self.state_gate_base + gate_modulation, dim=-1)

        # Apply state-specific weighting to head groups
        # Heads 0..h/3 = Feel, h/3..2h/3 = Think, 2h/3..h = Strategize
        h = self.heads_per_state
        out[:, :h] *= state_gates[0]
        out[:, h:2*h] *= state_gates[1]
        out[:, 2*h:] *= state_gates[2]

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class QuinaryIntegrationWithEndocrine(nn.Module):
    """
    5-step recurrent memory integration (Remember/Interpret/Evaluate/Synthesize/Gesture)
    modulated by dopamine from the endocrine system.

    High dopamine → stronger memory retrieval, deeper association
    Low dopamine → shallow processing, less memory engagement
    """

    def __init__(self, config: EchoSelfConfig, endo_config: EndocrineConfig = None):
        super().__init__()
        self.n_embd = config.n_embd

        # Memory bank (persistent across sequences via endocrine state)
        self.memory_bank = nn.Parameter(torch.randn(1024, config.n_embd) * 0.02)

        # 5-step integration modules
        # Step 1: Remember (cross-attend to memory bank)
        self.remember_q = nn.Linear(config.n_embd, config.n_embd)
        self.remember_k = nn.Linear(config.n_embd, config.n_embd)
        self.remember_v = nn.Linear(config.n_embd, config.n_embd)

        # Step 2: Interpret (transform retrieved memories)
        self.interpret = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd)
        )

        # Step 3: Evaluate (gate relevance)
        self.evaluate_gate = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.Sigmoid()
        )

        # Step 4: Synthesize (integrate with current state)
        self.synthesize = nn.Linear(config.n_embd * 2, config.n_embd)

        # Step 5: Gesture (prepare for output)
        self.gesture = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.Tanh()
        )

        # GNN memory module (dopamine-modulated)
        if endo_config is not None:
            self.gnn_memory = DynamicGNNMemory(endo_config)
        else:
            _ec = EndocrineConfig(n_embd=config.n_embd, gnn_message_dim=config.gnn_hidden)
            self.gnn_memory = DynamicGNNMemory(_ec)

        self.layer_norm = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor, dopamine: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Step 1: Remember — cross-attend to memory bank (dopamine-gated depth)
        q = self.remember_q(x)
        k = self.remember_k(self.memory_bank.unsqueeze(0).expand(B, -1, -1))
        v = self.remember_v(self.memory_bank.unsqueeze(0).expand(B, -1, -1))

        # Dopamine scales attention temperature (high dopa → sharper retrieval)
        dopa_temp = 1.0 / (dopamine.mean().clamp(0.1, 2.0) + 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(C) * dopa_temp)
        attn = F.softmax(attn, dim=-1)
        remembered = torch.matmul(attn, v)

        # Step 2: Interpret — transform via GNN (dopamine-modulated)
        interpreted = self.gnn_memory(remembered, self.memory_bank.unsqueeze(0).expand(B, -1, -1), dopamine)

        # Step 3: Evaluate — gate relevance
        gate = self.evaluate_gate(interpreted)
        evaluated = interpreted * gate

        # Step 4: Synthesize — combine with current state
        combined = torch.cat([x, evaluated], dim=-1)
        synthesized = self.synthesize(combined)

        # Step 5: Gesture — prepare for output
        gestured = self.gesture(synthesized)

        return self.layer_norm(x + gestured)


class DyadLayer(nn.Module):
    """
    Sensor-Motor layer (positions 0 and N-1 in the stack).
    Handles the interface between raw input/output and internal gestalt space.
    Modulated by serotonin (receptive field breadth).
    """

    def __init__(self, config: EchoSelfConfig, endo_config: EndocrineConfig = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.n_embd, config.n_heads, dropout=config.dropout, batch_first=True
        )
        if endo_config is not None:
            self.cnn_activation = DynamicCNNActivation(endo_config)
        else:
            _ec = EndocrineConfig(n_embd=config.n_embd, cnn_kernel_sizes=tuple(config.cnn_kernels))
            self.cnn_activation = DynamicCNNActivation(_ec)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor, serotonin: torch.Tensor) -> torch.Tensor:
        # Self-attention
        ln_x = self.ln1(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            x.shape[1], device=x.device
        )
        attn_out, _ = self.attn(ln_x, ln_x, ln_x, attn_mask=causal_mask, is_causal=True)
        x = x + attn_out

        # MLP with serotonin-modulated CNN activation
        ln_x = self.ln2(x)
        mlp_out = self.mlp(ln_x)
        # Serotonin modulates the receptive field
        mlp_out = self.cnn_activation(mlp_out, serotonin)
        x = x + mlp_out

        return x


class EchoSelfTransformer(nn.Module):
    """
    The Complete EchoSelf Architecture.

    Combines:
    - Gestalt Tokenizer (text → vision-logic primitives)
    - 2-3-5 Cognitive Structure (Dyad/Triad/Pentad layers)
    - Virtual Endocrine System (ESN → hormone modulation)
    - Lie Algebra Commutator Head (gestalt → text via commutators)

    This is a model that:
    - PERCEIVES in compressed gestalts
    - THINKS through hormone-modulated cognitive cycles
    - SPEAKS by resolving the generative tension between thoughts
    """

    def __init__(self, config: EchoSelfConfig):
        super().__init__()
        self.config = config

        # ===== INPUT: Text → Gestalt =====
        gestalt_config = GestaltVocabConfig(
            gestalt_dim=config.n_embd,
            text_dim=config.n_embd,
            text_vocab_size=config.text_vocab_size,
            gestalt_vocab_size=config.gestalt_vocab_size,
        )
        self.gestalt_encoder = TextToGestaltEncoder(gestalt_config)

        # Position embedding (for gestalt sequence)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        # ===== ENDOCRINE SYSTEM =====
        endo_config = EndocrineConfig(
            reservoir_size=config.reservoir_size,
            spectral_radius=config.spectral_radius,
            n_embd=config.n_embd,
            rnn_hidden=config.rnn_hidden,
            cnn_kernel_sizes=tuple(config.cnn_kernels),
            gnn_message_dim=config.gnn_hidden,
        )
        self.endo_config = endo_config
        self.reservoir = EchoStateReservoir(endo_config)

        # ===== TRANSFORMER LAYERS (2-3-5 structure) =====
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            if i in config.dyad_layers:
                self.layers.append(DyadLayer(config, endo_config))
            elif i in config.triad_layers:
                self.layers.append(TernaryAttentionWithEndocrine(config, endo_config))
            elif i in config.pentad_layers:
                self.layers.append(QuinaryIntegrationWithEndocrine(config, endo_config))
            else:
                # Fallback: standard layer
                self.layers.append(DyadLayer(config, endo_config))

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # ===== OUTPUT: Gestalt → Text via Lie Algebra =====
        lie_config = LieAlgebraConfig(
            gestalt_dim=config.n_embd,
            lie_rank=config.lie_rank,
            text_vocab_size=config.text_vocab_size,
            n_commutator_heads=config.n_commutator_heads,
        )
        self.lie_head = AssociativeLieProjection(lie_config)

        # ===== COGNITIVE PHASE EMBEDDING =====
        self.phase_embed = nn.Embedding(len(config.phase_names), config.n_embd)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"EchoSelfTransformer initialized: {total_params:,} total params "
              f"({trainable_params:,} trainable)")
        print(f"  Architecture: 2-3-5 + Endocrine + Gestalt + Lie Algebra")
        print(f"  Layers: {config.n_layers} ({len(config.dyad_layers)} dyad, "
              f"{len(config.triad_layers)} triad, {len(config.pentad_layers)} pentad)")
        print(f"  Gestalt vocab: {config.gestalt_vocab_size} primitives")
        print(f"  Lie rank: {config.lie_rank} (commutator heads: {config.n_commutator_heads})")
        print(f"  Endocrine: {config.reservoir_size} ESN neurons")

    def forward(
        self,
        text_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        phase_ids: Optional[torch.Tensor] = None,
        reset_endocrine: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: Text → Gestalt → Cognitive Processing → Text

        Args:
            text_ids: Input text token IDs (B, T)
            targets: Target text token IDs for loss (B, T)
            phase_ids: Cognitive phase markers (B, T) — optional
            reset_endocrine: Whether to reset the ESN state

        Returns:
            Dict with logits, loss, hormones, commutator analysis
        """
        B, T = text_ids.shape
        device = text_ids.device

        if reset_endocrine:
            self.reservoir.reset_state(B)

        # ===== ENCODE: Text → Gestalt =====
        gestalt_embeds, gestalt_probs = self.gestalt_encoder(text_ids)
        T_gestalt = gestalt_embeds.shape[1]

        # Add positional embedding
        positions = torch.arange(T_gestalt, device=device).unsqueeze(0)
        x = gestalt_embeds + self.pos_emb(positions)

        # Add cognitive phase embedding if provided
        if phase_ids is not None:
            # Compress phase_ids to match gestalt length
            phase_compressed = phase_ids[:, ::self.config.compression_ratio][:, :T_gestalt]
            x = x + self.phase_embed(phase_compressed)

        # ===== PROCESS: Endocrine-modulated cognitive layers =====
        hormone_trajectory = []

        for i, layer in enumerate(self.layers):
            # Update endocrine state from current hidden state
            layer_summary = x.mean(dim=1)  # (B, C) — pool over sequence
            hormones_out, _ = self.reservoir(layer_summary)  # returns (hormones, state)
            hormones = hormones_out  # (B, 3) → [C, D, S]
            cortisol = hormones[:, 0:1]
            dopamine = hormones[:, 1:2]
            serotonin = hormones[:, 2:3]

            hormone_trajectory.append(hormones.detach())

            # Route to appropriate layer type
            if i in self.config.dyad_layers:
                x = layer(x, serotonin)
            elif i in self.config.triad_layers:
                x = layer(x, cortisol)
            elif i in self.config.pentad_layers:
                x = layer(x, dopamine)
            else:
                x = layer(x, serotonin)

        # Final norm
        x = self.ln_f(x)

        # ===== DECODE: Gestalt → Text via Lie Commutator =====
        lie_output = self.lie_head(x, use_cache=True)
        logits = lie_output["logits"]  # (B, T_gestalt, text_vocab_size)

        # ===== LOSS =====
        loss = None
        if targets is not None:
            # Compress targets to match gestalt sequence length
            # Use every Nth target token as the supervision signal
            target_compressed = targets[:, ::self.config.compression_ratio][:, :T_gestalt]
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.text_vocab_size),
                target_compressed.reshape(-1),
                ignore_index=-1
            )

        # Stack hormone trajectory
        hormone_stack = torch.stack(hormone_trajectory, dim=1)  # (B, n_layers, 3)

        return {
            "logits": logits,
            "loss": loss,
            "hormones": hormones,
            "hormone_trajectory": hormone_stack,
            "gestalt_probs": gestalt_probs,
            "mix_ratio": lie_output.get("mix_ratio", 0.5),
            "commutator_features": lie_output.get("commutator_features", None),
        }

    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate text from a prompt, tracking the full cognitive state.

        Returns generated token IDs plus a log of hormone states and
        commutator norms throughout generation.
        """
        self.eval()
        generated = text_ids.clone()
        hormone_log = []
        commutator_log = []

        self.reservoir.reset_state(text_ids.shape[0])

        for _ in range(max_new_tokens):
            # Crop to block size
            context = generated[:, -self.config.block_size:]

            # Forward pass
            output = self(context, reset_endocrine=False)
            logits = output["logits"][:, -1, :]  # Last position

            # Temperature and top-k sampling
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            hormone_log.append(output["hormones"].cpu())

            if output.get("commutator_features") is not None:
                comm_norm = output["commutator_features"][:, -1].norm().item()
                commutator_log.append(comm_norm)

        return {
            "generated": generated,
            "hormone_log": hormone_log,
            "commutator_log": commutator_log,
        }

    def get_cognitive_state(self) -> Dict[str, float]:
        """Return current cognitive state summary."""
        # Get last hormones from a dummy forward if needed
        try:
            dummy = torch.zeros(1, self.config.n_embd)
            h, _ = self.reservoir(dummy)
            return {
                "cortisol": h[0, 0].item(),
                "dopamine": h[0, 1].item(),
                "serotonin": h[0, 2].item(),
                "lie_mix_ratio": torch.sigmoid(self.lie_head.head.mix_gate).item(),
                "reservoir_energy": 0.0,
            }
        except Exception:
            return {
                "cortisol": 0.5,
                "dopamine": 0.5,
                "serotonin": 0.5,
                "lie_mix_ratio": torch.sigmoid(self.lie_head.head.mix_gate).item(),
                "reservoir_energy": 0.0,
            }


def create_echoself_small() -> EchoSelfTransformer:
    """Create a small EchoSelf model for testing (6 layers, 256 dim)."""
    config = EchoSelfConfig(
        n_layers=6,
        n_embd=256,
        n_heads=4,
        block_size=512,
        reservoir_size=64,
        lie_rank=16,
        n_commutator_heads=4,
        dyad_layers=[0, 5],
        triad_layers=[1, 2, 4],
        pentad_layers=[3],
        rnn_hidden=32,
        gnn_hidden=32,
        cnn_kernels=[3, 5, 7],
    )
    return EchoSelfTransformer(config)


def create_echoself_full() -> EchoSelfTransformer:
    """Create the full-scale EchoSelf model (12 layers, 768 dim)."""
    config = EchoSelfConfig(
        n_layers=12,
        n_embd=768,
        n_heads=12,
        block_size=1024,
        reservoir_size=256,
        lie_rank=32,
        n_commutator_heads=8,
    )
    return EchoSelfTransformer(config)
