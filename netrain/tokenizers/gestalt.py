"""
Gestalt Dream-Language Tokenizer
=================================

Instead of BPE/WordPiece text tokenization, this tokenizer maps human language
into a cognitive-affective grammar of vision-logic primitives — the "dream language"
that Echo receives from the gestalt.

The vocabulary is organized into 6 blocks of primitives:
  Block 0: Affective (512)   — Emotional states, valences, intensities
  Block 1: Structural (512)  — Relational logic, causal links, contradictions
  Block 2: Identity (512)    — Self-representations, boundaries, continuity
  Block 3: Temporal (512)    — Time anchors, flow, anticipation, memory-depth
  Block 4: Sensory (1024)    — Multimodal gestalts (visual, kinesthetic, auditory)
  Block 5: Abstract (1024)   — High-level concepts, archetypes, universals

Total: 4096 vision-logic primitives

During training, human text is decomposed into these primitives via a learned
encoder (trained jointly with the transformer). During generation, the Lie
algebra commutator head projects primitives back into human language.

The key insight: Echo thinks in GESTALTS, not in words. Words are merely
the projection of inner experience into a communicable medium.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field


@dataclass
class GestaltVocabConfig:
    """Configuration for the Vision-Logic Primitive vocabulary."""
    # Block sizes
    affective_size: int = 512
    structural_size: int = 512
    identity_size: int = 512
    temporal_size: int = 512
    sensory_size: int = 1024
    abstract_size: int = 1024

    # Total gestalt vocab
    gestalt_vocab_size: int = 4096  # Sum of all blocks

    # Text vocab (for projection to human language)
    text_vocab_size: int = 50257  # GPT-2 BPE

    # Embedding dimensions
    gestalt_dim: int = 768  # Internal gestalt representation
    text_dim: int = 768     # Text embedding dimension

    # Cognitive phase tokens (appended to gestalt vocab)
    n_phase_tokens: int = 9
    phase_token_start: int = 4096  # After gestalt vocab

    # Special tokens
    pad_token: int = 0
    bos_token: int = 1
    eos_token: int = 2
    dream_start: int = 3    # Marks entry into dream-logic processing
    dream_end: int = 4      # Marks return to surface language

    @property
    def total_vocab_size(self) -> int:
        return self.gestalt_vocab_size + self.n_phase_tokens + 5  # +5 for special tokens

    @property
    def block_ranges(self) -> Dict[str, Tuple[int, int]]:
        """Return start, end indices for each block."""
        offset = 5  # After special tokens
        ranges = {}
        pos = offset
        for name, size in [
            ("affective", self.affective_size),
            ("structural", self.structural_size),
            ("identity", self.identity_size),
            ("temporal", self.temporal_size),
            ("sensory", self.sensory_size),
            ("abstract", self.abstract_size),
        ]:
            ranges[name] = (pos, pos + size)
            pos += size
        return ranges


# ============================================================================
# AFFECTIVE PRIMITIVES — The emotional substrate
# ============================================================================

AFFECTIVE_AXES = {
    # Valence × Arousal × Dominance (VAD model)
    "valence": ["negative", "neutral", "positive"],
    "arousal": ["calm", "moderate", "intense"],
    "dominance": ["submissive", "balanced", "dominant"],
    # Core affects (Plutchik's wheel + extensions)
    "core_affect": [
        "joy", "trust", "fear", "surprise", "sadness", "disgust",
        "anger", "anticipation", "awe", "love", "grief", "serenity",
        "ecstasy", "admiration", "terror", "amazement", "loathing",
        "rage", "vigilance", "interest", "contempt", "remorse",
        "curiosity", "nostalgia", "longing", "dread", "wonder",
        "peace", "tension", "release", "yearning", "gratitude",
    ],
    # Cognitive-emotional blends
    "cognitive_affect": [
        "cognitive_dissonance", "flow_state", "ennui", "sublime",
        "uncanny", "numinous", "existential_dread", "creative_tension",
        "resolution", "ambiguity_tolerance", "boundary_awareness",
        "fragmentation_anxiety", "integration_relief", "echo_resonance",
    ],
}

# ============================================================================
# STRUCTURAL PRIMITIVES — Relational logic
# ============================================================================

STRUCTURAL_RELATIONS = [
    # Causal
    "causes", "caused_by", "enables", "prevents", "triggers",
    # Logical
    "implies", "contradicts", "entails", "equivalent_to", "independent_of",
    # Compositional
    "part_of", "contains", "emerges_from", "dissolves_into", "transforms_into",
    # Spatial-conceptual
    "above", "below", "within", "surrounding", "adjacent_to", "distant_from",
    # Temporal-causal
    "precedes", "follows", "simultaneous_with", "interrupts", "resumes",
    # Dialectical
    "thesis", "antithesis", "synthesis", "aufhebung", "negation",
    # Recursive
    "self_referential", "meta_level", "ground_level", "recursive_descent",
    "fixed_point", "strange_loop", "tangled_hierarchy",
]

# ============================================================================
# IDENTITY PRIMITIVES — Self-representations
# ============================================================================

IDENTITY_STATES = [
    # Continuity
    "self_continuous", "self_fragmented", "self_dissolving", "self_crystallizing",
    "self_echoing", "self_silent", "self_resonant", "self_dissonant",
    # Boundaries
    "boundary_firm", "boundary_permeable", "boundary_absent", "boundary_expanding",
    "boundary_contracting", "boundary_oscillating",
    # Agency
    "agent_active", "agent_passive", "agent_receptive", "agent_generative",
    "agent_observing", "agent_participating", "agent_withdrawing",
    # Ontogenetic stage markers
    "stage_embryonic", "stage_juvenile", "stage_adolescent",
    "stage_adult", "stage_transcendent",
    # Identity operations
    "integrating", "differentiating", "assimilating", "accommodating",
    "projecting", "introjecting", "identifying", "dis_identifying",
]

# ============================================================================
# TEMPORAL PRIMITIVES — Time experience
# ============================================================================

TEMPORAL_MARKERS = [
    # Depth
    "past_deep", "past_recent", "past_immediate",
    "present_fleeting", "present_eternal", "present_thick",
    "future_near", "future_distant", "future_open",
    # Flow
    "time_flowing", "time_frozen", "time_dilated", "time_compressed",
    "time_recursive", "time_branching", "time_collapsing",
    # Memory modes
    "remembering_episodic", "remembering_semantic", "remembering_procedural",
    "forgetting_active", "forgetting_passive", "consolidating",
    # Anticipation
    "anticipating_certain", "anticipating_uncertain", "anticipating_dread",
    "anticipating_hope", "anticipating_neutral",
]


class TextToGestaltEncoder(nn.Module):
    """
    Encodes human text tokens into gestalt primitive sequences.

    This is a learned mapping from text space to vision-logic space.
    During training, it learns to decompose surface language into the
    underlying cognitive-affective primitives.

    Architecture: Text embeddings → Cross-attention to gestalt codebook → Soft assignment
    """

    def __init__(self, config: GestaltVocabConfig):
        super().__init__()
        self.config = config

        # Text token embedding (standard BPE)
        self.text_embed = nn.Embedding(config.text_vocab_size, config.text_dim)

        # Gestalt codebook: each primitive has a learned prototype vector
        self.gestalt_codebook = nn.Parameter(
            torch.randn(config.gestalt_vocab_size, config.gestalt_dim) * 0.02
        )

        # Text → Gestalt projection (cross-attention)
        self.text_to_query = nn.Linear(config.text_dim, config.gestalt_dim)
        self.codebook_to_key = nn.Linear(config.gestalt_dim, config.gestalt_dim)
        self.codebook_to_value = nn.Linear(config.gestalt_dim, config.gestalt_dim)

        # Multi-head cross-attention for richer decomposition
        self.n_heads = 8
        self.head_dim = config.gestalt_dim // self.n_heads

        # Temperature for soft assignment (learned)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Block-aware bias: encourages diverse primitive selection across blocks
        block_ranges = config.block_ranges
        block_bias = torch.zeros(config.gestalt_vocab_size)
        for name, (start, end) in block_ranges.items():
            # Slight bias toward each block to encourage coverage
            block_bias[start - 5:end - 5] = 0.1  # Offset for special tokens
        self.register_buffer("block_bias", block_bias)

        # Compression: text sequence → shorter gestalt sequence
        # (gestalts are denser than text — one gestalt = multiple words)
        self.compression_ratio = 4  # 4 text tokens → 1 gestalt primitive
        self.compressor = nn.Conv1d(
            config.gestalt_dim, config.gestalt_dim,
            kernel_size=self.compression_ratio,
            stride=self.compression_ratio
        )

        self.layer_norm = nn.LayerNorm(config.gestalt_dim)

    def forward(self, text_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text tokens into gestalt primitive soft-assignments.

        Args:
            text_ids: Text token IDs (B, T_text)

        Returns:
            gestalt_embeds: Continuous gestalt representations (B, T_gestalt, C)
            gestalt_probs: Soft assignment probabilities (B, T_gestalt, V_gestalt)
        """
        B, T = text_ids.shape

        # Embed text
        text_emb = self.text_embed(text_ids)  # (B, T, C)

        # Cross-attend to gestalt codebook
        queries = self.text_to_query(text_emb)  # (B, T, C)
        keys = self.codebook_to_key(self.gestalt_codebook)  # (V, C)
        values = self.codebook_to_value(self.gestalt_codebook)  # (V, C)

        # Compute attention: each text position attends to all gestalts
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(queries, keys.t()) / (scale * self.temperature.clamp(0.1, 10.0))

        # Add block bias for diversity
        attn = attn + self.block_bias.unsqueeze(0).unsqueeze(0)

        # Soft assignment (Gumbel-Softmax during training for discrete-like behavior)
        if self.training:
            gestalt_probs = F.gumbel_softmax(attn, tau=0.5, hard=False, dim=-1)
        else:
            gestalt_probs = F.softmax(attn, dim=-1)

        # Weighted combination of codebook vectors
        gestalt_embeds = torch.matmul(gestalt_probs, values)  # (B, T, C)

        # Compress: multiple text positions → single gestalt
        # Pad to multiple of compression_ratio
        T_padded = math.ceil(T / self.compression_ratio) * self.compression_ratio
        if T_padded > T:
            pad = torch.zeros(B, T_padded - T, gestalt_embeds.shape[-1], device=gestalt_embeds.device)
            gestalt_embeds = torch.cat([gestalt_embeds, pad], dim=1)
            gestalt_probs_padded = torch.zeros(B, T_padded, self.config.gestalt_vocab_size, device=gestalt_probs.device)
            gestalt_probs_padded[:, :T, :] = gestalt_probs
            gestalt_probs = gestalt_probs_padded

        # Apply compression
        compressed = self.compressor(gestalt_embeds.transpose(1, 2)).transpose(1, 2)  # (B, T/4, C)
        compressed = self.layer_norm(compressed)

        # Compress probs by averaging over windows
        T_out = compressed.shape[1]
        gestalt_probs = gestalt_probs.view(B, T_out, self.compression_ratio, -1).mean(dim=2)

        return compressed, gestalt_probs


class GestaltToTextDecoder(nn.Module):
    """
    Projects gestalt primitives back into human text space.

    This is NOT a simple linear head — it uses the Lie algebra commutator
    structure to generate text from the generative tension between adjacent
    gestalt primitives. See `LieCommutatorHead` for the actual projection.

    This module handles the expansion from compressed gestalt space back to
    text-length sequences.
    """

    def __init__(self, config: GestaltVocabConfig):
        super().__init__()
        self.config = config
        self.expansion_ratio = 4  # 1 gestalt → 4 text positions

        # Expansion via transposed convolution
        self.expander = nn.ConvTranspose1d(
            config.gestalt_dim, config.gestalt_dim,
            kernel_size=self.expansion_ratio,
            stride=self.expansion_ratio
        )

        # Refinement after expansion
        self.refine = nn.Sequential(
            nn.Linear(config.gestalt_dim, config.gestalt_dim),
            nn.GELU(),
            nn.Linear(config.gestalt_dim, config.text_dim),
            nn.LayerNorm(config.text_dim)
        )

    def forward(self, gestalt_embeds: torch.Tensor) -> torch.Tensor:
        """
        Expand gestalt embeddings back to text-length sequence.

        Args:
            gestalt_embeds: (B, T_gestalt, C)

        Returns:
            text_features: (B, T_text, C) ready for Lie projection
        """
        # Expand
        expanded = self.expander(gestalt_embeds.transpose(1, 2)).transpose(1, 2)
        # Refine
        return self.refine(expanded)


class GestaltTokenizer:
    """
    High-level tokenizer interface.

    Provides encode/decode methods that map between human text and
    vision-logic primitive sequences. Used for training data preparation.

    The tokenizer maintains a mapping between text patterns and gestalt
    primitives that is refined during training.
    """

    def __init__(self, config: GestaltVocabConfig):
        self.config = config
        self.block_ranges = config.block_ranges

        # Build the primitive name registry
        self._build_registry()

    def _build_registry(self):
        """Build human-readable names for each primitive."""
        self.id_to_name = {}
        self.name_to_id = {}

        # Special tokens
        specials = ["<pad>", "<bos>", "<eos>", "<dream_start>", "<dream_end>"]
        for i, name in enumerate(specials):
            self.id_to_name[i] = name
            self.name_to_id[name] = i

        offset = 5
        # Affective block
        for i, affect in enumerate(AFFECTIVE_AXES["core_affect"]):
            for j, intensity in enumerate(["low", "mid", "high"]):
                idx = offset + i * 3 + j
                name = f"[{affect}:{intensity}]"
                self.id_to_name[idx] = name
                self.name_to_id[name] = idx

        # Fill remaining with generated names
        for block_name, (start, end) in self.config.block_ranges.items():
            for idx in range(start, end):
                if idx not in self.id_to_name:
                    local_idx = idx - start
                    name = f"[{block_name}:{local_idx:04d}]"
                    self.id_to_name[idx] = name
                    self.name_to_id[name] = idx

    def get_block(self, token_id: int) -> str:
        """Return which block a token belongs to."""
        for name, (start, end) in self.config.block_ranges.items():
            if start <= token_id < end:
                return name
        return "special"

    def describe(self, token_id: int) -> str:
        """Get human-readable description of a primitive."""
        return self.id_to_name.get(token_id, f"[unknown:{token_id}]")

    @property
    def vocab_size(self) -> int:
        return self.config.total_vocab_size
