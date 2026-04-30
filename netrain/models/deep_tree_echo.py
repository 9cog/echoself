"""
Deep Tree Echo Transformer Model

A hierarchical transformer with recursive echo connections and tree-structured attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import math

from .layers import (
    TreeAttention,
    EchoLayer,
    HierarchicalPooling,
    RecursiveAttention,
    MemoryBank
)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings for improved position encoding."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute positional embeddings
        position = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", position, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class DeepTreeEchoBlock(nn.Module):
    """Single block of the Deep Tree Echo Transformer."""
    
    def __init__(self, config: Dict[str, Any], layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        arch = config['architecture']
        self.n_embd = arch['n_embd']
        self.n_heads = arch['n_heads']
        
        # Multi-head attention with optional tree structure
        self.use_tree_attention = layer_idx in arch.get('echo_layers', [])
        
        if self.use_tree_attention:
            self.attention = TreeAttention(
                n_embd=self.n_embd,
                n_heads=self.n_heads,
                tree_depth=arch['tree_depth'],
                branch_factor=arch['branch_factor'],
                dropout=arch['attention'].get('attention_dropout', 0.1)
            )
        else:
            self.attention = nn.MultiheadAttention(
                self.n_embd,
                self.n_heads,
                dropout=arch['attention'].get('attention_dropout', 0.1),
                batch_first=True
            )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd),
            nn.GELU(),
            nn.Linear(4 * self.n_embd, self.n_embd),
            nn.Dropout(arch['attention'].get('attention_dropout', 0.1))
        )
        
        # Echo layer if this is an echo layer
        if self.use_tree_attention and arch['echo']['enable_recursive_attention']:
            self.echo_layer = EchoLayer(
                n_embd=self.n_embd,
                echo_depth=arch['echo']['echo_depth'],
                echo_decay=arch['echo']['echo_decay']
            )
        else:
            self.echo_layer = None
        
        # Recursive attention if enabled
        if arch['echo']['enable_recursive_attention'] and layer_idx > arch['n_layers'] // 2:
            self.recursive_attention = RecursiveAttention(
                n_embd=self.n_embd,
                max_depth=arch['echo']['echo_depth']
            )
        else:
            self.recursive_attention = None
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        echo_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the block."""
        
        # Self-attention with residual connection
        residual = x
        x = self.ln1(x)
        
        if self.use_tree_attention:
            attn_out = self.attention(x, attention_mask)
        else:
            # For standard MultiheadAttention, we need to reshape the mask
            if attention_mask is not None:
                # Convert batch mask to proper format for MultiheadAttention
                # From [batch_size, seq_len] to [seq_len, seq_len]
                seq_len = x.size(1)
                # Create a causal mask
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
                attn_out, _ = self.attention(x, x, x, attn_mask=causal_mask)
            else:
                attn_out, _ = self.attention(x, x, x)
        
        x = residual + attn_out
        
        # Apply echo layer if present
        new_echo_state = None
        if self.echo_layer is not None:
            x, new_echo_state = self.echo_layer(x, echo_state)
        
        # Apply recursive attention if present
        if self.recursive_attention is not None:
            x = self.recursive_attention(x)
        
        # Feed-forward network with residual connection
        residual = x
        x = self.ln2(x)
        x = residual + self.ffn(x)
        
        return x, new_echo_state


class DeepTreeEchoTransformer(nn.Module):
    """
    Deep Tree Echo Transformer Model
    
    A hierarchical transformer architecture with:
    - Tree-structured attention mechanisms
    - Recursive echo connections
    - Hierarchical pooling
    - Memory banks for long-term dependencies
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        arch = config['architecture']
        
        # Model dimensions
        self.n_layers = arch['n_layers']
        self.n_embd = arch['n_embd']
        self.vocab_size = arch['vocab_size']
        self.block_size = arch['block_size']
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding = nn.Embedding(self.block_size, self.n_embd)
        
        # Rotary position embeddings if enabled
        if arch['attention'].get('use_rotary_embeddings', False):
            self.rotary_emb = RotaryPositionalEmbedding(
                self.n_embd // arch['n_heads'],
                arch['attention'].get('max_position_embeddings', 2048)
            )
        else:
            self.rotary_emb = None
        
        # Dropout
        self.dropout = nn.Dropout(arch['attention'].get('attention_dropout', 0.1))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DeepTreeEchoBlock(config, i) for i in range(self.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(self.n_embd)
        
        # Output projection
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        
        # Memory bank if enabled
        if arch['echo'].get('use_memory_bank', False):
            self.memory_bank = MemoryBank(
                n_embd=self.n_embd,
                memory_size=arch['echo'].get('memory_size', 512)
            )
        else:
            self.memory_bank = None
        
        # Hierarchical pooling if enabled
        if arch['tree'].get('enable_hierarchical_pooling', False):
            self.hierarchical_pooling = HierarchicalPooling(
                n_embd=self.n_embd,
                n_levels=arch['tree_depth']
            )
        else:
            self.hierarchical_pooling = None
        
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
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for training [batch_size, seq_len]
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing loss and/or logits
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)
        
        # Position embeddings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        # Clamp positions to max block size
        pos = torch.clamp(pos, max=self.block_size - 1)
        pos_emb = self.position_embedding(pos)
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Update memory bank if present
        if self.memory_bank is not None:
            memory_context = self.memory_bank.query(x)
            x = x + memory_context
        
        # Pass through transformer blocks
        echo_states = []
        echo_state = None
        
        for block in self.blocks:
            x, echo_state = block(x, attention_mask, echo_state)
            if echo_state is not None:
                echo_states.append(echo_state)
        
        # Apply hierarchical pooling if enabled
        if self.hierarchical_pooling is not None:
            x = self.hierarchical_pooling(x, echo_states)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # Update memory bank with final representations
        if self.memory_bank is not None and self.training:
            self.memory_bank.update(x.detach())
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': x,
                'echo_states': echo_states
            }
        else:
            return (loss, logits) if loss is not None else logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        use_tree_beam: bool = False
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Starting token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            use_tree_beam: Whether to use tree beam search
            
        Returns:
            Generated token IDs
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Get logits
                outputs = self.forward(input_ids[:, -self.block_size:])
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                filtered_logits = self._top_k_top_p_filtering(logits, top_k, top_p)
                
                # Sample next token
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check for EOS token (assuming 50256 is EOS)
                if next_token.item() == 50256:
                    break
        
        return input_ids
    
    # ------------------------------------------------------------------
    # Topology instrumentation (Layer 0)
    # ------------------------------------------------------------------

    def get_topology_state(self) -> dict:
        """Return a snapshot of current topology and per-block gradient norms.

        Returns
        -------
        dict with keys:
          n_layers, tree_depth, echo_layers, echo_depth_per_block,
          memory_size, per_block_grad_norm
        """
        echo_layers = []
        echo_depth_per_block = {}
        per_block_grad_norm = {}

        for idx, block in enumerate(self.blocks):
            if block.use_tree_attention:
                echo_layers.append(idx)
            if block.echo_layer is not None:
                echo_depth_per_block[idx] = block.echo_layer.echo_depth
            # Per-block gradient norm (0.0 if no grads yet)
            norms = [
                p.grad.norm().item()
                for p in block.parameters()
                if p.grad is not None
            ]
            per_block_grad_norm[idx] = float(sum(norms) / max(len(norms), 1))

        arch = self.config["architecture"]
        tree_depth = arch.get("tree_depth", 1)

        return {
            "n_layers": self.n_layers,
            "tree_depth": tree_depth,
            "echo_layers": echo_layers,
            "echo_depth_per_block": echo_depth_per_block,
            "memory_size": (
                self.memory_bank.memory_size
                if self.memory_bank is not None
                else 0
            ),
            "per_block_grad_norm": per_block_grad_norm,
        }

    def set_tree_depth(self, depth: int) -> None:
        """Grow or shrink tree attention depth for all echo blocks in-place.

        New projection heads are zero-initialised; removed heads are discarded.
        The optimizer state for changed parameters is dropped (caller must
        rebuild the optimizer if precise momentum continuity is required).
        """
        import copy

        arch = self.config["architecture"]
        old_depth = arch.get("tree_depth", 1)
        if depth == old_depth:
            return

        arch["tree_depth"] = depth

        for block in self.blocks:
            if not block.use_tree_attention or not hasattr(block, "attention"):
                continue
            attn = block.attention
            if not isinstance(attn, TreeAttention):
                continue

            n_embd = attn.n_embd
            n_heads = attn.n_heads
            dropout = attn.dropout.p

            if depth > old_depth:
                # Add new projection layers for the extra depths
                for _ in range(depth - old_depth):
                    new_proj = nn.ModuleDict(
                        {
                            "query": nn.Linear(n_embd, n_embd),
                            "key":   nn.Linear(n_embd, n_embd),
                            "value": nn.Linear(n_embd, n_embd),
                        }
                    )
                    for m in new_proj.values():
                        nn.init.zeros_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    attn.tree_projections.append(new_proj)
                    attn.level_norms.append(nn.LayerNorm(n_embd))
                # Add branch gates for the new intermediate levels
                extra_gates = depth - 1 - len(attn.branch_gates)
                for _ in range(max(0, extra_gates)):
                    gate = nn.Linear(n_embd, attn.branch_factor)
                    nn.init.zeros_(gate.weight)
                    if gate.bias is not None:
                        nn.init.zeros_(gate.bias)
                    attn.branch_gates.append(gate)

            else:
                # Contract: keep only the first `depth` projections
                attn.tree_projections = nn.ModuleList(
                    list(attn.tree_projections)[:depth]
                )
                attn.level_norms = nn.ModuleList(
                    list(attn.level_norms)[:depth]
                )
                attn.branch_gates = nn.ModuleList(
                    list(attn.branch_gates)[: max(0, depth - 1)]
                )

            attn.tree_depth = depth

    def set_echo_depth(self, block_idx: int, depth: int) -> None:
        """Resize the EchoLayer GRU stack for a specific block.

        Existing GRU cells are preserved; new cells are zero-initialised.
        """
        if block_idx >= len(self.blocks):
            return
        block = self.blocks[block_idx]
        if block.echo_layer is None:
            return

        el = block.echo_layer
        old_depth = el.echo_depth
        if depth == old_depth:
            return

        n_embd = el.n_embd

        if depth > old_depth:
            for _ in range(depth - old_depth):
                new_cell = nn.GRUCell(n_embd, n_embd).to(
                    next(el.parameters()).device
                )
                # Zero-init to be additive-safe
                for p in new_cell.parameters():
                    nn.init.zeros_(p)
                el.echo_cells.append(new_cell)
        else:
            el.echo_cells = nn.ModuleList(list(el.echo_cells)[:depth])

        el.echo_depth = depth

    def set_memory_size(self, new_size: int) -> None:
        """Grow or shrink the MemoryBank while preserving stored memories."""
        if self.memory_bank is None:
            return
        mb = self.memory_bank
        old_size = mb.memory_size
        if new_size == old_size:
            return

        device = mb.memory.device
        new_mem = torch.zeros(new_size, mb.n_embd, device=device)

        if new_size > old_size:
            # Copy existing memories into the beginning of the new buffer
            new_mem[:old_size] = mb.memory.data
        else:
            # Truncate: keep most recent `new_size` entries.
            # ptr is the write cursor; entries before ptr are the most recent.
            ptr = int(mb.memory_ptr.item())
            if ptr >= new_size:
                # Enough valid recent entries exist before ptr
                tail = mb.memory.data[ptr - new_size : ptr].clone()
                new_mem = tail
            elif ptr > 0:
                # ptr < new_size: the buffer hasn't fully wrapped, use all until ptr
                new_mem[:ptr] = mb.memory.data[:ptr]
            # else ptr == 0: buffer is empty or just reset; new_mem stays zeros

        mb.memory = nn.Parameter(new_mem, requires_grad=False)
        mb.memory_ptr = nn.Parameter(
            torch.zeros(1, dtype=torch.long, device=device), requires_grad=False
        )
        mb.memory_size = new_size

    def activate_echo_layer(self, block_idx: int) -> None:
        """Promote a standard MHA block to TreeAttention + EchoLayer mid-training.

        The existing MHA weights are discarded; tree projections start
        zero-initialised so the transition is gradual (additive to residual).
        """
        if block_idx >= len(self.blocks):
            return
        block = self.blocks[block_idx]
        if block.use_tree_attention:
            return  # already active

        arch = self.config["architecture"]
        n_embd = block.n_embd
        n_heads = block.n_heads

        # Replace attention module
        block.attention = TreeAttention(
            n_embd=n_embd,
            n_heads=n_heads,
            tree_depth=arch.get("tree_depth", 1),
            branch_factor=arch.get("branch_factor", 4),
            dropout=arch["attention"].get("attention_dropout", 0.1),
        )
        # Zero-init so initial contribution is near-zero
        for m in block.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Add echo layer
        if arch["echo"]["enable_recursive_attention"]:
            block.echo_layer = EchoLayer(
                n_embd=n_embd,
                echo_depth=arch["echo"].get("echo_depth", 1),
                echo_decay=arch["echo"].get("echo_decay", 0.95),
            )

        block.use_tree_attention = True

    # ------------------------------------------------------------------
    # End topology instrumentation
    # ------------------------------------------------------------------

    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 0.0,
        filter_value: float = -float('Inf')
    ) -> torch.Tensor:
        """Filter logits using top-k and/or top-p filtering."""
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        
        return logits