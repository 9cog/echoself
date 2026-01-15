"""
Custom layers for Deep Tree Echo Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class TreeAttention(nn.Module):
    """
    Tree-structured attention mechanism with hierarchical processing.
    """
    
    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        tree_depth: int = 3,
        branch_factor: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.tree_depth = tree_depth
        self.branch_factor = branch_factor
        self.head_dim = n_embd // n_heads
        
        # Query, Key, Value projections for each tree level
        self.tree_projections = nn.ModuleList([
            nn.ModuleDict({
                'query': nn.Linear(n_embd, n_embd),
                'key': nn.Linear(n_embd, n_embd),
                'value': nn.Linear(n_embd, n_embd),
            })
            for _ in range(tree_depth)
        ])
        
        # Branch gating mechanism
        self.branch_gates = nn.ModuleList([
            nn.Linear(n_embd, branch_factor)
            for _ in range(tree_depth - 1)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for each level
        self.level_norms = nn.ModuleList([
            nn.LayerNorm(n_embd) for _ in range(tree_depth)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through tree attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, n_embd]
        """
        batch_size, seq_len, _ = x.shape
        
        # Process through tree levels
        tree_outputs = []
        current_x = x
        
        for level in range(self.tree_depth):
            # Get Q, K, V for current level
            proj = self.tree_projections[level]
            q = proj['query'](current_x)
            k = proj['key'](current_x)
            v = proj['value'](current_x)
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            
            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                # Expand mask to match scores shape
                # attention_mask is [batch_size, seq_len]
                # scores is [batch_size, n_heads, seq_len, seq_len]
                if attention_mask.dim() == 2:
                    # Create causal mask instead of using the provided mask
                    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
                    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
                else:
                    scores = scores.masked_fill(attention_mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(self.dropout(attn_weights), v)
            
            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.n_embd
            )
            
            # Apply layer norm
            attn_output = self.level_norms[level](attn_output)
            tree_outputs.append(attn_output)
            
            # Apply branch gating if not last level
            if level < self.tree_depth - 1:
                gate_weights = F.softmax(self.branch_gates[level](current_x), dim=-1)
                # Weighted combination for next level
                current_x = current_x + attn_output
        
        # Combine tree outputs
        combined = sum(tree_outputs) / len(tree_outputs)
        output = self.out_proj(combined)
        
        return self.dropout(output)


class EchoLayer(nn.Module):
    """
    Echo layer that maintains and propagates historical information.
    """
    
    def __init__(
        self,
        n_embd: int,
        echo_depth: int = 3,
        echo_decay: float = 0.95
    ):
        super().__init__()
        self.n_embd = n_embd
        self.echo_depth = echo_depth
        self.echo_decay = echo_decay
        
        # Echo state processing
        self.echo_transform = nn.Linear(n_embd * 2, n_embd)
        self.echo_gate = nn.Linear(n_embd * 2, n_embd)
        
        # Echo memory cells
        self.echo_cells = nn.ModuleList([
            nn.GRUCell(n_embd, n_embd) for _ in range(echo_depth)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        echo_state: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through echo layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            echo_state: Previous echo states
            
        Returns:
            Output tensor and new echo states
        """
        batch_size, seq_len, _ = x.shape
        
        if echo_state is None:
            echo_state = [torch.zeros(batch_size, self.n_embd, device=x.device) 
                         for _ in range(self.echo_depth)]
        
        # Process each position in sequence
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Update echo states through depth levels
            current_input = x_t
            level_outputs = []
            new_echo_state = []
            
            for level in range(self.echo_depth):
                # Update echo state at this level
                new_state = self.echo_cells[level](current_input, echo_state[level])
                new_echo_state.append(new_state)
                level_outputs.append(new_state)
                
                # Decay and propagate to next level
                current_input = new_state * self.echo_decay
            
            # Update echo_state for next timestep
            echo_state = new_echo_state
            
            # Combine echo information
            combined_echo = sum(level_outputs) / len(level_outputs)
            
            # Gate mechanism for echo integration
            concat_input = torch.cat([x_t, combined_echo], dim=-1)
            gate = torch.sigmoid(self.echo_gate(concat_input))
            transformed = self.echo_transform(concat_input)
            
            # Apply gated echo
            output = gate * transformed + (1 - gate) * x_t
            outputs.append(output)
        
        # Stack outputs
        output_tensor = torch.stack(outputs, dim=1)
        
        # Return the final echo state
        return output_tensor, echo_state


class RecursiveAttention(nn.Module):
    """
    Recursive attention mechanism for deep reasoning.
    """
    
    def __init__(
        self,
        n_embd: int,
        max_depth: int = 3
    ):
        super().__init__()
        self.n_embd = n_embd
        self.max_depth = max_depth
        
        # Recursive processing layers
        self.recursive_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, n_embd),
                nn.LayerNorm(n_embd),
                nn.ReLU(),
                nn.Linear(n_embd, n_embd)
            )
            for _ in range(max_depth)
        ])
        
        # Depth gating
        self.depth_gate = nn.Linear(n_embd, max_depth)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply recursive attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            
        Returns:
            Processed tensor
        """
        # Determine depth weights
        depth_weights = F.softmax(self.depth_gate(x.mean(dim=1)), dim=-1)
        
        # Apply recursive processing
        outputs = []
        current = x
        
        for depth in range(self.max_depth):
            current = self.recursive_layers[depth](current) + current
            outputs.append(current)
        
        # Weighted combination based on depth
        output = torch.zeros_like(x)
        for depth in range(self.max_depth):
            weight = depth_weights[:, depth:depth+1].unsqueeze(1)
            output = output + weight * outputs[depth]
        
        return output


class HierarchicalPooling(nn.Module):
    """
    Hierarchical pooling for tree-structured representations.
    """
    
    def __init__(
        self,
        n_embd: int,
        n_levels: int = 3
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_levels = n_levels
        
        # Pooling layers for each level
        self.pool_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_embd)
            )
            for _ in range(n_levels)
        ])
        
        # Combination layer
        self.combine = nn.Linear(n_embd * n_levels, n_embd)
    
    def forward(
        self,
        x: torch.Tensor,
        level_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Apply hierarchical pooling.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            level_features: Optional features from different levels
            
        Returns:
            Pooled representation
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply pooling at different granularities
        pooled_features = []
        
        for level in range(self.n_levels):
            # Determine pooling window size
            window_size = 2 ** level
            
            if window_size > seq_len:
                # Global pooling
                pooled = x.mean(dim=1, keepdim=True)
                pooled = pooled.expand(-1, seq_len, -1)
            else:
                # Local pooling with stride
                pooled_list = []
                for i in range(0, seq_len, window_size):
                    end = min(i + window_size, seq_len)
                    window = x[:, i:end, :]
                    pooled_window = window.mean(dim=1, keepdim=True)
                    pooled_list.extend([pooled_window] * (end - i))
                
                pooled = torch.cat(pooled_list[:seq_len], dim=1)
            
            # Transform pooled features
            pooled = self.pool_layers[level](pooled)
            pooled_features.append(pooled)
        
        # Combine all levels
        combined = torch.cat(pooled_features, dim=-1)
        output = self.combine(combined)
        
        return output


class MemoryBank(nn.Module):
    """
    Memory bank for storing and retrieving long-term dependencies.
    """
    
    def __init__(
        self,
        n_embd: int,
        memory_size: int = 512
    ):
        super().__init__()
        self.n_embd = n_embd
        self.memory_size = memory_size
        
        # Memory storage
        self.register_buffer('memory', torch.zeros(memory_size, n_embd))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Query, key, value projections for memory
        self.memory_query = nn.Linear(n_embd, n_embd)
        self.memory_key = nn.Linear(n_embd, n_embd)
        self.memory_value = nn.Linear(n_embd, n_embd)
        
        # Output projection
        self.memory_out = nn.Linear(n_embd, n_embd)
    
    def update(self, x: torch.Tensor):
        """
        Update memory bank with new information.
        
        Args:
            x: New information to store [batch_size, seq_len, n_embd]
        """
        with torch.no_grad():
            batch_size, seq_len, _ = x.shape
            
            # Flatten batch and sequence dimensions
            x_flat = x.view(-1, self.n_embd)
            
            # Determine how many items to store
            n_items = min(x_flat.size(0), self.memory_size)
            
            # Update memory in circular fashion
            ptr = self.memory_ptr.item()
            
            for i in range(n_items):
                self.memory[ptr] = x_flat[i].clone()
                ptr = (ptr + 1) % self.memory_size
            
            self.memory_ptr[0] = ptr
    
    def query(self, x: torch.Tensor) -> torch.Tensor:
        """
        Query memory bank for relevant information.
        
        Args:
            x: Query tensor [batch_size, seq_len, n_embd]
            
        Returns:
            Retrieved memory context
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute query, key, value
        q = self.memory_query(x)  # [batch_size, seq_len, n_embd]
        k = self.memory_key(self.memory.unsqueeze(0))  # [1, memory_size, n_embd]
        v = self.memory_value(self.memory.unsqueeze(0))  # [1, memory_size, n_embd]
        
        # Expand for batch
        k = k.expand(batch_size, -1, -1)
        v = v.expand(batch_size, -1, -1)
        
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.n_embd)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Retrieve weighted values
        memory_context = torch.bmm(attn_weights, v)
        
        # Project output
        output = self.memory_out(memory_context)
        
        return output