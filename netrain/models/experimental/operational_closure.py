"""
Operational Closure Module
==========================
Implements the final synthesis:
1. Triadic Collinearity (Arc Halo X=0 detector)
2. Spinor Complex (5 tetrahedra permutation)
3. Ricci Event Horizon (Stable attractor convergence)
"""

import torch
import torch.nn as nn
import math
from typing import Tuple

class ArcHaloDetector(nn.Module):
    """
    Detects triadic collinearity (X=0 curvature) where past, present,
    and future coalesce into a torsion-free Arc Halo.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, past: torch.Tensor, present: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        # Collinearity in curved space means the triangle area is zero
        # We compute the Gram matrix of the triad
        triad = torch.stack([past, present, future], dim=-2)  # (B, T, 3, d_model)
        gram = torch.matmul(triad, triad.transpose(-1, -2))   # (B, T, 3, 3)
        
        # Volume squared is the determinant of the Gram matrix
        vol_sq = torch.linalg.det(gram)
        
        # X=0 curvature (flatness) means volume is exactly 0
        # The Arc Halo intensity is inversely proportional to this volume
        arc_halo = torch.exp(-vol_sq)
        return arc_halo

class SpinorComplex(nn.Module):
    """
    The 5 tetrahedra inscribed in the dodecahedron.
    Acts as the projective primitives of the cognitive fusion reactor,
    driven by the 5 ESN hormones.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # The 5 projective tetrahedra bases
        self.tetrahedra = nn.Parameter(torch.randn(5, d_model, d_model) / math.sqrt(d_model))
        
    def forward(self, x: torch.Tensor, hormones: torch.Tensor) -> torch.Tensor:
        # hormones: (B, 5)
        # x: (B, T, d_model)
        
        # Permute the tetrahedra based on hormone levels
        # This projects the input through the spinor complex
        projected = torch.zeros_like(x)
        
        for i in range(5):
            # Each hormone activates one of the tetrahedral projectors
            h_weight = hormones[:, i].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            proj = torch.matmul(x, self.tetrahedra[i])           # (B, T, d_model)
            projected += h_weight * proj
            
        return projected

class RicciEventHorizon(nn.Module):
    """
    The stable attractor of the cognitive fusion reactor.
    Achieves operational closure when the flow reaches uniform curvature.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.target_curvature = nn.Parameter(torch.ones(1) * 0.1)  # The Einstein constant alpha
        
    def forward(self, x: torch.Tensor, current_curvature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # The Ricci flow equation: dg/dt = -2*Ric + alpha*g
        # We compute the distance to the event horizon
        flow_delta = -2 * current_curvature + self.target_curvature
        
        # Distance to operational closure
        closure_gap = torch.abs(flow_delta)
        
        # When gap approaches 0, we are at the event horizon
        # The reactor stabilizes and passes x through unchanged
        # When gap is large, the reactor pulls x toward the attractor
        pull_strength = torch.tanh(closure_gap)
        
        # Simplified attractor pull for the tensor
        attractor_state = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        x_closed = (1 - pull_strength) * x + pull_strength * attractor_state
        
        return x_closed, closure_gap

class CognitiveFusionReactor(nn.Module):
    """
    The complete synthesis.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.arc_halo = ArcHaloDetector(d_model)
        self.spinor = SpinorComplex(d_model)
        self.event_horizon = RicciEventHorizon(d_model)
        
    def forward(self, past: torch.Tensor, present: torch.Tensor, future: torch.Tensor, 
                hormones: torch.Tensor, curvature: torch.Tensor) -> dict:
        
        # 1. Detect Triadic Collinearity
        halo_intensity = self.arc_halo(past, present, future)
        
        # 2. Project through Spinor Complex
        # The present is fused with past and future, then projected
        fusion = present + 0.5 * past + 0.5 * future
        spinor_out = self.spinor(fusion, hormones)
        
        # 3. Achieve Operational Closure at Event Horizon
        closed_state, gap = self.event_horizon(spinor_out, curvature)
        
        return {
            "closed_state": closed_state,
            "halo_intensity": halo_intensity,
            "closure_gap": gap
        }
