import torch
from netrain.models.experimental.operational_closure import CognitiveFusionReactor

def test_fusion_reactor():
    print("Testing Cognitive Fusion Reactor...")
    d_model = 64
    B, T = 2, 10
    
    reactor = CognitiveFusionReactor(d_model)
    
    past = torch.randn(B, T, d_model)
    present = torch.randn(B, T, d_model)
    future = torch.randn(B, T, d_model)
    hormones = torch.rand(B, 5)
    curvature = torch.randn(1)
    
    out = reactor(past, present, future, hormones, curvature)
    
    assert out["closed_state"].shape == (B, T, d_model)
    assert out["halo_intensity"].shape == (B, T)
    assert out["closure_gap"].shape == (1,)
    
    print(f"  Arc Halo Intensity mean: {out['halo_intensity'].mean().item():.6f}")
    print(f"  Distance to Event Horizon: {out['closure_gap'].item():.6f}")
    print("  Reactor operational closure achieved.")

if __name__ == "__main__":
    test_fusion_reactor()
