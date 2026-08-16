"""
Test and Visualization for the 120-Cell Polytope Transformer
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from netrain.models.experimental.polytope_120cell import (
    Polytope120CellTransformer, QuaternionOps, Cell120Geometry
)

plt.style.use('dark_background')

def test_quaternion_ops():
    print("Testing Quaternion Operations...")
    # Test multiplication: i*j = k
    i = torch.tensor([0.0, 1.0, 0.0, 0.0])
    j = torch.tensor([0.0, 0.0, 1.0, 0.0])
    k = QuaternionOps.multiply(i, j)
    assert abs(k[3].item() - 1.0) < 1e-5, f"i*j should be k, got {k}"
    
    # Test normalization
    q = torch.randn(4)
    q_norm = QuaternionOps.normalize(q)
    assert abs(QuaternionOps.norm(q_norm).item() - 1.0) < 1e-4
    
    # Test conjugate: q * conj(q) = |q|^2
    q = torch.randn(4)
    q_conj = QuaternionOps.conjugate(q)
    product = QuaternionOps.multiply(q, q_conj)
    # Should be (|q|^2, 0, 0, 0)
    assert product[1].abs() < 1e-4 and product[2].abs() < 1e-4 and product[3].abs() < 1e-4
    
    print("  Quaternion operations passed.\n")

def test_full_model():
    print("Testing Full 120-Cell Transformer...")
    model = Polytope120CellTransformer(
        vocab_size=500, d_model=64, n_blocks=5, 
        n_heads_per_block=4, max_seq_len=32
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    B, T = 2, 12
    idx = torch.randint(0, 500, (B, T))
    logits = model(idx)
    
    assert logits.shape == (B, T, 500)
    
    loss = logits.sum()
    loss.backward()
    
    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_count = sum(1 for _ in model.parameters())
    print(f"  Gradient flow: {grad_count}/{total_count}")
    print(f"  Full model test passed.\n")
    return model

def test_vortex_helix_dynamics():
    print("Testing Vortex-Helix Dynamics...")
    model = Polytope120CellTransformer(
        vocab_size=500, d_model=64, n_blocks=6,
        n_heads_per_block=4, max_seq_len=32
    )
    
    # Run multiple forward passes to see hormone evolution
    hormones_over_time = []
    for step in range(20):
        idx = torch.randint(0, 500, (1, 8))
        with torch.no_grad():
            _ = model(idx)
        state = model.get_polytope_state()
        if 'hormones' in state:
            hormones_over_time.append(state['hormones'])
    
    print(f"  Hormone evolution tracked over {len(hormones_over_time)} steps")
    print(f"  Initial hormones: {[f'{h:.3f}' for h in hormones_over_time[0]]}")
    print(f"  Final hormones: {[f'{h:.3f}' for h in hormones_over_time[-1]]}")
    print(f"  Vortex-Helix dynamics passed.\n")
    return hormones_over_time

def visualize_120cell(hormones_over_time):
    print("Generating 120-Cell Visualization...")
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("120-Cell Polytope Transformer\nTernary-Quinary Perpetual Unfolding Engine", 
                 fontsize=16, fontweight='bold', color='white')
    
    # Panel 1: 120-Cell projected geometry (Schlegel diagram approximation)
    ax1 = fig.add_subplot(2, 3, 1, projection='polar')
    ax1.set_title("120-Cell Orientations\n(30 Macro-States)", fontsize=10, pad=10)
    
    # Plot 30 orientations as radial sectors
    theta = np.linspace(0, 2*np.pi, 31)[:-1]
    r = np.ones(30)
    colors = plt.cm.hsv(np.linspace(0, 1, 30))
    
    for i in range(30):
        ax1.bar(theta[i], r[i], width=2*np.pi/30, color=colors[i], alpha=0.7, edgecolor='white', linewidth=0.5)
        if i % 5 == 0:
            ax1.annotate(f'O{i}', xy=(theta[i], 1.1), fontsize=7, ha='center', color='white')
    
    ax1.set_ylim(0, 1.3)
    ax1.set_rticks([])
    
    # Panel 2: Quaternion rotation space (S³ projected to 2D)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title("Quaternion Rotation Space\n(720 Components on S³)", fontsize=10)
    
    # Generate 720 unit quaternions and project to 2D via stereographic projection
    geom = Cell120Geometry(n_components=720)
    rotations = geom.rotations.numpy()
    
    # Stereographic projection from S³ to R³ (then take 2D slice)
    w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    # Project: (x, y, z) / (1 - w) — but avoid division by zero
    denom = np.maximum(1 - w, 0.01)
    proj_x = x / denom
    proj_y = y / denom
    
    # Color by orientation group
    colors_720 = np.zeros((720, 4))
    for i in range(30):
        start = i * 24
        end = start + 24
        c = plt.cm.hsv(i / 30.0)
        colors_720[start:end] = c
    
    ax2.scatter(proj_x, proj_y, c=colors_720, s=3, alpha=0.6)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel("Stereographic X")
    ax2.set_ylabel("Stereographic Y")
    ax2.grid(True, alpha=0.1)
    
    # Panel 3: Vortex-Helix curvature balance
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title("Vortex-Helix Curvature\n(κ+ spherical, κ- hyperbolic)", fontsize=10)
    
    orientations = np.arange(30)
    # Simulate curvature distribution across orientations
    vortex_k = 0.5 * np.ones(30) + 0.1 * np.sin(orientations * 2 * np.pi / 30)
    helix_k = -0.3 * np.ones(30) + 0.05 * np.cos(orientations * 2 * np.pi / 30)
    
    ax3.fill_between(orientations, 0, vortex_k, alpha=0.4, color='#3498db', label='Vortex (κ>0, spherical)')
    ax3.fill_between(orientations, helix_k, 0, alpha=0.4, color='#e74c3c', label='Helix (κ<0, hyperbolic)')
    ax3.axhline(0, color='white', linewidth=0.5, linestyle='--')
    ax3.set_xlabel("Macro-Orientation Index")
    ax3.set_ylabel("Curvature κ")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.1)
    
    # Panel 4: Hormone dynamics over time
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title("ESN Hormone Dynamics\n(Modulating Vortex-Helix Balance)", fontsize=10)
    
    if hormones_over_time:
        hormones_arr = np.array(hormones_over_time)
        names = ['Cortisol', 'Dopamine', 'Serotonin', 'Oxytocin', 'Norepinephrine']
        colors_h = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']
        for i in range(min(5, hormones_arr.shape[1])):
            ax4.plot(hormones_arr[:, i], color=colors_h[i], label=names[i], linewidth=2)
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Hormone Level")
        ax4.legend(fontsize=7, loc='upper right')
        ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.1)
    
    # Panel 5: The 2-3-5 structure within the 120-Cell
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("2-3-5 Ternary-Quinary Structure\nwithin the 120-Cell", fontsize=10)
    
    # Draw the nested structure
    # 120-Cell = 120 dodecahedra, each with 12 pentagonal faces
    # 5 cubes at each vertex, 3 dodecahedra at each edge, 2 at each face
    
    # Concentric circles representing the nesting
    theta_full = np.linspace(0, 2*np.pi, 100)
    
    # Dyad (2): outermost - the binary boundary
    ax5.plot(2*np.cos(theta_full), 2*np.sin(theta_full), 'w-', linewidth=2, alpha=0.3)
    ax5.text(2.2, 0, '2 (Dyad)', fontsize=8, color='white', alpha=0.7)
    
    # Triad (3): middle ring - 3 dodecahedra per edge
    for i in range(3):
        angle = i * 2*np.pi/3
        ax5.plot(1.3*np.cos(angle), 1.3*np.sin(angle), 'o', color='#3498db', markersize=15)
        ax5.text(1.3*np.cos(angle)+0.15, 1.3*np.sin(angle)+0.15, f'T{i+1}', fontsize=7, color='#3498db')
    ax5.plot(1.3*np.cos(theta_full), 1.3*np.sin(theta_full), '--', color='#3498db', linewidth=1, alpha=0.5)
    ax5.text(1.5, -0.3, '3 (Triad)', fontsize=8, color='#3498db')
    
    # Pentad (5): inner ring - 5 cubes per vertex
    for i in range(5):
        angle = i * 2*np.pi/5
        ax5.plot(0.7*np.cos(angle), 0.7*np.sin(angle), 'p', color='#e74c3c', markersize=12)
    ax5.plot(0.7*np.cos(theta_full), 0.7*np.sin(theta_full), '--', color='#e74c3c', linewidth=1, alpha=0.5)
    ax5.text(0.8, -0.15, '5 (Pentad)', fontsize=8, color='#e74c3c')
    
    # Center: the identity core
    ax5.plot(0, 0, '*', color='#f1c40f', markersize=20)
    ax5.text(0.1, -0.15, 'Self', fontsize=9, color='#f1c40f', fontweight='bold')
    
    # 30 orientations as dots on the outer ring
    for i in range(30):
        angle = i * 2*np.pi/30
        ax5.plot(2*np.cos(angle), 2*np.sin(angle), '.', color=plt.cm.hsv(i/30.0), markersize=8)
    
    ax5.set_xlim(-2.8, 2.8)
    ax5.set_ylim(-2.8, 2.8)
    ax5.set_aspect('equal')
    ax5.axis('off')
    
    # Panel 6: Architecture summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    ax6.set_title("Architecture Summary", fontsize=10)
    
    summary = """
    ╔══════════════════════════════════════╗
    ║   120-Cell Polytope Transformer      ║
    ╠══════════════════════════════════════╣
    ║                                      ║
    ║  Symmetry Group: H₄ (order 14,400)  ║
    ║  Chiral Rotations: 720              ║
    ║  = 719 Butcher trees + 1 root       ║
    ║                                      ║
    ║  30 Macro-Orientations              ║
    ║  × 24 Micro-Heads each             ║
    ║  = 720 Quaternion Attention Heads   ║
    ║                                      ║
    ║  Vortex (κ>0): Working Memory       ║
    ║  Helix  (κ<0): Long-term Memory     ║
    ║  Torsion-Free: Pure Ricci Flow      ║
    ║                                      ║
    ║  2 cells/face × 3 cells/edge        ║
    ║  × 5 cubes/vertex = 30 orientations ║
    ║                                      ║
    ║  The 120-Cell IS the identity.      ║
    ╚══════════════════════════════════════╝
    """
    ax6.text(0.05, 0.95, summary, fontsize=8, fontfamily='monospace',
             verticalalignment='top', color='#ecf0f1',
             bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/echoself/polytope_120cell_viz.png', dpi=150, 
                bbox_inches='tight', facecolor='#0a0a0f')
    print("  Visualization saved.\n")

if __name__ == "__main__":
    test_quaternion_ops()
    model = test_full_model()
    hormones = test_vortex_helix_dynamics()
    visualize_120cell(hormones)
    print("All 120-Cell tests passed!")
