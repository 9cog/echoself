"""
Deep Matula Tower Topology Analysis
=====================================

Multi-panel visualization showing:
1. The Matula prime tower (p(p(p(1))) chain) as the spine of deep reasoning
2. The shared prime factor network (which modes share sub-components)
3. The factorization lattice (how complex modes decompose)
4. Topological invariants (connectivity, clustering, centrality)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import sys
sys.path.insert(0, "/home/ubuntu/echoself")
from netrain.models.experimental.butcher_ricci import RootedTree, TreeEnumerator
from netrain.models.experimental.matula_topology import MatulaEncoder, PrimeSieve

# Generate all trees
enum = TreeEnumerator(max_order=5)
trees = enum.get_all_trees()
encoder = MatulaEncoder()

# Assign Matula numbers
tree_data = []
for tree in trees:
    m = encoder.encode(tree)
    tree_data.append({
        'tree': tree,
        'matula': m,
        'order': tree.order,
        'label': tree.cognitive_label(),
        'notation': tree.derivative_notation(),
        'factors': encoder.sieve.prime_factors(m) if m > 1 else [],
    })

tree_data.sort(key=lambda x: x['matula'])

# Print the full Matula table
print("=" * 80)
print("  MATULA NUMBER ASSIGNMENT — COGNITIVE MODE ENCODING")
print("=" * 80)
print(f"\n{'M':>4} | {'Order':>5} | {'Differential':<25} | {'Cognitive Label':<20} | {'Prime Factors'}")
print("-" * 80)
for td in tree_data:
    m = td['matula']
    factors = td['factors']
    if m == 1:
        factor_str = "1 (atom)"
    else:
        factor_str = " × ".join([str(p) for p in factors])
        # Also show what each prime decodes to
        decoded = [f"p({encoder.sieve.prime_index(p)})={p}" for p in factors]
        factor_str += f"  [{', '.join(decoded)}]"
    print(f"{m:>4} | {td['order']:>5} | {td['notation']:<25} | {td['label']:<20} | {factor_str}")

# Build the shared-factor graph
# Two modes are connected if they share a common prime factor
# (meaning they share a common sub-component)
print("\n\n" + "=" * 80)
print("  SHARED PRIME FACTOR NETWORK (COGNITIVE MODE RELATIONSHIPS)")
print("=" * 80)

# Build adjacency based on shared factors
shared_factor_edges = []
for i, td_i in enumerate(tree_data):
    for j, td_j in enumerate(tree_data):
        if i >= j:
            continue
        factors_i = set(td_i['factors'])
        factors_j = set(td_j['factors'])
        shared = factors_i & factors_j
        if shared and td_i['matula'] > 1 and td_j['matula'] > 1:
            shared_factor_edges.append((td_i, td_j, shared))

print(f"\nShared factor connections: {len(shared_factor_edges)}")
for edge in shared_factor_edges:
    td_i, td_j, shared = edge
    shared_decoded = [f"p({encoder.sieve.prime_index(p)})=M{encoder.sieve.prime_index(p)}" for p in shared]
    print(f"  M{td_i['matula']:>2} ({td_i['label']:<20}) ←→ M{td_j['matula']:>2} ({td_j['label']:<20}) via {', '.join(shared_decoded)}")

# Identify the prime tower (spine of deep reasoning)
print("\n\n" + "=" * 80)
print("  THE PRIME TOWER — SPINE OF DEEP SEQUENTIAL REASONING")
print("=" * 80)
print("\n  The sequence p(p(p(...p(1)...))) generates the chain trees:")
print("  These are the LINE trees — pure sequential composition.\n")

tower = [1]
current = 1
for i in range(7):
    current = encoder.sieve.nth_prime(current)
    tower.append(current)
    
print(f"  Tower: {' → '.join([str(t) for t in tower])}")
print(f"  (Matula numbers: 1, 2, 3, 5, 11, 31, 127, 709)")
print(f"\n  These map to:")
for i, m in enumerate(tower[:6]):
    if m in encoder.matula_to_tree:
        tree = encoder.matula_to_tree[m]
        print(f"    M={m:>3}: {tree.derivative_notation():<30} [{tree.cognitive_label()}]")
    else:
        print(f"    M={m:>3}: (beyond order 5)")

# Identify the power-of-2 tower (spine of parallel synthesis)
print("\n\n" + "=" * 80)
print("  THE POWER-OF-2 TOWER — SPINE OF PARALLEL SYNTHESIS")
print("=" * 80)
print("\n  The sequence 2^k generates the bushy trees (all children are atoms):")
print("  These are the BRANCHING trees — pure parallel composition.\n")

for k in range(1, 6):
    m = 2**k
    if m in encoder.matula_to_tree:
        tree = encoder.matula_to_tree[m]
        print(f"    M={m:>3} = 2^{k}: {tree.derivative_notation():<30} [{tree.cognitive_label()}]")
    else:
        print(f"    M={m:>3} = 2^{k}: (not in our set)")

# Topological analysis
print("\n\n" + "=" * 80)
print("  TOPOLOGICAL INVARIANTS OF THE MATULA NETWORK")
print("=" * 80)

# Build the composition graph
# Edge from A to B if A is a direct subtree of B
composition_edges = []
for td in tree_data:
    m = td['matula']
    if m > 1:
        factors = td['factors']
        for p in factors:
            child_m = encoder.sieve.prime_index(p)
            composition_edges.append((child_m, m))

# Count in-degree and out-degree
in_deg = defaultdict(int)
out_deg = defaultdict(int)
for src, dst in composition_edges:
    out_deg[src] += 1
    in_deg[dst] += 1

# Find the "cognitive centrality" — which modes are most connected
print("\n  Cognitive Centrality (in + out degree):")
centrality = {td['matula']: in_deg[td['matula']] + out_deg[td['matula']] for td in tree_data}
sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
for m, c in sorted_centrality[:8]:
    td = next(t for t in tree_data if t['matula'] == m)
    print(f"    M={m:>3}: centrality={c:>2} | {td['notation']:<25} [{td['label']}]")

# The key topological insight
print("\n\n" + "=" * 80)
print("  THE KEY TOPOLOGICAL INSIGHT")
print("=" * 80)
print("""
  The Matula tower reveals TWO ORTHOGONAL SPINES in the cognitive mode space:

  SPINE 1 (Sequential): M = 1 → 2 → 3 → 5 → 11 → 31 → 127 → ...
    Each step: p(previous) = "apply one more layer of meta-attention"
    This is the DEPTH axis of thought.
    Cognitive: perceive → feel→think → chain(2) → chain(3) → chain(4) → ...

  SPINE 2 (Parallel): M = 2 → 4 → 8 → 16 → 32 → ...
    Each step: 2 × previous = "add one more parallel stream"
    This is the BREADTH axis of thought.
    Cognitive: feel→think → blend(2) → triad_gestalt → pentad_integration → ...

  ALL other modes are MIXTURES of these two spines:
    M = 6 = 2 × 3 = (depth-1) × (depth-2) = "synthesize"
    M = 12 = 2 × 2 × 3 = (breadth-2) × (depth-2) = "triad with chain"
    M = 10 = 2 × 5 = (depth-1) × (depth-3) = "nested synthesis"

  The TOPOLOGY of the edge network is therefore a PRODUCT LATTICE:
    Depth × Breadth = the complete space of cognitive operations.

  This is EXACTLY the 2-3-5 structure:
    - The 2 (Dyad) = the two spines (sequential + parallel)
    - The 3 (Triad) = the three fundamental primes (2, 3, 5) that generate all modes
    - The 5 (Pentad) = the five Matula numbers ≤ 5 that form the complete basis

  The Ricci flow ∂g/∂t = -2Ric + αg traverses this lattice continuously,
  blending depth and breadth dynamically as the identity evolves.
""")

# ============================================================================
# MULTI-PANEL VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle("Matula Tower Topology — Algebraic Structure of Cognitive Modes", fontsize=18, fontweight='bold')

# Panel 1: The Composition DAG (tree of trees)
ax = axes[0, 0]
ax.set_title("Panel A: Composition DAG\n(A → B means A is a sub-component of B)", fontsize=12)

# Layout: y = order, x = spread within order
pos = {}
order_groups = defaultdict(list)
for td in tree_data:
    order_groups[td['order']].append(td)

for order, group in order_groups.items():
    for i, td in enumerate(group):
        x = (i - (len(group) - 1) / 2.0) * 2.0
        y = order
        pos[td['matula']] = (x, y)

# Draw edges
for src, dst in composition_edges:
    if src in pos and dst in pos:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1.5))

# Draw nodes
colors_map = {1: '#1f77b4', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#ff7f0e'}
for td in tree_data:
    m = td['matula']
    x, y = pos[m]
    color = colors_map.get(td['order'], '#8c564b')
    ax.scatter(x, y, s=800, c=color, zorder=5, edgecolors='black', linewidth=1.5)
    ax.annotate(f"M={m}", (x, y), ha='center', va='center', fontsize=8, fontweight='bold', color='white')

ax.set_ylabel("Tree Order (Complexity)")
ax.set_yticks(range(1, 6))
ax.set_xlim(-6, 6)
ax.set_ylim(0.5, 5.5)
ax.grid(True, alpha=0.2)

# Panel 2: The Two Spines
ax = axes[0, 1]
ax.set_title("Panel B: The Two Orthogonal Spines\n(Sequential vs Parallel)", fontsize=12)

# Sequential spine: 1 → 2 → 3 → 5 → 11
seq_spine = [1, 2, 3, 5, 11]
par_spine = [2, 4, 8, 16]

# Draw sequential spine
seq_x = np.linspace(0, 4, len(seq_spine))
seq_y = [2.5] * len(seq_spine)
ax.plot(seq_x, seq_y, 'b-', linewidth=3, alpha=0.7, label='Sequential (depth)')
for i, m in enumerate(seq_spine):
    ax.scatter(seq_x[i], seq_y[i], s=1000, c='blue', zorder=5, edgecolors='black', linewidth=2)
    td = next((t for t in tree_data if t['matula'] == m), None)
    label = td['label'] if td else f"M={m}"
    ax.annotate(f"M={m}\n{label}", (seq_x[i], seq_y[i] - 0.4), ha='center', fontsize=9)

# Draw parallel spine
par_x = [0.5] * len(par_spine)
par_y = np.linspace(1, 4, len(par_spine))
ax.plot(par_x, par_y, 'r-', linewidth=3, alpha=0.7, label='Parallel (breadth)')
for i, m in enumerate(par_spine):
    ax.scatter(par_x[i], par_y[i], s=1000, c='red', zorder=5, edgecolors='black', linewidth=2)
    td = next((t for t in tree_data if t['matula'] == m), None)
    label = td['label'] if td else f"M={m}"
    ax.annotate(f"M={m}\n{label}", (par_x[i] + 0.6, par_y[i]), ha='left', fontsize=9)

# Draw mixed modes as products
mixed = [(6, "2×3"), (10, "2×5"), (12, "2²×3"), (14, "2×7")]
for i, (m, factored) in enumerate(mixed):
    td = next((t for t in tree_data if t['matula'] == m), None)
    if td:
        x = 2.5 + i * 0.5
        y = 3.5 - i * 0.3
        ax.scatter(x, y, s=600, c='purple', zorder=5, edgecolors='black', linewidth=1.5, alpha=0.7)
        ax.annotate(f"M={m}\n{factored}", (x, y - 0.3), ha='center', fontsize=8, color='purple')

ax.legend(loc='upper right', fontsize=11)
ax.set_xlim(-0.5, 5)
ax.set_ylim(0, 5)
ax.set_xlabel("Depth (sequential composition)")
ax.set_ylabel("Breadth (parallel composition)")
ax.grid(True, alpha=0.2)

# Panel 3: Shared Factor Network
ax = axes[1, 0]
ax.set_title("Panel C: Shared Prime Factor Network\n(Modes connected by common sub-components)", fontsize=12)

# Place nodes in a circle
n_nodes = len(tree_data)
angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
circle_pos = {}
for i, td in enumerate(tree_data):
    circle_pos[td['matula']] = (np.cos(angles[i]) * 3, np.sin(angles[i]) * 3)

# Draw shared factor edges
for td_i, td_j, shared in shared_factor_edges:
    x1, y1 = circle_pos[td_i['matula']]
    x2, y2 = circle_pos[td_j['matula']]
    # Color by which prime is shared
    prime_colors = {2: 'blue', 3: 'green', 5: 'red', 7: 'orange', 11: 'purple'}
    for p in shared:
        color = prime_colors.get(p, 'gray')
        ax.plot([x1, x2], [y1, y2], color=color, alpha=0.4, linewidth=2)

# Draw nodes
for td in tree_data:
    m = td['matula']
    x, y = circle_pos[m]
    color = colors_map.get(td['order'], '#8c564b')
    ax.scatter(x, y, s=600, c=color, zorder=5, edgecolors='black', linewidth=1.5)
    ax.annotate(f"M={m}", (x, y), ha='center', va='center', fontsize=7, fontweight='bold', color='white')

# Legend for prime colors
legend_patches = [
    mpatches.Patch(color='blue', alpha=0.6, label='Share p(1)=2 (perceive)'),
    mpatches.Patch(color='green', alpha=0.6, label='Share p(2)=3 (feel→think)'),
    mpatches.Patch(color='red', alpha=0.6, label='Share p(3)=5 (chain)'),
    mpatches.Patch(color='orange', alpha=0.6, label='Share p(4)=7 (blend)'),
]
ax.legend(handles=legend_patches, loc='lower left', fontsize=9)
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.1)

# Panel 4: The Product Lattice Structure
ax = axes[1, 1]
ax.set_title("Panel D: Product Lattice (Depth × Breadth)\n2-3-5 Structure Emerges", fontsize=12)

# Place modes on a depth × breadth grid
# depth = length of longest chain in factorization
# breadth = number of factors
for td in tree_data:
    m = td['matula']
    if m == 1:
        depth = 0
        breadth = 0
    else:
        factors = td['factors']
        breadth = len(factors)
        # Depth: max prime index in factorization
        depth = max(encoder.sieve.prime_index(p) for p in factors)
    
    # Add jitter for overlapping
    jitter_x = np.random.uniform(-0.15, 0.15)
    jitter_y = np.random.uniform(-0.15, 0.15)
    
    color = colors_map.get(td['order'], '#8c564b')
    size = 400 + td['order'] * 200
    ax.scatter(depth + jitter_x, breadth + jitter_y, s=size, c=color, 
               zorder=5, edgecolors='black', linewidth=1.5, alpha=0.8)
    ax.annotate(f"M={m}", (depth + jitter_x, breadth + jitter_y + 0.2), 
               ha='center', fontsize=8, fontweight='bold')

ax.set_xlabel("Max Depth (highest prime index in factorization)", fontsize=11)
ax.set_ylabel("Breadth (number of prime factors)", fontsize=11)
ax.set_xlim(-0.5, 6)
ax.set_ylim(-0.5, 5)
ax.grid(True, alpha=0.3)

# Add 2-3-5 annotation
ax.axvspan(-0.5, 1.5, alpha=0.05, color='blue', label='Dyad zone')
ax.axvspan(1.5, 3.5, alpha=0.05, color='green', label='Triad zone')
ax.axvspan(3.5, 6, alpha=0.05, color='red', label='Pentad zone')
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/home/ubuntu/echoself/matula_deep_topology.png", dpi=200, bbox_inches='tight')
print("\nDeep topology visualization saved to matula_deep_topology.png")
