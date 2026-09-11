"""
Script to test the output of the graphs 
"""
import torch
import numpy as np
from collections import Counter

graphs = torch.load("data/processed/abide_pytorch_geometric_graphs.pt", weights_only=False)

print(f"Total graphs: {len(graphs)}\n")

# Look at one example in detail
g = graphs[0]
print("Example graph (index 0):")
print(f"  number of nodes:    {g.num_nodes}")
print(f"  number of edges:    {g.num_edges}")
print(f"  number of nodes features: {g.num_features}")
print(f"  isolated nodes:     {g.has_isolated_nodes()}")
print(f"  self loops:         {g.has_self_loops()}")
print(f"  is directed:        {g.is_directed()}")
print(f"  x (node features):  {g.x.shape}")
print(f"  edge_index:         {g.edge_index.shape}")
print(f"  edge_weight:        {g.edge_weight.shape}")
print(f"  y (label):          {g.y.item()}")
print(f"  site:               {g.site}")
print()

# Check consistency across the whole dataset
node_counts = [g.x.shape[0] for g in graphs]
feature_dims = [g.x.shape[1] for g in graphs]
edge_counts = [g.edge_index.shape[1] for g in graphs]

print(f"Node counts:    min={min(node_counts)}, max={max(node_counts)} (should all be 200)")
print(f"Feature dims:   min={min(feature_dims)}, max={max(feature_dims)} (should all be 200)")
print(f"Edge counts:    min={min(edge_counts)}, max={max(edge_counts)}, mean={np.mean(edge_counts):.1f}")
print()

# Label balance
labels = [g.y.item() for g in graphs]
print(f"Label distribution: {Counter(labels)}  (1=ASD, 0=TDC)")
print()

# Site distribution
sites = [g.site for g in graphs]
print("Site distribution:")
for site, count in sorted(Counter(sites).items()):
    print(f"  {site:12s}: {count}")

# Check for any NaN/Inf
for i, g in enumerate(graphs):
    if torch.isnan(g.x).any() or torch.isinf(g.x).any():
        print(f"WARNING: graph {i} (site={g.site}) has NaN/Inf in x")
    if torch.isnan(g.edge_weight).any() or torch.isinf(g.edge_weight).any():
        print(f"WARNING: graph {i} (site={g.site}) has NaN/Inf in edge_weight")