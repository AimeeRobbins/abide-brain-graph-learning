from build_fc_graph import FCGraphBuilder
from build_ec_graph import ECGraphBuilder
from abide_data_loader import AbideDataLoader
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

pheno_df = pd.read_csv('../Phenotypic_V1_0b_preprocessed1.csv')
data_dir = '../abide_data/Outputs/cpac/filt_global/rois_cc200'
data_loader = AbideDataLoader(data_dir=data_dir, pheno_df=pheno_df)

# Load a single subject
X_time, y, site, sid, _, _ = data_loader.load_subject('Caltech_0051461_rois_cc200.1D')

print(f"Subject:    {sid}")
print(f"Site:       {site}")
print(f"Diagnosis:  {'ASD' if y == 1 else 'TDC'}")
print(f"X_time shape: {X_time.shape}")   # should be (200, timepoints)

fc_graph = FCGraphBuilder()
adj, stats = fc_graph.build_graph(X_time)
#fc_graph.save_graph(adj)

#ec_graph = ECGraphBuilder()
#adj = ec_graph.get_node_features(X_time)
#ec_graph.save_graph(adj)
#print(stats["num_nodes"])
#print(stats["num_edges"])
#print(stats["density"])
#print(stats["mean_weight"])
#print(stats["std_weight"])

# adjacency matrix
plt.figure(figsize=(8, 8))
plt.imshow(adj, cmap='coolwarm')
cbar = plt.colorbar(label='Correlation')
cbar.set_label('Correlation', fontsize=14)
plt.title("Functional Connectivity Matrix", fontsize=18)
plt.xlabel("ROI", fontsize=16)
plt.ylabel("ROI", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# Plot roi time series
#plt.figure(figsize=(10, 4))
#plt.plot(
#        X_time[0],
#        label=f"ROI {0 + 1}",
#        linewidth=8
#    )
#plt.xlabel("Time point")
#plt.ylabel("BOLD signal")
#plt.title(f"ROI 1 fMRI Time Series")
#plt.tight_layout()

plt.show()