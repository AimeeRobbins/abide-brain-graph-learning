from build_fc_graph import FCGraphBuilder
from abide_data_loader import AbideDataLoader
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

pheno_df = pd.read_csv('Phenotypic_V1_0b_preprocessed1.csv')
data_dir = './abide_data/Outputs/cpac/filt_global/rois_cc200'
data_loader = AbideDataLoader(pheno_df=pheno_df, data_dir=data_dir)

# Load a single subject
X_time, y, site, sid = data_loader.load_subject('./abide_data/Outputs/cpac/filt_global/rois_cc200/Pitt_0050030')

print(f"Subject:    {sid}")
print(f"Site:       {site}")
print(f"Diagnosis:  {'ASD' if y == 1 else 'TDC'}")
print(f"X_time shape: {X_time.shape}")   # should be (200, timepoints)

fc_graph = FCGraphBuilder()
adj = fc_graph.build_graph(X_time)
fc_graph.save_graph(adj)

plt.figure(figsize=(8, 8))
plt.imshow(adj, cmap='coolwarm')
plt.colorbar(label='Correlation')
plt.title("Functional Connectivity Matrix")
plt.xlabel("ROI")
plt.ylabel("ROI")

plt.show()