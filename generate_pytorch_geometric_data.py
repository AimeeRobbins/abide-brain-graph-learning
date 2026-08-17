
import numpy as np
import torch
from abide_data_loader import AbideDataLoader
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os

def process_subject(path, timeseries, label):
    """Returns a data object containing edge_index, edge_weight, x, y"""
    # Obtain the adjacency matrix A 
    A = np.load(path)
    # Check the shape of the adjacency matric and time series graph
    assert A.shape[0] == timeseries.shape[0], \
        f"Mismatch: A has {A.shape[0]} nodes, timeseries has {timeseries.shape[0]}"
    assert A.shape[0] == A.shape[1] , \
        f"Mismatch: Rows of A has {A.shape[0]} nodes, columns has {A.shape[1]}"
    A_tensor = torch.tensor(A, dtype=torch.float)

    # Represent A as edge_index and edge_weight
    edge_index = torch.nonzero(A_tensor, as_tuple=False).t()
    edge_weight = A_tensor[edge_index[0], edge_index[1]]
    
    # Node features, x (mean and standard deviation)
    mean = timeseries.mean(axis=1)
    std = timeseries.std(axis=1)
    node_features = np.column_stack([mean, std])
    x = torch.tensor(node_features, dtype=torch.float)

    # diagnosis label, y
    y = torch.tensor([label], dtype=torch.long)
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y
    )

graphs = []

pheno_df = pd.read_csv('Phenotypic_V1_0b_preprocessed1.csv')
data_dir = './abide_data/Outputs/cpac/filt_global/rois_cc200'
data_loader = AbideDataLoader(data_dir=data_dir, pheno_df=pheno_df)

# Load several subjects
for path in os.listdir(data_dir)[:12]:
    timeseries, label, site, sid = data_loader.load_subject(path)

    #print(f"Subject:    {sid}")
    #print(f"Site:       {site}")
    #print(f"Diagnosis:  {'ASD' if label == 1 else 'TDC'}")
    #print(f"X_time shape: {timeseries.shape}")   # should be (200, timepoints)    

    ec_graph_path = f"ec_graphs/{sid}_ec_graph.npy"
    if os.path.exists(ec_graph_path):
        data = process_subject(ec_graph_path, timeseries, label)
        graphs.append(data)
    else:
        print(f"Missing EC graph: {ec_graph_path}")

#print(graphs)

loader = DataLoader(
    graphs,
    batch_size=4,
    shuffle=True
)

for batch in loader:
    print(batch)


