
import numpy as np
import torch
from abide_data_loader import AbideDataLoader
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
from build_ec_graph import ECGraphBuilder
from build_fc_graph import FCGraphBuilder

ec_builder = ECGraphBuilder()
fc_builder = FCGraphBuilder()

def build_subject_graph(roi_time_series, label, site):
    """Returns a Object containg a tensor data object (edge_index, edge_weight, x, y) and site"""
    # Obtain the adjacency matrix EC Graph and convert to tensor
    ec, ec_stats = ec_builder.build_graph(roi_time_series)
    A = torch.tensor(ec, dtype=torch.float)

    # Represent A as edge_index and edge_weight
    edge_index = torch.nonzero(A, as_tuple=False).t()
    edge_weight = A[edge_index[0], edge_index[1]]

    # Obtain the raw FC to use as the node features
    fc_raw = fc_builder.get_node_features(roi_time_series)
    x = torch.tensor(fc_raw, dtype=torch.float)

    # diagnosis label, y
    y = torch.tensor([label], dtype=torch.long)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y
    )

    data.site = site

    return data

def build_subject_graph_swapped(roi_time_series, label, site):
    """Returns a Object containg a tensor data object (edge_index, edge_weight, x, y) and site"""
    # Obtain the adjacency matrix FC Graph and convert to tensor
    fc, fc_stats = fc_builder.build_graph(roi_time_series)
    A = torch.tensor(fc, dtype=torch.float)

    # Represent A as edge_index and edge_weight
    edge_index = torch.nonzero(A, as_tuple=False).t()
    edge_weight = A[edge_index[0], edge_index[1]]

    # Obtain the raw EC to use as the node features
    ec_raw = ec_builder.get_node_features(roi_time_series)
    x = torch.tensor(ec_raw, dtype=torch.float)

    # diagnosis label, y
    y = torch.tensor([label], dtype=torch.long)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y
    )

    data.site = site

    return data

# main program to create graphs
graphs = []

pheno_df = pd.read_csv('../Phenotypic_V1_0b_preprocessed1.csv')
data_dir = '../abide_data/Outputs/cpac/filt_global/rois_cc200'
data_loader = AbideDataLoader(data_dir=data_dir, pheno_df=pheno_df)

# Load several subjects
for path in os.listdir(data_dir):
    timeseries, label, site, sid, _, _ = data_loader.load_subject(path)

    data = build_subject_graph_swapped(timeseries, label, site)
    graphs.append(data)

# Save
torch.save(graphs, "../graphs/abide_pytorch_geometric_graphs.pt")


