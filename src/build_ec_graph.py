import numpy as np
import pandas as pd
import os
#from scipy import stats

class ECGraphBuilder:
    """
    Builder for Effective Connectivity graphs using Pearson correlation and lagged correlation from ROI time series.
    """

    def __init__(self, k=5, sparsify=True, normalise=True, self_loops=True):
        """
        k: top-k edges per ROI
        sparsify: selects top k edges per row of ROI
        normalise: row-wise normalisation
        self_loops: set diagonals to 1
        """
        self.k = k
        self.sparsify = sparsify
        self.normalise = normalise
        self.self_loops = self_loops

    def corr2_coeff(self, A, B, eps=1e-8):
        A_mA = A - A.mean(1)[:, None]
        B_mB = B - B.mean(1)[:, None]
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)
        return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]) + eps)

    def compute_ec(self, roi_time_series):
        """
        Builds effective connectivity graph using Pearson correlation and lagged correlation from ROI time series.
        roi_time_series: ROI time series matrix with shape (n_rois, timepoints) 
        """
        lag = 1
        x1 = roi_time_series[:, :-lag]
        x2 = roi_time_series[:, lag:]
        
        ec = self.corr2_coeff(x1, x2)
        ec = np.nan_to_num(ec, nan=0.0)
        
        # Remove correlation for diagonal
        np.fill_diagonal(ec, 0)

        # Clip numerical noise outside [-1,1]
        ec = np.clip(ec, -1, 1)

        return ec

    def get_node_features(self, x_time):
        """Fast method to obtain just the raw EC graph, no sparsify/normalize/self-loop work."""
        return self.compute_ec(x_time)

    def sparsify_ec(self, ec):
        """
        Sparsifies the graph to keep the top k strongest connections (measures absolute magnitude)
        ec: Effective connectivity graph as an adjacency matrix (n_rois, n_rois) 
        """
        if(self.sparsify):
            n = ec.shape[0]
            sparse = np.zeros_like(ec)

            # top-k per row
            for i in range(n):
                row = ec[i].copy()
                idx = np.argsort(np.abs(row))[-self.k:] # Get indexes of largest values
                sparse[i, idx] = ec[i, idx]

            # ensure symmetry
            # sparse = np.maximum(sparse, sparse.T)

            return sparse
        else:
            return ec

    def add_self_loops(self, adj):
        """
        Makes the diagonals 1
        adj: Effective connectivity graph as an adjacency matrix (n_rois, n_rois) 
        """
        if(self.self_loops):
            np.fill_diagonal(adj, 1.0)
        return adj

    def normalise_ec(self, adj):
        """
        Makes the diagonals 1
        adj: Effective connectivity graph as an adjacency matrix (n_rois, n_rois) 
        """
        if(self.normalise):
            row_sum = np.abs(adj).sum(axis=1)[:, None]
            row_sum[row_sum == 0] = 1
            adj = adj / row_sum
        return adj

    def graph_stats(self, adj):
        n_edges = np.sum(adj != 0)
        density = n_edges / (adj.shape[0] ** 2)
        weights = adj[adj != 0]
        abs_weights = np.abs(weights)

        return {
            "num_nodes": adj.shape[0],
            "num_edges": int(n_edges),
            "density": float(density),
            "mean_weight": float(np.mean(abs_weights)),
            "std_weight": float(np.std(weights))
        }

    def build_graph(self, roi_time_series):
        """
        Function which is called to fully build the ec graph, returning the adjacency matrix
        roi_time_series : np.ndarray, shape (200, timepoints)
            ROI time series matrix
        
        Returns:
        adj : np.ndarray Adjacency matrix with shape (200, 200) 
        stats_dict : dictionary of the graph statistics
            
        """

        ec = self.compute_ec(roi_time_series)
        adj = self.sparsify_ec(ec)
        adj = self.add_self_loops(adj)
        adj = self.normalise_ec(adj)
        stats_dict = self.graph_stats(adj)
        return adj, stats_dict

    def save_graph(self, adj, path="ec_graph.npy"):
        """
        Saves graph as .npy file
        adj: Effective connectivity graph as an adjacency matrix (n_rois, n_rois)
        path: destination path for file to be saved to, default is "ec_graph.npy"
        """
        np.save(path, adj)