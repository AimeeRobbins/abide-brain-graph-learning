import numpy as np
import pandas as pd
import os
#from scipy import stats

class FCGraphBuilder:
    """
    Builder for Functional Connectivity graphs using Pearson correlation from ROI time series.
    """

    def __init__(self, k=5, sparsify=True, normalise=True, self_loops=True):
        """
        k: top-k edges per ROI
        normalise: row-wise normalisation
        self_loops: set diagonals to 1
        """
        self.k = k
        self.sparsify = sparsify
        self.normalise = normalise
        self.self_loops = self_loops

    def compute_fc(self, x_time):
        """
        Builds functional connectivity graphs using Pearson correlation from ROI time series
        x_time: shape (n_rois, timepoints)
            ROI time series matrix
        """
        # Use pearson correlation for fc graph
        fc = np.corrcoef(x_time)
        # Clip numerical noise outside [-1,1]
        fc = np.clip(fc, -1, 1)
        return fc

    def sparsify_fc(self, fc):
        if(self.sparsify):
            n = fc.shape[0]
            sparse = np.zeros_like(fc)

            # top-k per row
            for i in range(n):
                row = fc[i].copy()
                idx = np.argsort(np.abs(row))[-self.k:] # Get indexes of largest values
                sparse[i, idx] = fc[i, idx]

            # ensure symmetry
            sparse = np.maximum(sparse, sparse.T)

            return sparse
        else:
            return fc

    def add_self_loops(self, adj):
        if(self.add_self_loops):
            np.fill_diagonal(adj, 1.0)
        return adj

    def normalise_fc(self, adj):
        if(self.normalise):
            row_sum = np.abs(adj).sum(axis=1)[:, None]
            row_sum[row_sum == 0] = 1
            adj = adj / row_sum
        return adj

    def build_graph(self, x_time):
        """
        Parameters
        ----------
        X_time : np.ndarray, shape (200, timepoints)
            ROI time series matrix
        
        Returns
        -------
        A : np.ndarray, shape (200, 200)
            Adjacency matrix
        F : np.ndarray, shape (200, 200)
            Correlation matrix (full, before sparsification)
        stats_dict : dict
            
        """
        # Compute Pearson Correlation Matrix with shape (200,200)
        fc = self.compute_fc(x_time)
        adj = self.sparsify_fc(fc)
        adj = self.add_self_loops(adj)
        adj = self.normalise_fc(adj)
        return adj
       
    def graph_stats(self, adj):
        n_edges = np.sum(adj != 0)
        density = n_edges / (adj.shape[0] ** 2)

        return {
            "num_nodes": adj.shape[0],
            "num_edges": int(n_edges),
            "density": float(density),
            "mean_weight": float(np.mean(adj)),
            "std_weight": float(np.std(adj))
        }

    def save_graph(self, adj, path="fc_graph.npy"):
        np.save(path, adj)

        