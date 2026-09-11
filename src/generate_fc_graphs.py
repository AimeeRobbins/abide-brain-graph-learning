if __name__ == '__main__':
    # Import packages
    import argparse
    import os
    import sys
    from pathlib import Path
    from build_fc_graph import FCGraphBuilder
    from abide_data_loader import AbideDataLoader
    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd

    directory = Path("../abide_data/Outputs/cpac/filt_global/rois_cc200")

    for file in directory.iterdir():
        print(file.name)
        pheno_df = pd.read_csv('../Phenotypic_V1_0b_preprocessed1.csv')
        data_dir = '../abide_data/Outputs/cpac/filt_global/rois_cc200'
        data_loader = AbideDataLoader(data_dir=data_dir, pheno_df=pheno_df)

        # Load a single subject
        X_time, y, site, sid = data_loader.load_subject(file.name)

        print(f"Subject:    {sid}")
        print(f"Site:       {site}")
        print(f"Diagnosis:  {'ASD' if y == 1 else 'TDC'}")
        print(f"X_time shape: {X_time.shape}")   # should be (200, timepoints)

        fc_graph = FCGraphBuilder()
        adj = fc_graph.build_graph(X_time)
        fc_graph.save_graph(adj, f"fc_graphs/{sid}_fc_graph.npy")

        """
        plt.figure(figsize=(8, 8))
        plt.imshow(adj, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.title(f"{sid} Functional Connectivity Matrix")
        plt.xlabel("ROI")
        plt.ylabel("ROI")
        plt.show()
        """
