from sklearn.model_selection import train_test_split
import torch
from collections import Counter
from train_model import Trainer
import pandas as pd

if __name__ == '__main__':
    # Load saved graphs
    graphs = torch.load(
        "../graphs/abide_pytorch_geometric_graphs.pt",
        weights_only=False
    )

    all_results = []
    trainer = Trainer()

    for site in sorted({g.site for g in graphs}):
        result = trainer.run(held_out_site=site)
        all_results.append(result)

    pd.DataFrame(all_results).to_csv("../results/gcn_loso_results.csv", index=False)