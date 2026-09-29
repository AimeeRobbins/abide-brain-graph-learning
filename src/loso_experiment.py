import torch
from train_model import Trainer
import pandas as pd
from GCN import GCN
from GCN1 import GCN1

if __name__ == '__main__':
    # Load saved graphs
    graphs = torch.load(
        "../graphs/abide_pytorch_geometric_graphs.pt",
        weights_only=False
    )

    sites = sorted({g.site for g in graphs})
    trainer = Trainer()

    baseline_results = trainer.run_repeated_loso(sites, GCN, "baseline", n_repeats=5)
    identity_results  = trainer.run_repeated_loso(sites, GCN1, "coral_identity", n_repeats=5, use_coral=True, coral_weight=0.2)

    all_results = baseline_results + identity_results
    df = pd.DataFrame(all_results)
    df.to_csv("../results/swap_gcn_loso_results_all.csv", index=False)

    metrics = ["test_accuracy", "test_balanced_accuracy", "test_f1", "test_auc"]

    # find average across sites, within each seed -> one "LOSO score" per seed
    per_seed_loso = df.groupby(["model", "seed"])[metrics].mean().reset_index()
    per_seed_loso.to_csv("../results/1_coral_gcn_loso_per_seed.csv", index=False)

    # find average across seeds, within each seed -> one score per site
    per_site_loso = df.groupby(["model", "site"])[metrics].mean().reset_index()
    per_site_loso.to_csv("../results/1_coral_gcn_loso_per_site.csv", index=False)

    # mean ± std across seeds
    final_summary = per_seed_loso.groupby("model")[metrics].agg(["mean", "std"])
    final_summary.to_csv("../results/1_coral_gcn_loso_final_summary.csv")

    print(final_summary)