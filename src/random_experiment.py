import torch
from train_model import Trainer
import pandas as pd
from GCN import GCN
from GCN1 import GCN1

if __name__ == '__main__':
    trainer = Trainer()
    baseline_results = []
    identity_results = []

    for i in range(5):
        baseline_result = trainer.run(held_out_site=None, model_class=GCN, model_tag=f"gcn_{i}")
        #identity_result = trainer.run(held_out_site=None, model_class=GCN1, model_tag=f"identity_{i}")

        baseline_results.append(baseline_result)
        #identity_results.append(identity_result)

    all_results = baseline_results #+ identity_results
    df = pd.DataFrame(all_results)
    df.to_csv("../results/swap_gcn_random_results_all.csv", index=False)