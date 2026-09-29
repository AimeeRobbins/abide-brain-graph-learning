import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
)
from splits import DataSplitter

def extract_fc_features(graph):
    """
    Flatten a subject's FC matrix into a single feature vector.
    Assumes graph.x is the (200, 200) FC matrix used as node features
    (each row is one ROI's FC profile) — use the upper triangle to avoid
    duplicated/redundant values (FC is symmetric).
    """
    fc_matrix = graph.x.numpy()   # (200, 200)
    iu = np.triu_indices(fc_matrix.shape[0], k=1)  # upper triangle, excluding diagonal
    return fc_matrix[iu]   # (19900,) flat vector

def build_X_y(graphs):
    X = np.stack([extract_fc_features(g) for g in graphs])
    y = np.array([g.y.item() for g in graphs])
    return X, y

def run_simple_baseline_loso(sites, graphs):
    splitter = DataSplitter()
    all_results = []

    for site in sites:
        train_g, val_g, test_g = splitter.create_loso_split(in_graphs=graphs, held_out_site=site)

        # combine train+val for simple baseline
        X_train, y_train = build_X_y(train_g + val_g)
        X_test, y_test = build_X_y(test_g)

        clf = LogisticRegression(max_iter=1000, C=0.1)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        result = {
            "site": site,
            "test_accuracy": accuracy_score(y_test, preds),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, preds),
            "test_f1": f1_score(y_test, preds, zero_division=0),
            "test_auc": roc_auc_score(y_test, probs),
        }
        all_results.append(result)
        print(f"{site}: AUC={result['test_auc']:.4f}, Acc={result['test_accuracy']:.4f}")

    return all_results


if __name__ == '__main__':
    graphs = torch.load("../graphs/abide_pytorch_geometric_graphs.pt", weights_only=False)
    sites = sorted({g.site for g in graphs})

    results = run_simple_baseline_loso(sites, graphs)
    df = pd.DataFrame(results)
    df.to_csv("../results/simple_baseline_loso.csv", index=False)

    print("\nOverall mean ± std across sites:")
    print(df[["test_accuracy", "test_balanced_accuracy", "test_f1", "test_auc"]].agg(["mean", "std"]))