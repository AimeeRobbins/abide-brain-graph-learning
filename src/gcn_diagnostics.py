import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader

from splits import DataSplitter
from GCN import GCN  
from GCN1 import GCN1 

def get_test_embeddings(model, test_loader):
    """Run the model on test data, return (embeddings, labels) as numpy arrays."""
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            emb = model.get_graph_embedding(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            embeddings.append(emb.cpu().numpy())
            labels.extend(batch.y.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)
    return embeddings, labels


def plot_tsne(embeddings, labels, tag, save_dir="../results"):
    """Reduce embeddings to 2D via t-SNE and plot, colored by class."""
    n_samples = embeddings.shape[0]

    if n_samples <= 5:
        print(f"[{tag}] Too few test samples ({n_samples}) for t-SNE — skipping plot.")
        return

    # t-SNE needs perplexity < n_samples; guard against tiny test sets
    perplexity = min(30, max(5, n_samples // 3))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=["TD", "ASD"])
    plt.title(f"GCN embeddings (t-SNE): {tag}")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/tsne_{tag}.png")
    plt.close()
    print(f"Saved: {save_dir}/tsne_{tag}.png")


def check_embedding_variance(embeddings, tag):
    """Quick numeric check: how spread out are embeddings across different subjects?"""
    per_dim_var = embeddings.var(axis=0)
    print(f"[{tag}] Mean per-dimension variance across test subjects: {per_dim_var.mean():.6f}")

    corr_matrix = np.corrcoef(embeddings)
    off_diag = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
    print(f"[{tag}] Mean pairwise correlation between subjects' embeddings: {off_diag.mean():.4f}")


if __name__ == '__main__':
    held_out_site = "CALTECH"       # pick whichever site you want to inspect
    model_tag = "coral_seed0"     # must match the tag used when you trained/saved this checkpoint

    # --- rebuild the same split used during training ---
    graphs = torch.load("../graphs/abide_pytorch_geometric_graphs.pt", weights_only=False)
    splitter = DataSplitter()
    train_g, val_g, test_g = splitter.create_loso_split(in_graphs=graphs, held_out_site=held_out_site)
    test_loader = DataLoader(test_g, batch_size=8, shuffle=False)

    # --- load the trained model ---
    model = GCN()
    tag = f"{held_out_site}_{model_tag}"
    model.load_state_dict(torch.load(f"../models/best_gcn_{tag}.pt"))

    # --- run diagnostics ---
    embeddings, labels = get_test_embeddings(model, test_loader)
    plot_tsne(embeddings, labels, tag)
    check_embedding_variance(embeddings, tag)