from sklearn.model_selection import train_test_split
import torch
from collections import Counter

class DataSplitter:
    """
    Contains methods to split the data into training, validation and testing
    Includes random and loso split
    """
    
    def create_random_split(self, graphs):
        """Randomly split graphs into training, validation and test sets."""

        # First split: 70% training, 30% temporary
        train_graphs, remaining = train_test_split(
            graphs,
            test_size=0.30,
            random_state=42,
            stratify=[graph.y.item() for graph in graphs]
        )

        # Second split: 15% validation, 15% test overall
        val_graphs, test_graphs = train_test_split(
            remaining,
            test_size=0.50,
            random_state=42,
            stratify=[graph.y.item() for graph in remaining]
        )

        return train_graphs, val_graphs, test_graphs

    def create_loso_split(self, graphs, held_out_site, val_size=0.2, random_state=42):
        """Split graphs into training, validation and test sets where the test set is an entire site."""
        test_graphs = []
        remaining = []

        for g in graphs:
            if g.site == held_out_site:
                test_graphs.append(g)
            else:
                remaining.append(g)

        if not test_graphs:
            raise ValueError(f"No graphs found for site '{held_out_site}'")

        # Split the remaining graphs into training and validation
        train_graphs, val_graphs = train_test_split(
            remaining,
            test_size=val_size,
            random_state=random_state,
            stratify=[graph.y.item() for graph in remaining]
        )

        return train_graphs, val_graphs, test_graphs

if __name__ == '__main__':
    # Load saved graphs
    graphs = torch.load(
        "../graphs/abide_pytorch_geometric_graphs.pt",
        weights_only=False
    )

    splitter = DataSplitter()

    for site in sorted({g.site for g in graphs}):
        train, val, test = splitter.create_loso_split(graphs, site)
        print(site, len(train), len(val), len(test))
        print("Test labels:", Counter(g.y.item() for g in test))
        print("Train labels:", Counter(g.y.item() for g in train))
        print()