from sklearn.model_selection import train_test_split
import torch
class DataSplitter:
    """
    Contains methods to split the data into training, validation and testing
    Includes random and loso split
    """
    #graphs = torch.load("data/processed/abide_pytorch_geometric_graphs.pt", weights_only=False)

    def create_random_split(self, graphs):
        """Randomly split graphs into training, validation and test sets."""

        # First split: 70% training, 30% temporary
        train_graphs, temp_graphs = train_test_split(
            graphs,
            test_size=0.30,
            random_state=42,
            stratify=[graph.y.item() for graph in graphs]
        )

        # Second split: 15% validation, 15% test overall
        val_graphs, test_graphs = train_test_split(
            temp_graphs,
            test_size=0.50,
            random_state=42,
            stratify=[graph.y.item() for graph in temp_graphs]
        )

        return train_graphs, val_graphs, test_graphs

    def create_loso_split(graphs, held_out_site):
        print()

    #train_graphs, val_graphs, test_graphs = create_random_split(graphs)
        

    #print("Train:", len(train_graphs))
    #print("Validation:", len(val_graphs))
    #print("Test:", len(test_graphs))