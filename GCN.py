import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(200, 16, normalize=False)
        self.conv2 = GCNConv(16, 4, normalize=False)

        self.classifier = torch.nn.Linear(4, 2)

    def forward(self, x, edge_index, edge_weight, batch):

        # First GCN layer
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        # Turn node embeddings into graph embeddings
        x = global_mean_pool(x, batch)

        # ASD/control prediction
        x = self.classifier(x)

        return x