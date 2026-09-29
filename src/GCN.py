import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(200, 32, normalize=False)
        self.conv2 = GCNConv(32, 8, normalize=False)

        self.classifier = torch.nn.Linear(8, 2)

    def get_graph_embedding(self, x, edge_index, edge_weight, batch):
        # First GCN layer
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        # Second GCN layer
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Turn node embeddings into graph embeddings
        x = global_mean_pool(x, batch)
        return x

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.get_graph_embedding(x, edge_index, edge_weight, batch)
        # ASD/control prediction
        x = self.classifier(x)
        return x