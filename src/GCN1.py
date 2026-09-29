import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN1(torch.nn.Module):
    """
    Identity-aware GCN: adds a learned ROI embedding to each node's FC-row feature vector before the first conv layer
    """
    def __init__(self, num_rois=200, roi_embed_dim=16):
        super().__init__()

        self.num_rois = num_rois
        self.roi_embedding = torch.nn.Embedding(num_rois, roi_embed_dim)

        self.conv1 = GCNConv(200 + roi_embed_dim, 32, normalize=False)
        self.conv2 = GCNConv(32, 8, normalize=False)
        self.classifier = torch.nn.Linear(8, 2)

    def get_graph_embedding(self, x, edge_index, edge_weight, batch):
        roi_ids = torch.arange(x.size(0), device=x.device) % self.num_rois
        roi_embeds = self.roi_embedding(roi_ids)
        x = torch.cat([x, roi_embeds], dim=-1)

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