import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.nn import global_max_pool

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # 4. Global Pooling.
        x = global_max_pool(x, data.batch)

        return F.log_softmax(x, dim=1)

