import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch import Tensor

from torch_geometric.nn import global_max_pool, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, num_node_features , num_classes , hidden_channels):
        super(GCN, self).__init__()
        
        self.conv1 = GCNConv(num_node_features, hidden_channels) # dataset.num_node_features
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes) # dataset.num_classes 

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

import torch
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv
from layers import PointGNNConv
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MLP


class PointNet(torch.nn.Module):
    def __init__(self, 
                num_node_features=3, 
                num_classes=40, 
                MLP_h: list = [3, 64, 3],
                MLP_f: list = [6, 64, 3],
                MLP_g: list = [3, 64, 3]
        ):
        super().__init__()

        self.num_node_features = num_node_features
        self.num_classes = num_classes

        self.conv1 = PointNetConv(mlp_h=MLP(MLP_h), mlp_f=MLP(MLP_f), mlp_g=MLP(MLP_g), aggr="max")
        self.conv2 = PointNetConv(mlp_h=MLP(MLP_h), mlp_f=MLP(MLP_f), mlp_g=MLP(MLP_g), aggr="max")
        self.conv3 = PointNetConv(mlp_h=MLP(MLP_h), mlp_f=MLP(MLP_f), mlp_g=MLP(MLP_g), aggr="max")

        self.linear = nn.Linear(num_node_features, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        pos = x

        x = self.conv1(x, pos, edge_index)
        x = F.relu(x)

        x = F.dropout(x, training=self.training)
        x = self.conv2(x, pos, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, pos, edge_index)

        x = global_mean_pool(x, data.batch)

        x = self.linear(x)

        return F.log_softmax(x, dim=1)
