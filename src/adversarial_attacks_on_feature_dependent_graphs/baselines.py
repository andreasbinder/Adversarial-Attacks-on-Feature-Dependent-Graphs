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

from typing import List, Optional, Union

import torch

from torch_geometric.nn import MLP
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.typing import Adj

class PointGNNConv(MessagePassing):
    r"""The PointGNN operator from the `"Point-GNN: Graph Neural Network for
    3D Object Detection in a Point Cloud" <https://arxiv.org/abs/2003.01251>`_
    paper, where the graph is statically constructed using radius-based cutoff.
    The relative position is used in the message passing step to introduce
    global translation invariance.
    To also counter shifts in the local neighborhood of the center vertex,
    the authors propose an alignment offset.

    Args:
        state_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        MLP_h (MLP): Calculate alignment off-set  :math:`\Delta` x
        MLP_f (MLP): Calculate edge update using relative coordinates,
            alignment off-set and neighbor feature.
        MLP_g (MLP): Transforms aggregated messages.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          node coordinates :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, state_channels: int, MLP_h: MLP, MLP_f: MLP, MLP_g: MLP,
                 aggr: Optional[Union[str, List[str],
                                      Aggregation]] = "max", **kwargs):
        # kwargs.setdefault('aggr', 'max')
        self.state_channels = state_channels

        super().__init__(aggr, **kwargs)

        self.lin_h = MLP_h
        self.lin_f = MLP_f
        self.lin_g = MLP_g

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_h.reset_parameters()
        self.lin_f.reset_parameters()
        self.lin_g.reset_parameters()

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj) -> Tensor:
        # propagate_type: (x: Tensor, pos: Tensor)
        out = self.propagate(edge_index, x=x, pos=pos)
        out = self.lin_g(out)
        return x + out

    def message(self, pos_j: Tensor, pos_i: Tensor, x_i: Tensor, x_j: Tensor) -> Tensor:
        delta = self.lin_h(x_i)
        e = torch.cat([pos_j - pos_i + delta, x_j], dim=-1)
        e = self.lin_f(e)
        return e

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.state_channels}, '
                f'lin_h={self.lin_h}, lin_f={self.lin_f},'
                f'lin_g={self.lin_g})')

import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import PointGNNConv, MLP
from torch_geometric.testing import is_full_test

def test_pointgnn_conv():
    x = torch.randn(6, 3)
    edge_index = torch.tensor([[0, 1, 1, 1, 2, 5], [1, 2, 3, 4, 3, 4]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(6, 6))
    MLP_h = MLP([3, 64, 3])
    MLP_f = MLP([6, 64, 3])
    MLP_g = MLP([3, 64, 3])

    conv = PointGNNConv(state_channels=3, MLP_h=MLP_h, MLP_f=MLP_f, MLP_g=MLP_g)
    assert conv.__repr__() == 'PointGNNConv(3, lin_h=MLP(3, 64, 3),' \
                        ' lin_f=MLP(6, 64, 3),lin_g=MLP(3, 64, 3))'
    out = conv(x, x, edge_index)

    assert out.size() == (6, 3)
    assert torch.allclose(conv(x, x, adj.t()), out, atol=1e-6)

    if is_full_test:
        t = '(Tensor, Tensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, x, edge_index).tolist() == out.tolist()

        t = '(Tensor, Tensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, x, adj.t()), out, atol=1e-5)
