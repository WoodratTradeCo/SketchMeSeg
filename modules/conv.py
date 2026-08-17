import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter

# from .layers import GATLayer, PoolLayer
from .basic import MLPLinear
from .dynamic import DilatedKnnGraph
from .tg_conv import MRConv, EdgeConv2, SAGEConv2

class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, opt):
        super(GraphConv, self).__init__()

        self.gconv = tg.nn.EdgeConv(
            nn=MLPLinear(channels=[in_channels*2, out_channels], act_type='relu', norm_type='None'),
            aggr='max'
        )

    def forward(self, x, edge_index, data=None):
        """
        x: (BxN) x F
        """

        return self.gconv(x, edge_index)

class DynConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, dilation, opt, knn_type='matrix'):
        super(DynConv, self).__init__(in_channels, out_channels, opt)
        self.k = 8
        self.d = dilation
        self.dilated_knn_graph = DilatedKnnGraph(8, dilation, True, 0.1, knn_type=knn_type)
        self.mixedge = True

    def forward(self, x, edge_index, data=None):
        """
        x: (BxN) x F
        """
        dyn_edge_index = self.dilated_knn_graph(x, data['batch'])
        if self.mixedge:
            dyn_edge_index = torch.unique(torch.cat([edge_index, dyn_edge_index], dim=1), dim=1)
        
        # TODO: calculate edge_attr use pos
        dyn_edge_attr = None

        return super(DynConv, self).forward(x, dyn_edge_index, {'edge_attr':dyn_edge_attr})

