import pandas as pd
import numpy as np
import math
import tqdm
import csv
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import random
import igraph as ig
from scipy import sparse
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder
from copy import deepcopy as cp


class GINConv(MessagePassing):
    def __init__(self, input_dim, emb_dim):
        super(GINConv, self).__init__(aggr='add')
        self.input_dim=input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * emb_dim),
            nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim))
        self.epsilon = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index):
        h = self.mlp((1 + self.epsilon) * x + self.propagate(edge_index, x=x))
        return h

    def message(self, x_j):
        return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out

    ### GCN convolution along the graph structure


class GCNConv(MessagePassing):
    def __init__(self, input_dim, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = nn.Linear(input_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index):
        # x = self.linear(x)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(col, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm) + F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


class gnn_model(nn.Module):
    def __init__(self, input_dim, emb_dim, num_layers, out_dim=33, pool='add', gnn_type='gin', device = None, K=50):
        super(gnn_model, self).__init__()
        self.device = device
        self.W = nn.Parameter(torch.randn(3000)) #trainable weight add as parameter       
        self.K = K
        self.out_indim = K * (emb_dim // K)
        self.emb_layer = nn.Linear(input_dim, emb_dim)
        #self.emb_layer = nn.Linear(1, emb_dim)
        self.num_layers = num_layers
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.convs = nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        # if gnn_type == 'gin':
        #    self.convs.append(GINConv(input_dim, emb_dim))
        # elif gnn_type == 'gcn':
        #    self.convs.append(GCNConv(input_dim, emb_dim))
        # else:
        #    ValueError('Undefined GNN type called {}'.format(gnn_type))
        # self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        for i in range(self.num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(input_dim, emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'add':
            self.pool_layer = global_add_pool
        elif pool == 'mean':
            self.pool_layer = global_mean_pool
        else:
            ValueError('Undefined GNN type called {}'.format(pool))
        
        self.pred_layer = nn.Linear(self.out_indim, out_dim)

    def forward(self, x, edge_index, batch):
        # print(x.size())
        # new_gt = torch.gt(batch, 2998)
        # print(new_gt)
        # print(new_gt.any())
        # printt

        _, topk_indices = self.W.topk(self.K)

        h = self.emb_layer(x)

        h = torch.mul(h, self.W.to(self.device))

        h_list = [h]
        

        # print(x.size())
        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, 0.25, training=self.training)
            else:
                h = F.dropout(F.relu(h), 0.25, training=self.training)
            h_list.append(h)

        return self.pred_layer(self.pool_layer(h_list[-1], batch)), topk_indices

