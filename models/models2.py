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
from copy import deepcopy as cp

class readout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(readout, self).__init__()
        self.mlp_layers = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                        nn.Dropout(p=0.5),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, out_dim))
        self.batchnorm_layer = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        for i in range(1):
            x = self.mlp_layers[3 * i](x)
            x = self.batchnorm_layer(x)
            x = self.mlp_layers[3 * i + 1](x)
            x = self.mlp_layers[3 * i + 2](x)
        x = self.mlp_layers[-1](x)
        return x

class GCN_conv(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GCN_conv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, adj, x, b_size, graph_size):
        device = x.device
        out = torch.zeros(x.size(0), self.output_dim).to(device)
        x = torch.mm(x, self.weight)
        #print(x.shape)
        deg = 1 + torch.sum(adj, dim=1)
        #print(deg.shape)
        accum = 0
        for i in range(b_size):
            x_i = x[accum:accum+graph_size]
            #print(x_i.shape)
            x_i = torch.mm(adj, x_i) + x_i
            x_i = torch.div(x_i, deg.reshape(-1,1))
            out[accum:accum+graph_size] = x_i 
            accum += graph_size
        if self.use_bias:
            out += self.bias
        return out


class GIN_conv(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_hidden, train_eps=True):
        super(GIN_conv, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, mlp_hidden),
                                nn.Dropout(p=0.5),
                                nn.ReLU(),
                                nn.Linear(mlp_hidden, output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_eps = train_eps
        if self.train_eps:
            self.eps = nn.Parameter(torch.Tensor(1))
        else:
            self.eps = 0
        self.reset_parameters()

    def reset_parameters(self):
        if self.train_eps:
            nn.init.zeros_(self.eps)

    def forward(self, adj, x, b_size, graph_size):
        device = x.device
        out = torch.zeros(x.size(0), self.output_dim).to(device)
        #x = torch.mm(x, self.weight)
        accum = 0
        for i in range(b_size):
            x_i = x[accum:accum+graph_size]
            x_i = (1+self.eps)*x_i + torch.sparse.mm(adj, x_i)
            out[accum:accum+graph_size] = x_i 
            accum += graph_size
        out = self.mlp(out)
        return out



class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layer=3, mlp_dim=32, out_hidden=128, Nnodes=3686, dropout=0.5, gnn_type='gcn', use_cl=True, train_eps=True, readout_in=True):
        super(GNN, self).__init__()
        self.gnn_type = gnn_type
        self.use_cl = use_cl
        self.num_gene = Nnodes
        self.num_layer = num_layer
        self.input_dim = input_dim
        self.dropout = dropout
        self.readout_in=readout_in
        
        if self.use_cl:
            self.emb_layer = nn.Embedding(Nnodes, input_dim)
            self.map_layer = nn.Linear(1+Nnodes, input_dim)
            #self.map_layer = nn.Linear(1, input_dim)
        else:
            self.map_layer = nn.Linear(1, input_dim)

        self.conv_layers  = torch.nn.ModuleList()
        if gnn_type == 'gcn':
            self.conv_layers.append(GCN_conv(input_dim, hidden_dim)) 
            for i in range(self.num_layer-1):
                self.conv_layers.append(GCN_conv(hidden_dim, hidden_dim)) 
        else:
            self.conv_layers.append(GIN_conv(input_dim=input_dim, output_dim=hidden_dim, mlp_hidden=mlp_dim, train_eps=train_eps)) 
            for i in range(self.num_layer-1):
                self.conv_layers.append(GIN_conv(input_dim=hidden_dim, output_dim=hidden_dim, mlp_hidden=mlp_dim, train_eps=train_eps))

        out_indim = Nnodes * hidden_dim
        if self.readout_in:
            self.readout_layer = readout(out_indim, out_hidden, out_dim)
        else:
            self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj, batch_size):
        device = x.device
        if self.use_cl:
            A = torch.cat([adj] * batch_size, dim=0)
            emb = self.map_layer(torch.cat([x,A],dim=-1))
            #emb = self.map_layer(x)
            #emb = F.dropout(emb, self.dropout, training=self.training)
            #cl = torch.LongTensor(np.arange(self.num_gene)).to(device)
            #cl_emb_i = self.emb_layer(cl)
            #cl_emb = torch.zeros(x.size(0), self.input_dim).to(device)
            #accum = 0
            #for i in range(batch_size):
            #    cl_emb[accum:accum+self.num_gene] = cl_emb_i
            #    accum += self.num_gene
            #emb += cl_emb

        else:
            emb = self.map_layer(x)
        for l in range(self.num_layer):
            emb = self.conv_layers[l](adj, emb, batch_size, self.num_gene)
            emb = F.dropout(emb, self.dropout, training=self.training)
        if self.readout_in:
            h = emb.reshape(batch_size, -1)
            out = self.readout_layer(h)
        else:
            h = emb.reshape(batch_size, self.num_gene, -1)
            out = torch.sum(h, dim=1)
            out = self.lin(out)
        return out















