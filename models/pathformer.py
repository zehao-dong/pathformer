import torch
import torch.nn as nn
from torch import optim
import random
import math
import igraph as ig
from scipy import sparse
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder
from copy import deepcopy as cp



class MEGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,  Nnodes=3676):
        super(MEGNN, self).__init__()
        self.Nnodes = Nnodes
        self.lin_layer = nn.Linear(in_dim, out_dim)
        self.out_mlp = nn.Sequential(nn.Linear(out_dim * Nnodes, hidden_dim),
                                        nn.Dropout(p=0.5),
                                        nn.ReLU(),
                                 nn.Linear(hidden_dim, out_dim* Nnodes))
        self.batchnorm_layer =nn.BatchNorm1d(hidden_dim)
    def forward(self, x):
        # input shape: B * N * d_k
        b_size = x.size(0)
        x = self.lin_layer(x)
        x = x.reshape(b_size,-1)
        for i in range(1):
            x = self.out_mlp[3 * i](x)
            x = self.batchnorm_layer(x)
            x = self.out_mlp[3 * i + 1](x)
            x = self.out_mlp[3 * i + 2](x)
        x = self.out_mlp[-1](x)
        x = x.reshape(b_size, self.Nnodes, -1)
        return x


def clones(module, N):
    return nn.ModuleList([cp(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    # W is the pair-wise pathway attention
    d_k = query.size(-1)
    n_head = query.size(1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print(query.shape)
    # print(scores.shape)
    # print(d_k)
    scores = scores
    if mask is not None:
        mask = torch.stack([mask] * n_head, dim=1)
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value), attn


class pathAttention(nn.Module):
    def __init__(self, d_model, n_head, drop_out, hidden_dim, Nnodes=3676):
        # d_model is the out_dim1 and out_dim2
        # hidden_dim is the hidden_dim1 and hidden_dim2
        super(pathAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.gnn_layers = clones(MEGNN(d_model, hidden_dim, d_model, Nnodes), 2)
        self.linears = clones(nn.Linear(d_model, self.d_model), 2)
        self.dropout = nn.Dropout(p=drop_out)
        # self.W = W

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        query, key = [l(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2) for l, x in
                      zip(self.gnn_layers, (query, key))]
        value = self.linears[0](value).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        return self.linears[1](x), attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_out):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_out, hidden_dim, Nnodes=3676):
        super(SelfAttentionBlock, self).__init__()

        self.norm = nn.LayerNorm(d_model)
        self.attn = pathAttention(d_model, n_head, drop_out, hidden_dim, Nnodes)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x, mask):
        x_ = self.norm(x)
        x_, attn = self.attn(x_, x_, x_, mask)
        return self.dropout(x_) + x, attn


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, drop_out):
        super(FeedForwardBlock, self).__init__()

        self.norm = nn.LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, drop_out)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x_ = self.norm(x)
        x_ = self.feed_forward(x_)
        return self.dropout(x_) + x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, drop_out, hidden_dim, Nnodes=3676):
        super(EncoderBlock, self).__init__()
        self.self_attn = SelfAttentionBlock(d_model, n_head, drop_out, hidden_dim, Nnodes)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, drop_out)

    def forward(self, x, mask):
        x, attn = self.self_attn(x, mask)
        x = self.feed_forward(x)
        return x, attn


class Encoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, Nnodes):
        super(Encoder, self).__init__()

        self.layers = clones(EncoderBlock(d_model, n_head, d_ff, drop_out, hidden_dim, Nnodes), n_layers)
        self.norms = clones(nn.LayerNorm(d_model), n_layers)

    def forward(self, x, mask):
        outputs = []
        attns = []
        for layer, norm in zip(self.layers, self.norms):
            x, attn = layer(x, mask)
            x = norm(x)
            outputs.append(x)
            attns.append(attn)
        return outputs[-1], outputs, attns


class GraphEncoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, Nnodes):
        super(GraphEncoder, self).__init__()
        # Forward Transformers
        self.encoder_f = Encoder(d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, Nnodes)

    def forward(self, x, mask):
        h_f, hs_f, attns_f = self.encoder_f(x, mask)
        return h_f, hs_f, attns_f

    @staticmethod
    def get_embeddings(h_x):
        h_x = h_x.cpu()
        return h_x[0]


class copynumberEmbedding(nn.Module):
    def __init__(self, n_vocab, d_model):
        super(copynumberEmbedding, self).__init__()
        self.w2e = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.w2e(x) * math.sqrt(self.d_model)


class geneexpEmbedding(nn.Module):
    def __init__(self, n_vocab, d_model):
        super(geneexpEmbedding, self).__init__()
        self.w2e = nn.Linear(1, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.w2e(x) * math.sqrt(self.d_model)


class nodedegEmbedding(nn.Module):
    def __init__(self, n_vocab, d_model):
        super(nodedegEmbedding, self).__init__()
        self.w2e = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.w2e(x) * math.sqrt(self.d_model)


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


class Pathormer(nn.Module):
    def __init__(self, n_copy, copy_dim, d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, out_hidden,
                 Nnodes=3686, out_dim=33, device=None):
        super(Pathormer, self).__init__()
        #self.in_dim = 1 + copy_dim + Nnodes
        self.in_dim = 1 + Nnodes
        self.d_model = d_model
        #self.geneEmb = geneexpEmbedding(n_vocab, d_model)
        self.copyEmb = copynumberEmbedding(n_copy, copy_dim)
        self.translayer = nn.Linear(self.in_dim, self.d_model)
        self.dropout_op = nn.Dropout(p=drop_out)        
        self.graph_encoder = GraphEncoder(d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, Nnodes)
        out_indim = Nnodes * d_model
        self.readout_layer = readout(out_indim, out_hidden, out_dim)

    #def forward(self, Xg, Xc, adj, mask=None):
    def forward(self, Xg, adj, mask=None):
        emb_g = Xg
        #emb_c = self.dropout_op(self.copyEmb(Xc))
        #emb = torch.cat([emb_g, emb_c, adj], dim=-1)
        emb = torch.cat([emb_g, adj], dim=-1)
        emb = self.dropout_op(self.translayer(emb))
        
        b_size = emb.size(0)
        h, _, attn_list = self.graph_encoder(emb,mask)
        h = h.reshape(b_size, -1)
        out = self.readout_layer(h)
        return out, attn_list
