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

from torch.nn import init

class MEGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,  Nnodes=3676):
        super(MEGNN, self).__init__()
        self.lin_layer = nn.Linear(in_dim, out_dim)
        self.out_mlp = nn.Sequential(nn.Linear(out_dim * Nnodes, hidden_dim),
                                        nn.Dropout(p=0.5),
                                        nn.ReLU(),
                                 nn.Linear(hidden_dim, out_dim* Nnodes))
        self.batchnorm_layer =nn.BatchNorm1d(hidden_dim)
        self.Nnodes = Nnodes
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
    #q,k,v都是8*2*3000*4
    d_k = query.size(-1)
    n_head = query.size(1)
    # This scaling factor helps prevent the dot products from becoming too large 
    # and causing the softmax function to have extremely small gradients, 
    # which can lead to difficulties in training.
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print(query.shape)
    # print(scores.shape)
    # print(d_k)
    scores = scores
    if mask is not None:
        mask = torch.stack([mask] * n_head, dim=1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    #softmax: which normalizes the scores to form a valid probability distribution.
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
        #算attention
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

        #This helps to stabilize the input to the attention mechanism 
        # and makes training more robust.
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
        # This ensures that the activations are normalized 
        # before they are passed to the next layer in the encoder, 
        # which further improves the stability and performance of the network during training.
        self.norms = clones(nn.LayerNorm(d_model), n_layers)

    def forward(self, x, mask):
        outputs = []
        attns = []
        for layer, norm in zip(self.layers, self.norms):
            x, attn = layer(x, mask)
            #x:8*3000*8
            #attn: 8*2*3000*3000
            x = norm(x)
            #x:8*3000*8
            outputs.append(x)
            attns.append(attn)
        return outputs[-1], outputs, attns


class GraphEncoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, Nnodes):
        super(GraphEncoder, self).__init__()
        # Forward Transformers
        self.encoder_f = Encoder(d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, Nnodes)

    def forward(self, x, mask):
        #x: 8*3000*8
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

class sort_readout(nn.Module):
    def __init__(self, K, feat_dim, hidden_dim, out_dim, Nnodes=3686):
        super(sort_readout, self).__init__()
        
        self.W = nn.Parameter(torch.randn(Nnodes)) #trainable weight add as parameter
        # torch.nn.init.normal_(self.W, mean=0.0, std=1.0)
        self.K = K #K= topK, here is 1000
        self.out_indim = K * feat_dim # feat_dim = 8, out_indim = 1000*8 = 8000
        self.mlp_layers = nn.Sequential(nn.Linear(self.out_indim, hidden_dim),
                                        nn.Dropout(p=0.5),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, out_dim))
        self.batchnorm_layer = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, device):
        b_size = x.size(0)
        #ABS
        _, topk_indices = torch.abs(self.W).topk(self.K)#only indice needed
        # _, topk_indices = self.W.topk(self.K)
        # x = x.reshape(b_size, x.size(2), -1)

        x = torch.mul(x, ((torch.abs(self.W).unsqueeze(0)).unsqueeze(2)).to(device))

        # x = x.reshape(b_size,x.size(2),-1)
        topk_x = x.index_select(1, topk_indices.to(device))#only keep topk gene features

        mp = topk_x.reshape(b_size, -1)
        for i in range(1):
            mp = self.mlp_layers[3 * i](mp)
            mp = self.batchnorm_layer(mp)
            mp = self.mlp_layers[3 * i + 1](mp)
            mp = self.mlp_layers[3 * i + 2](mp)
        mp = self.mlp_layers[-1](mp)
        return mp, topk_indices

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


class topK_Pathormer_reverse_v2(nn.Module):
    def __init__(self, n_copy, copy_dim, d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, out_hidden,
                 Nnodes=3686, out_dim=33, device=None, K=50):
        super(topK_Pathormer_reverse, self).__init__()
        self.W = nn.Parameter(torch.randn(Nnodes)) #trainable weight add as parameter
        # torch.nn.init.normal_(self.W, mean=0.0, std=1.0)
        self.K = K #K= topK, here is 1000
        self.in_dim = 1 + K
        self.d_model = d_model
        self.device = device
        #self.geneEmb = geneexpEmbedding(n_vocab, d_model)
        self.copyEmb = copynumberEmbedding(n_copy, copy_dim)
        self.translayer = nn.Linear(self.in_dim, self.d_model) #3001 -> 8
        self.dropout_op = nn.Dropout(p=drop_out)        
        self.graph_encoder = GraphEncoder(d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, K)
        out_indim = K * d_model
        self.readout_layer = readout(out_indim, out_hidden, out_dim)
    
    def _get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    #def forward(self, Xg, Xc, adj, mask=None):
    def forward(self, Xg, adj, mask=None):
        b_size = Xg.size(0)
        #P = torch.exp(self.W)/ torch.sum(torch.exp(self.W))
        P = torch.exp(torch.abs(self.W))/ torch.sum(torch.exp(torch.abs(self.W)))

        _, topk_indices = torch.abs(self.W).topk(self.K)#only indice needed
        #_, topk_indices = P.topk(self.K)#only indice needed
        device = self._get_device()
        weight_grad_cahnnel = torch.stack([torch.abs(self.W)]*b_size, dim=0).unsqueeze(2).to(device)
        #weight_grad_cahnnel = torch.stack([P]*b_size, dim=0).unsqueeze(2).to(device)
        #P_scale = torch.stack([topk_P]*b_size, dim=0).to(device)

        emb_g = Xg.index_select(1, topk_indices.to(device))
        adj = adj.index_select(1, topk_indices.to(device)).index_select(2, topk_indices.to(device))
        #weight_grad_cahnnel = weight_grad_cahnnel.index_select(1, topk_indices.to(device))
        weight_grad_cahnnel = weight_grad_cahnnel.index_select(1, topk_indices.to(device))
        #emb_c = self.dropout_op(self.copyEmb(Xc))
        #emb = torch.cat([emb_g, emb_c, adj], dim=-1)
        #emb = torch.cat([emb_g, adj, weight_grad_cahnnel], dim=-1)
        #emb = self.dropout_op(self.translayer(emb))
        emb = torch.cat([emb_g, adj], dim=-1)
        emb = self.translayer(emb)
        emb = torch.mul(emb, weight_grad_cahnnel)
        emb = self.dropout_op(emb)
        
        h, _, attn_list = self.graph_encoder(emb,mask)
        ##### add
        #h = torch.mul(h, weight_grad_cahnnel)
        #h = h * P_scale
        h = h.reshape(b_size, -1)
        out = self.readout_layer(h)
        return out, topk_indices, attn_list, P



class topK_Pathormer_reverse(nn.Module):
    def __init__(self, n_copy, copy_dim, d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, out_hidden,
                 Nnodes=3686, out_dim=33, device=None, K=50):
        super(topK_Pathormer_reverse, self).__init__()
        self.W = nn.Parameter(torch.randn(Nnodes)) #trainable weight add as parameter
        # torch.nn.init.normal_(self.W, mean=0.0, std=1.0)
        self.K = K #K= topK, here is 1000
        self.in_dim = 1 + Nnodes
        self.d_model = d_model
        self.device = device
        #self.geneEmb = geneexpEmbedding(n_vocab, d_model)
        self.copyEmb = copynumberEmbedding(n_copy, copy_dim)
        self.translayer = nn.Linear(self.in_dim, self.d_model) #3001 -> 8
        self.dropout_op = nn.Dropout(p=drop_out)        
        #self.graph_encoder = GraphEncoder(d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, K)
        self.graph_encoder = GraphEncoder(d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, Nnodes)
        self.Nnodes = Nnodes
        out_indim = Nnodes * d_model
        self.readout_layer = readout(out_indim, out_hidden, out_dim)
    
    def _get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    #def forward(self, Xg, Xc, adj, mask=None):
    def forward(self, Xg, adj, mask=None):
        b_size = Xg.size(0)
        #P = torch.exp(self.W)/ torch.sum(torch.exp(self.W))
        P = torch.exp(torch.abs(self.W))/ torch.sum(torch.exp(torch.abs(self.W)))

        #_, topk_indices = torch.abs(self.W).topk(self.K)#only indice needed
        _, topk_indices = P.topk(self.K)#only indice needed
        device = self._get_device()
        #weight_grad_cahnnel = torch.stack([torch.abs(self.W)]*b_size, dim=0).unsqueeze(2).to(device)
        weight_grad_cahnnel = torch.stack([P]*b_size, dim=0).unsqueeze(2).to(device)
        #P_scale = torch.stack([topk_P]*b_size, dim=0).to(device)
        weight_grad_cahnnel = weight_grad_cahnnel.index_select(1, topk_indices.to(device))
        emb_g = torch.zeros(b_size,self.Nnodes,1).to(device)
        emb_g[:, topk_indices] = torch.mul(Xg.index_select(1, topk_indices.to(device)), weight_grad_cahnnel)
        emb_adj = torch.zeros(b_size, self.Nnodes, self.Nnodes).to(device)
        emb_adj[:, topk_indices][:,:,topk_indices] = torch.mul(adj.index_select(1, topk_indices.to(device)).index_select(2, topk_indices.to(device)), weight_grad_cahnnel)
        #adj = adj.index_select(1, topk_indices.to(device)).index_select(2, topk_indices.to(device))
        #weight_grad_cahnnel = weight_grad_cahnnel.index_select(1, topk_indices.to(device))

        #emb_c = self.dropout_op(self.copyEmb(Xc))
        #emb = torch.cat([emb_g, emb_c, adj], dim=-1)
        #emb = torch.cat([emb_g, adj, weight_grad_cahnnel], dim=-1)
        #emb = self.dropout_op(self.translayer(emb))
        #emb_g = torch.mul(emb_g, weight_grad_cahnnel)
        #emb_adj = torh.mul(emb_adj, weight_grad_cahnnel)
        emb = torch.cat([emb_g, emb_adj], dim=-1)
        #print(emb.shape)
        #emb = torch.mul(emb, weight_grad_cahnnel)
        #emb = torch.cat([emb_g, adj], dim=-1)
        emb = self.translayer(emb)
        #emb = torch.mul(emb, weight_grad_cahnnel)
        emb = self.dropout_op(emb)
        
        h, _, attn_list = self.graph_encoder(emb,mask)
        ##### add
        #h = torch.mul(h, weight_grad_cahnnel)
        #h = h * P_scale
        h = h.reshape(b_size, -1)
        out = self.readout_layer(h)
        return out, topk_indices, attn_list, P 


class topK_Pathormer(nn.Module):
    def __init__(self, n_copy, copy_dim, d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, out_hidden,
                 Nnodes=3686, out_dim=33, device=None, K=50):
        super(topK_Pathormer, self).__init__()
        #self.in_dim = 1 + copy_dim + Nnodes
        self.in_dim = 1 + Nnodes
        self.K = K
        self.d_model = d_model
        self.device = device
        #self.geneEmb = geneexpEmbedding(n_vocab, d_model)
        self.copyEmb = copynumberEmbedding(n_copy, copy_dim)
        self.translayer = nn.Linear(self.in_dim, self.d_model) #3001 -> 8
        self.dropout_op = nn.Dropout(p=drop_out)        
        self.graph_encoder = GraphEncoder(d_model, n_head, d_ff, drop_out, n_layers, hidden_dim, Nnodes)
        self.readout_layer = sort_readout(self.K, d_model, out_hidden, out_dim, Nnodes)

    #def forward(self, Xg, Xc, adj, mask=None):
    def _get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    def forward(self, Xg, adj, mask=None):
        #batch_x是8*3000
        #Xg 就是batch_x.unsqueeze(2), 8*3000*1
        emb_g = Xg

        #emb_c = self.dropout_op(self.copyEmb(Xc))
        #emb = torch.cat([emb_g, emb_c, adj], dim=-1)

        #emb: 8*3000*3001
        emb = torch.cat([emb_g, adj], dim=-1) #last dimension
        

        #translayer #3001 -> 8 , nn.Linear
        #emb: 8*3000*8 相当于每个genes
        emb = self.dropout_op(self.translayer(emb))
        
        # b_size = emb.size(0)
        h, _, attn_list = self.graph_encoder(emb,mask)
        
        
        #h = h.reshape(b_size, -1)
        out, topk_indices = self.readout_layer(h, self._get_device())
        
        return out, topk_indices, attn_list