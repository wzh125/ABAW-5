import numpy as np
import torch.nn.functional as F
from swin_transformer import swin_transformer_base
from graph import normalize_digraph
from basic_block import *
from mae_pretrain.modeling_pretrain import mae_face


class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.size = d_model
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(self.norm(x), self.norm(x), self.norm(x)))
        return x + self.dropout(self.feed_forward(self.norm(x)))

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([])
        for i in range(4):
            self.linears.append(nn.Linear(d_model, d_model))
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(GNN, self).__init__()
        # in_channels: dim of node feature
        # num_classes: num of nodes
        # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
        # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
        # X' = ReLU(X + BN(V(X) + A x U(X)) )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(self.in_channels,self.in_channels)
        self.V = nn.Linear(self.in_channels,self.in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)

        # init
        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape

        # build dynamical graph
        if self.metric == 'dots':
            si = x.detach()
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'cosine':
            si = x.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'l1':
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si)
            si = si.sum(dim=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()

        else:
            raise Exception("Error: wrong metric: ", self.metric)

        # GNN process
        A = normalize_digraph(adj)
        aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
        x = self.relu(x + self.bnv(aggregate + self.U(x)))
        return x


class Head(nn.Module):
    def __init__(self, task, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(Head, self).__init__()
        self.task = task
        self.in_channels = in_channels

        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.gnn = GNN(self.in_channels, self.num_classes,neighbor_num=neighbor_num,metric=metric)

        self.encoder = EncoderLayer(d_model=self.in_channels, h=4, d_ff=2*self.in_channels, dropout=0.1)

        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.sc)

    def forward(self, x, B, T):
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        # FGG
        for module_num in range(3):
            f_v = self.gnn(f_v)
            BT,N,d = f_v.shape
            f_v = f_v.view(B,T,N,d).transpose(1,2)
            f_v = f_v.contiguous().view(B*N,T,d)

            f_v = self.encoder(f_v,mask=None)
            f_v = f_v.view(B,N,T,d).transpose(1,2)
            f_v = f_v.contiguous().view(B*T,N,d)

        b, n, c = f_v.shape
        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1)
        cl = (cl * sc.view(1, n, c)).sum(dim=-1)

        return cl


class MAE_Graph(nn.Module):
    def __init__(self, task, num_classes=12, backbone='mae', neighbor_num=4, metric='dots'):
        super(MAE_Graph, self).__init__()
        if backbone == 'mae':
            self.backbone = mae_face(pretrained=True)
        self.in_channels = self.backbone.num_features
        self.out_channels = self.in_channels // 2

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(task, self.out_channels, num_classes, neighbor_num, metric)

    def forward(self, x):
        # x: b d c
        B,T,C,H,W = x.shape

        x = x.view(B*T, C, H, W)
        x = self.backbone(x, mask=None)
        x = self.global_linear(x)
        cl = self.head(x, B, T)
        return cl


class swin(nn.Module):
    def __init__(self, task, num_classes=12, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super(swin, self).__init__()
        self.backbone = swin_transformer_base()
        self.in_channels = self.backbone.num_features
        self.out_channels = self.in_channels // 2

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(task, self.out_channels, num_classes, neighbor_num, metric)

    def forward(self, x):
        # x: b d c
        B,T,C,H,W = x.shape

        x = x.view(B*T, C, H, W)
        x = self.backbone(x)
        x = self.global_linear(x)
        cl = self.head(x, B, T)
        return cl

