import argparse, time, math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from functools import partial


def gcn_msg(edge):
    msg = edge.src['h'] #* edge.src['norm']
    return {'m': msg}


def gcn_reduce(nets, nodes):
    for i, net in enumerate(nets):
        mailbox = nodes.mailbox['m'][i].clone()
        h = net[0](mailbox)
        o = net[1](h)
        mask = o.round()
        nodes.mailbox['m'][i] = mailbox * mask
    accum = torch.sum(nodes.mailbox['m'], 1) * nodes.data['norm']
    return {'h': accum}


def gcn_reduce3(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1) * nodes.data['norm']
    return {'h': accum}


class NodeApplyModule(nn.Module):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        h = nodes.data['h']
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.node_update = NodeApplyModule(out_feats, activation, bias)
        self.reset_parameters()
        self.nets = []
        for _ in self.g.nodes():
            linear = nn.Linear(in_feats, 1)
            sigmoid = nn.Sigmoid()
            self.nets.append([linear, sigmoid])

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        gcn_reduce2 = partial(gcn_reduce, self.nets)
        self.g.ndata['h'] = h
        self.g.update_all(gcn_msg, gcn_reduce2)
        self.g.ndata['h'] = torch.mm(self.g.ndata['h'], self.weight)
        self.g.update_all(gcn_msg, gcn_reduce3, self.node_update)
        h = self.g.ndata.pop('h')
        return h


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers=1,
                 activation=torch.relu,
                 dropout=0):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # # input layer
        # self.layers.append(GCNLayer(g, in_feats, n_hidden, activation, dropout))
        # # hidden layers
        # for i in range(n_layers - 1):
        #     self.layers.append(GCNLayer(g, n_hidden, n_hidden, activation, dropout))
        # # output layer
        self.layers.append(GCNLayer(g, in_feats, n_classes, None, dropout))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h.mean(dim=0)
