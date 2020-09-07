import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, GATConv


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_size, num_heads=2, feat_drop=0.2, attn_drop=0.2)
        self.conv2 = GATConv(hidden_size, num_classes, num_heads=1, feat_drop=0.2, attn_drop=0.2)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = torch.mean(h, dim=1)
        h = self.conv2(g, h)
        # self.h_ = h
        h = torch.mean(h, dim=1)
        return h
