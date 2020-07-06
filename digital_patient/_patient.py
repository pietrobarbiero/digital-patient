import itertools

import dgl
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv
import matplotlib.pyplot as plt
import seaborn as sns
from conformalgnn.base import RegressorMixin


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
        self.conv3 = GraphConv(hidden_size, num_classes)
        self.classify = nn.Linear(num_classes, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h
        # h = torch.relu(h)
        # g.ndata['h'] = h
        # hg = dgl.mean_nodes(g, 'h')
        # logits = self.classify(hg).T
        # return logits


class DigitalPatient(RegressorMixin):

    def __init__(self, epochs=30, lr=0.01, window_size=10):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.window_size = window_size

    def draw(self, ax, nx_G, all_logits, i):
        cls1color = '#00FFFF'
        cls2color = '#FF00FF'
        pos = {}
        colors = []
        for v in range(34):
            pos[v] = all_logits[i][v].numpy()
            cls = pos[v].argmax()
            colors.append(cls1color if cls else cls2color)
        ax.cla()
        ax.axis('off')
        ax.set_title('Epoch: %d' % i)
        nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
                         with_labels=True, node_size=300, ax=ax)

    def build_graph(self, elist):
        # All edges are stored in two numpy arrays. One for source endpoints
        # while the other for destination endpoints.
        # elist = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 2), (2, 1)]
        # elist = [(1, 0), (0, 1)]
        # Construct a DGLGraph
        self.G_ = dgl.DGLGraph(elist)

    def fit(self, x_train, y_train):
        node_embed = nn.Embedding(x_train.shape[2], x_train.shape[1])
        edge_embed = nn.Embedding(self.G_.batch_num_edges[0], self.window_size)
        self.G_.ndata['feat'] = node_embed.weight
        self.G_.edata['w'] = edge_embed.weight

        # self.net_ = GCN(x_train.shape[1], 10, x_train.shape[2])
        self.net_ = GCN(x_train.shape[1], self.window_size, self.window_size)

        inputs = torch.tensor(x_train)
        labels = torch.tensor(y_train)  # their labels are different

        optimizer = torch.optim.Adagrad(itertools.chain(self.net_.parameters(),
                                                        node_embed.parameters(),
                                                        edge_embed.parameters()),
                                        lr=self.lr)
        # optimizer = torch.optim.Adagrad(self.net_.parameters(), lr=lr)
        all_logits = []
        for epoch in range(self.epochs):
            loss_list = []
            for b, (x, y) in enumerate(zip(inputs, labels)):
                logits = self.net_(self.G_, x.T)
                # we save the logits for visualization later
                all_logits.append(logits.detach())
                loss = F.mse_loss(logits, y.T)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch} | Loss: {np.mean(loss_list):.4f}')

        return self

    def predict(self, x):
        predictions = []
        for b, xi in enumerate(x):
            pred = self.net_(self.G_, torch.tensor(xi.T)).detach().numpy().squeeze().T
            predictions.append(pred)
        return np.array(predictions)
