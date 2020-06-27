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


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
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


class DigitalPatient:

    def __init__(self):
        return

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

    def train(self, x_train, y_train, epochs=100, lr=0.01):
        embed = nn.Embedding(x_train.shape[2], x_train.shape[1])
        self.G_.ndata['feat'] = embed.weight
        # self.net_ = GCN(x_train.shape[1], 10, x_train.shape[2])
        self.net_ = GCN(x_train.shape[1], 10, 1)

        inputs = torch.tensor(x_train)
        labels = torch.tensor(y_train[:, :, np.newaxis])  # their labels are different

        optimizer = torch.optim.Adagrad(itertools.chain(self.net_.parameters(), embed.parameters()), lr=lr)
        # optimizer = torch.optim.Adagrad(self.net_.parameters(), lr=lr)
        all_logits = []
        for epoch in range(epochs):
            loss_list = []
            for b, (x, y) in enumerate(zip(inputs, labels)):
                logits = self.net_(self.G_, x.T)
                # we save the logits for visualization later
                all_logits.append(logits.detach())
                loss = F.mse_loss(logits, y)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch} | Loss: {np.mean(loss_list):.4f}')

        return

    def predict(self, x_test):
        preds = []
        for b, x in enumerate(x_test):
            pred = self.net_(self.G_, torch.tensor(x.T)).detach().numpy().squeeze()
            preds.append(pred)
        return np.vstack(preds)
