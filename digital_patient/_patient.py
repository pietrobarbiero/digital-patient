import itertools

import dgl
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._nn import GCN
from .conformal.base import RegressorMixin


class DigitalPatient(RegressorMixin):

    def __init__(self, G, epochs=30, lr=0.01, window_size=10):
        super().__init__()
        self.G = G
        self.epochs = epochs
        self.lr = lr
        self.window_size = window_size

    def fit(self, x_train, y_train):
        """
        Fit model

        :param x_train: training data
        :param y_train: training labels
        :return:
        """
        # initialize GCN
        node_embed = nn.Embedding(x_train.shape[2], x_train.shape[1])
        edge_embed = nn.Embedding(self.G.batch_num_edges()[0], self.window_size)
        self.G.ndata['feat'] = node_embed.weight
        self.G.edata['w'] = edge_embed.weight
        self.G = dgl.add_self_loop(self.G)
        self.net_ = GCN(x_train.shape[1], self.window_size, 1)
        optimizer = torch.optim.Adagrad(itertools.chain(self.net_.parameters(),
                                                        node_embed.parameters(),
                                                        edge_embed.parameters()),
                                        lr=self.lr)
        # define inputs and outputs
        inputs = torch.tensor(x_train)
        labels = torch.tensor(y_train[:, 0])

        # train the model
        self.all_logits_ = []
        for epoch in range(self.epochs):
            loss_list = []
            for b, (x, y) in enumerate(zip(inputs, labels)):
                # forward pass
                logits = self.net_(self.G, x.T)
                loss = F.mse_loss(logits.squeeze(), y)

                # we save the logits and the loss for visualization later
                self.all_logits_.append(logits.detach())
                loss_list.append(loss.item())

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # break

            print(f'Epoch {epoch} | Loss: {np.mean(loss_list):.4f}')

        return self

    def predict(self, x):
        """
        Predict labels for the given input

        :param x: input data
        :return:
        """
        predictions = []
        for b, xi in enumerate(x):
            trajectories = []
            for k in range(5):
                trajectory = xi.T
                for j in range(self.window_size):
                    pred = self.net_(self.G, torch.tensor(trajectory)).detach().numpy()
                    trajectory = np.concatenate([trajectory[:, 1:], pred], axis=1)
                trajectories.append(trajectory)
            trajectories = np.stack(trajectories)
            predictions.append(trajectories)
            if b > 1:
                break
        return np.array(predictions)

    def draw(self, ax, epoch=-1):
        """
        Draw graph

        :param ax: figure axes
        :param epoch: epoch to draw
        :return:
        """
        cls1color = '#00FFFF'
        cls2color = '#FF00FF'
        pos = {}
        colors = []
        for v in range(34):
            pos[v] = self.all_logits_[epoch][v].numpy()
            cls = pos[v].argmax()
            colors.append(cls1color if cls else cls2color)
        ax.cla()
        ax.axis('off')
        nx.draw_networkx(self.G.to_networkx().to_undirected(),
                         pos, node_color=colors,
                         with_labels=True, node_size=300, ax=ax)
