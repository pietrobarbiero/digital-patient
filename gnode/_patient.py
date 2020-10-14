import itertools

import dgl
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ._nn import GCN


class DigitalPatient():

    def __init__(self, G, epochs=30, lr=0.01, window_size=10):
        super().__init__()
        self.G = G
        self.epochs = epochs
        self.lr = lr
        self.window_size = window_size

    def fit(self, x_train, y_train, nx_G=None, pos=None, node_labels=None, display=False, result_dir=None):
        """
        Fit model

        :param x_train: training data
        :param y_train: training labels
        :return:
        """
        # initialize GCN
        node_embed = nn.Embedding(x_train.shape[2], x_train.shape[1])
        # edge_embed = nn.Embedding(self.G.batch_num_edges[0], self.window_size)
        self.G.ndata['feat'] = node_embed.weight
        # self.G.edata['w'] = edge_embed.weight
        # normalization
        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        self.G.ndata['norm'] = norm.unsqueeze(1)

        self.net_ = GCN(self.G, x_train.shape[1], self.window_size, y_train.shape[1])
        optimizer = torch.optim.Adagrad(itertools.chain(self.net_.parameters(),
                                                        node_embed.parameters()),
                                        lr=self.lr)
        # define inputs and outputs
        inputs = torch.tensor(x_train)
        labels = torch.tensor(y_train)

        # train the model
        self.all_logits_ = []
        for epoch in range(self.epochs):
            mse_loss_list = []
            attention_loss_list = []
            for b, (x, y) in enumerate(zip(inputs, labels)):
                self.net_.train()
                # if self.net_.conv1.graph_ is None:
                #     attention = torch.zeros(1)
                # else:
                #     attention1 = self.net_.conv1.graph_.edata['a'].clone().detach()
                #     attention2 = self.net_.conv2.graph_.edata['a'].clone().detach()
                #     attention = attention1 * attention2

                # forward pass
                logits = self.net_(x.T)
                mse_loss = F.mse_loss(logits.squeeze(), y)
                # attention_loss = torch.norm(attention, 2)
                # loss = mse_loss + attention_loss
                mse_loss_list.append(mse_loss.item())
                # attention_loss_list.append(attention_loss.item())

                # backward pass
                optimizer.zero_grad()
                mse_loss.backward()
                optimizer.step()
                # break

            # if display:
            #     attention_np = attention.detach().numpy().squeeze()
            #     plt.figure()
            #     nx.draw_networkx_nodes(nx_G, pos)
            #     nx.draw_networkx_edges(nx_G, pos, width=(attention_np+1)**2,
            #                            edge_color=attention_np, edge_cmap=plt.get_cmap('Reds'))
            #     nx.draw_networkx_labels(nx_G, pos, font_color='w', labels=node_labels)
            #     plt.tight_layout()
            #     plt.savefig(f'{result_dir}/graph_{epoch}.png')
            #     plt.show()

            print(f'Epoch {epoch}'
                  f' | MSE loss: {np.mean(mse_loss_list):.4f}'
                  f' | Attention loss: {np.mean(attention_loss_list):.4f}')

        return self

    def predict(self, x, trajectory=True):
        """
        Predict labels for the given input

        :param x: input data
        :return:
        """
        predictions = []
        for b, xi in enumerate(x):
            if trajectory:
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
            else:
                pred = self.net_(torch.tensor(xi.T)).detach().numpy()
                predictions.append(pred)
        return np.array(predictions).squeeze()

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
