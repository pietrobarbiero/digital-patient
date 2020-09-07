import os

import dgl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from sklearn.model_selection import KFold

import digital_patient

from digital_patient.conformal.base import RegressorAdapter
from digital_patient.conformal.icp import IcpRegressor
from digital_patient.conformal.nc import RegressorNc
from examples.load_data2 import load_physiology


def main():
    # create directory to save results
    output_dir = 'cardiac-model'
    data_dir = os.path.join(output_dir, 'data')
    result_dir = os.path.join(output_dir, 'results')
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # load data
    df = pd.read_csv(os.path.join(data_dir, 'data.csv'), index_col=0)
    var_names = [name.split(' ')[0] for name in df.columns]
    x = df.values.astype('float32')
    reps = 10
    x = np.tile(x.T, reps=reps).T
    # # check
    # plt.figure()
    # plt.plot(x[:500, 0], x[:500, 1])
    # plt.show()
    # # scale data
    # scaler = StandardScaler()
    # scaler = scaler.fit(x)
    # x = scaler.transform(x)
    # create sample lists
    samples = []
    labels = []
    window_size = 1000
    for batch in range(x.shape[0] - 2 * window_size):
        print(f"{batch} - {batch + window_size - 2} -> {batch + window_size - 1} - {batch + 2 * window_size - 3}")
        samples.append(x[batch:batch + window_size - 2])
        labels.append(x[batch + window_size - 1:batch + 2 * window_size - 3])
    samples = np.array(samples)
    labels = np.array(labels)
    # create CV splits
    skf = KFold(n_splits=5, shuffle=True)
    trainval_index, test_index = [split for split in skf.split(samples)][0]
    skf2 = KFold(n_splits=5, shuffle=True)
    train_index, val_index = [split for split in skf2.split(np.arange(trainval_index.size))][0]
    x_train, x_val = samples[trainval_index[train_index]], samples[trainval_index[val_index]]
    y_train, y_val = labels[trainval_index[train_index]], labels[trainval_index[val_index]]
    x_test, y_test = samples[test_index], labels[test_index]
    # create edge list
    edge_list = []
    for i in range(df.shape[1]):
        for j in range(df.shape[1]):
            edge_list.append((i, j))

    # instantiate a digital patient model
    G = dgl.DGLGraph(edge_list)
    dp = digital_patient.DigitalPatient(G, epochs=20, lr=0.01, window_size=window_size-2)

    # # plot the graph corresponding to the digital patient
    # nx_G = dp.G.to_networkx()
    # pos = nx.circular_layout(nx_G)
    # node_labels = {}
    # for i, cn in enumerate(var_names):
    #     node_labels[i] = cn
    # plt.figure()
    # nx.draw(nx_G, pos, alpha=0.3)
    # nx.draw_networkx_labels(nx_G, pos, labels=node_labels)
    # plt.tight_layout()
    # plt.savefig(f'{result_dir}/graph.png')
    # plt.show()

    # instantiate the model, train and predict
    dp.fit(x_train, y_train)
    predictions = dp.predict(x_test)

    # plot the results
    sns.set_style('whitegrid')
    for i, name in enumerate(var_names):
        for j in range(predictions.shape[0]):
            xi = y_test[j, :, i]
            pi = predictions[j, :, i]

            if name == 't':
                continue

            ti = labels[0, :, 0]
            # tik = np.repeat(ti, pi.shape[0])
            pik = np.hstack(pi)

            plt.figure()
            plt.plot(ti, xi, label='true')
            for pik in pi:
                plt.plot(ti, pik, c='r', alpha=0.2)
            # sns.lineplot(tik, pik, alpha=0.2, ci=0.9)
            # plt.fill_between(ti, pi[:, 0], pi[:, 1], alpha=0.2, label='predicted')
            plt.title(name)
            plt.legend()
            # plt.ylabel(ylabel)
            plt.xlabel('time')
            plt.tight_layout()
            plt.savefig(f'{result_dir}/{name}_{j}.png')
            plt.show()
            break

    return


if __name__ == '__main__':
    main()
