import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.integrate import solve_ivp
import dgl
import os

import gnode


def ll_equations(t, z):
    x1, x2, x3, x4, x5, x6, x7 = z
    x1dt = 1.4 * x3 - 0.9 * x1
    x2dt = 2.5 * x5 - 1.5 * x2
    x3dt = 0.6 * x7 - 0.8 * x2 * x3
    x4dt = 2 - 1.3 * x3 * x4
    x5dt = 0.7 * x1 - x4 * x5
    x6dt = 0.3 * x1 - 3.1 * x6
    x7dt = 1.8 * x6 - 1.5 * x2 * x7
    dzdt = np.array([x1dt, x2dt, x3dt, x4dt, x5dt, x6dt, x7dt])
    return dzdt


def laub_loomis():
    x10 = 1.2
    x20 = 1.05
    x30 = 1.5
    x40 = 2.4
    x50 = 1
    x60 = 0.1
    x70 = 0.45

    conc_t0 = np.array([x10, x20, x30, x40, x50, x60, x70])
    tau = 0.01

    ODE_args = ()
    sol = solve_ivp(fun=ll_equations, t_span=[0, 20],
                    y0=conc_t0,
                    max_step=tau,
                    args=ODE_args, method="LSODA")

    # plt.figure()
    # # fig, ax = plt.subplots()
    # # sns.lineplot(np.linspace(0, 20, len(sol['y'][3])), sol['y'][3])
    # plt.scatter(np.linspace(0, 20, len(sol['y'][3])), sol['y'][3])
    # # plt.axis('equal')
    # # ax.axis('off')
    # plt.show()
    # return

    return sol['y'].T


def main():
    result_dir = 'results/ll_prehuman/'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    x = laub_loomis()
    x = x.astype('float32')
    scaler = StandardScaler()
    scaler = scaler.fit(x)
    x = scaler.transform(x)

    window_size = 20
    samples = []
    labels = []
    for batch in range(x.shape[0]-window_size+1):
        print(f"{batch} - {batch+window_size-2} -> {batch+window_size-1}")
        samples.append(x[batch:batch+window_size-2])
        labels.append(x[batch+window_size-1])

    samples = np.array(samples)
    labels = np.array(labels)

    skf = KFold(n_splits=5, shuffle=True)
    train_index, val_index = [split for split in skf.split(samples)][0]
    x_train, x_val = samples[train_index], samples[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    # edge_list = [
    #     (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
    #     (2, 0),
    #     (4, 1),
    #     (6, 2), (1, 2),
    #     (2, 3),
    #     (0, 4), (3, 4),
    #     (0, 5),
    #     (5, 6), (1, 6)
    # ]
    edge_list = []
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            edge_list.append((i, j))
    G = dgl.DGLGraph(edge_list)
    dp = gnode.DigitalPatient(G, epochs=10, lr=0.05, window_size=window_size-2)

    # plot the graph corresponding to the digital patient
    nx_G = dp.G.to_networkx()
    pos = nx.circular_layout(nx_G)
    # pos = nx.spring_layout(nx_G)
    node_labels = {}
    for i, cn in enumerate(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']):
        node_labels[i] = cn
    plt.figure()
    nx.draw(nx_G, pos)#, alpha=0.3)
    nx.draw_networkx_labels(nx_G, pos, font_color='w', labels=node_labels)
    plt.tight_layout()
    plt.savefig(f'{result_dir}/graph.png')
    plt.show()
    # return

    dp.fit(x_train, y_train, nx_G, pos, node_labels, display=True, result_dir=result_dir)
    predictions = dp.predict(x_val, trajectory=False)
    # predictions = dp.predict(x_train, trajectory=False)

    t = np.arange(0, len(y_val))
    plt.figure()
    # fig, ax = plt.subplots(figsize=[2, 4])
    plt.scatter(t, y_val[:, 3], label='True dynamics')
    # plt.scatter(y_train[:, 0], y_train[:, 1], label='True dynamics')
    plt.scatter(t, predictions[:, 3], label='GNN predictions')
    # plt.axis('equal')
    # ax.axis('off')
    plt.legend()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.savefig(f'{result_dir}/x4.png')
    plt.show()

    return


if __name__ == '__main__':
    main()
