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

import digital_patient


def vdp_equations(t, z, mu):
    x, y = z
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    dzdt = np.array([dxdt, dydt])
    return dzdt


def van_der_pol():
    x0 = 2
    y0 = 0
    mu = 4

    conc_t0 = np.array([x0, y0])
    tau = 0.1

    ODE_args = (mu,)
    sol = solve_ivp(fun=vdp_equations, t_span=[0, 150],
                    y0=conc_t0,
                    max_step=tau,
                    args=ODE_args, method="LSODA")

    # plt.figure()
    # fig, ax = plt.subplots()
    # sns.scatterplot(sol['y'][0], sol['y'][1])
    # plt.axis('equal')
    # ax.axis('off')
    # plt.show()

    return sol['y'].T


def main():
    result_dir = './results/vdp/'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    x = van_der_pol()
    x = x.astype('float32')
    scaler = StandardScaler()
    scaler = scaler.fit(x)
    x = scaler.transform(x)

    window_size = 100
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

    edge_list = [
        (0, 1), (1, 0),
        (1, 1), (0, 0)
    ]
    G = dgl.DGLGraph(edge_list)
    dp = digital_patient.DigitalPatient(G, epochs=5, lr=0.01, window_size=window_size-2)

    # plot the graph corresponding to the digital patient
    nx_G = dp.G.to_networkx()
    # pos = nx.circular_layout(nx_G)
    pos = nx.spring_layout(nx_G)
    node_labels = {}
    for i, cn in enumerate(['x', 'y']):
        node_labels[i] = cn
    plt.figure()
    nx.draw(nx_G, pos)#, alpha=0.3)
    nx.draw_networkx_labels(nx_G, pos, font_color='w', labels=node_labels)
    plt.tight_layout()
    plt.savefig(f'{result_dir}/graph.png')
    plt.show()

    dp.fit(x_train, y_train)
    predictions = dp.predict(x_val, trajectory=False)
    # predictions = dp.predict(x_train, trajectory=False)

    plt.figure(figsize=[2, 4])
    fig, ax = plt.subplots(figsize=[2, 4])
    sns.scatterplot(y_val[:, 0], y_val[:, 1], label='True dynamics')
    # sns.scatterplot(y_train[:, 0], y_train[:, 1], label='True dynamics')
    sns.scatterplot(predictions[:, 0], predictions[:, 1], label='GNN predictions')
    plt.axis('equal')
    ax.axis('off')
    plt.legend()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    # plt.savefig("van_der_pol.png")
    plt.show()

    return


if __name__ == '__main__':
    main()
