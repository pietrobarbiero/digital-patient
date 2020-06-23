import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.integrate import solve_ivp

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
    x = van_der_pol()
    x = x.astype('float32')

    window_size = 5
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

    dp = digital_patient.DigitalPatient()
    dp.build_graph()

    nx_G = dp.G_.to_networkx()  # .to_undirected()
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = nx.kamada_kawai_layout(nx_G)
    plt.figure()
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()

    dp.train(x_train, y_train, epochs=5)
    predictions = dp.predict(x_val)

    plt.figure(figsize=[2, 4])
    fig, ax = plt.subplots(figsize=[2, 4])
    sns.scatterplot(y_val[:, 0], y_val[:, 1], label='True dynamics')
    sns.scatterplot(predictions[:, 0], predictions[:, 1], label='GNN predictions')
    plt.axis('equal')
    ax.axis('off')
    plt.legend()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.savefig("van_der_pol.png")
    plt.show()

    return


if __name__ == '__main__':
    main()
