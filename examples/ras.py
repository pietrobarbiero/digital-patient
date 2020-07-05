import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.integrate import solve_ivp

import digital_patient
from scipy import interpolate


def main():
    # TODO: change message function
    # TODO: graph embedding for predictions
    # TODO: predict trajectory not only the next state

    # x_ras = pd.read_csv('data/70/DKD_drug-5_glu-10_infection-0_renal-impaired.csv')
    # x_diabetes = pd.read_csv('data/70/DIABETES_glu-17.csv')
    x_ras = pd.read_csv('data/70/DKD_drug-5_glu-17_infection-0_renal-normal.csv')
    x_ras.drop(['angII_norm', 'IR'], axis=1, inplace=True)
    x_diabetes = pd.read_csv('data/70/DIABETES_glu-5.csv')
    # x_cardio = pd.read_csv('data/70/CARDIO_drug-5_glu-10_infection-0_renal-impaired.csv')
    tx_ras = x_ras['t']
    tx_diabetes = x_diabetes['t']

    t = np.linspace(3, 4.99, 200)

    x_list = []
    for c in x_ras.columns:
        f = interpolate.interp1d(tx_ras, x_ras[c].values)
        x_list.append(f(t))

    # x = np.vstack([x_angII, x_diacid, x_glu, x_diacid, x_glu]).T
    # x = np.vstack([x_angII, x_diacid, x_glu]).T
    x = np.vstack(x_list).T
    x = x.astype('float32')
    t2 = t

    reps = 20
    x = np.tile(x.T, reps=reps).T
    t2 = np.arange(0, len(x)) / (np.max(t) * reps)
    x[:, 0] = t2

    plt.figure()
    plt.plot(t2, x[:, 0])
    plt.show()

    x = StandardScaler().fit_transform(x)

    window_size = 1000
    samples = []
    labels = []
    t_list = []
    for batch in range(x.shape[0]-2*window_size+1):
        print(f"{batch} - {batch+window_size-2} -> {batch+window_size-1} - {batch+2*window_size-3}")
        samples.append(x[batch:batch+window_size-2])
        labels.append(x[batch+window_size-1:batch+2*window_size-3])
        t_list.append(t2[batch+window_size-1:batch+2*window_size-3])

    samples = np.array(samples)
    labels = np.array(labels)
    t_list = np.array(t_list)

    skf = KFold(n_splits=5, shuffle=True)
    train_index, val_index = [split for split in skf.split(samples)][0]
    x_train, x_val = samples[train_index], samples[val_index]
    y_train, y_val = labels[train_index], labels[val_index]
    t_list = t_list[val_index]


    dp = digital_patient.DigitalPatient()
    # elist = [(1, 0), (2, 0), (3, 1), (4, 2)]
    # elist = [(0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (2, 2), (1, 0), (0, 1)]
    elist = [
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
        (6, 6), (7, 7), (8, 8), (9, 9), (10, 10),
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (0, 7), (0, 8), (0, 9), (0, 10),
        (4, 1), (3, 1), (1, 7), (1, 5), (6, 5), (5, 3),
        (5, 7), (5, 8), (5, 9), (10, 7), (10, 1)
    ]
    # elist = [(0, 0), (1, 1), (2, 2), (1, 0), (2, 0)]
    # elist = [(0, 0), (1, 1), (1, 0)]
    dp.build_graph(elist)

    nx_G = dp.G_.to_networkx()  # .to_undirected()
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = nx.kamada_kawai_layout(nx_G)
    plt.figure()
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()

    dp.train(x_train, y_train, epochs=30, lr=0.01, window_size=window_size-2)
    predictions = dp.predict(x_val)

    sns.set_style('whitegrid')

    n_rows = int(np.sqrt(dp.G_.batch_num_nodes)+1)

    for j, (yv, yp, t) in enumerate(zip(y_val, predictions, t_list)):
        yp = yp.T
        fig, ax = plt.subplots(figsize=[10, 10])
        for i, c in enumerate(x_ras.columns):
            plt.subplot(n_rows, n_rows, i+1)
            plt.title(c)
            plt.plot(t, yv[:, i], c='blue')
            plt.plot(t+np.min(t), yp[:, i], c='orange')
            plt.ylabel('concentration [ng/mL]')
            plt.xlabel('t [sec]')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
        plt.tight_layout()
        plt.savefig(f'RAS_{j}.png')
        plt.show()
        break

    # fig, ax = plt.subplots(figsize=[10, 10])
    # for i, c in enumerate(x_ras.columns):
    #     plt.subplot(n_rows, n_rows, i+1)
    #     for yv, yp, t in zip(y_val, predictions, t_list):
    #         if np.max(t) < 14:
    #             yp = yp.T
    #             plt.title(c)
    #             plt.plot(t, yv[:, i], c='blue')
    #             plt.scatter(t, yp[:, i], alpha=0.5, c='orange', marker='.')
    #             plt.ylabel('concentration [ng/mL]')
    #             plt.xlabel('t [sec]')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
    # plt.tight_layout()
    # plt.savefig('RAS.png')
    # plt.show()

    return


if __name__ == '__main__':
    main()
