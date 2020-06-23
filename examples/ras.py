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
    x_ras = pd.read_csv('data/70/DKD_drug-5_glu-10_infection-0_renal-normal.csv')
    x_diabetes = pd.read_csv('data/70/DIABETES_glu-5.csv')
    # x_cardio = pd.read_csv('data/70/CARDIO_drug-5_glu-10_infection-0_renal-impaired.csv')

    tx_ras = x_ras['t']
    x_ras = x_ras[['angII', 'diacid']].values.astype('float32')
    tx_diabetes = x_diabetes['t']
    x_diabetes = x_diabetes[['G', 'I']].values.astype('float32')
    tx_ras2 = tx_diabetes[tx_diabetes<tx_ras.max()]
    x_diabetes = x_diabetes[tx_diabetes<tx_ras.max()]

    f_angII = interpolate.interp1d(tx_ras, x_ras[:, 0])
    x_angII = f_angII(tx_ras2)
    f_diacid = interpolate.interp1d(tx_ras, x_ras[:, 1])
    x_diacid = f_diacid(tx_ras2)

    x = np.vstack([x_angII, x_diacid, x_diabetes[:, 0]]).T
    # x = np.vstack([x_angII, x_diacid]).T
    x = x.astype('float32')

    x = StandardScaler().fit_transform(x)

    window_size = 10
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

    dp.train(x_train, y_train, epochs=3, lr=0.01)
    predictions = dp.predict(x_val)

    t = np.arange(0, len(y_val)) / tx_ras2.max()
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=[10, 4])

    plt.subplot(131)
    plt.plot(t, y_val[:, 0], )
    plt.scatter(t, predictions[:, 0], alpha=0.5, c='orange')
    plt.ylabel('[ANG II]')
    plt.xlabel('t [sec]')

    plt.subplot(132)
    plt.plot(t, y_val[:, 1], label='True dynamics')
    plt.scatter(t, predictions[:, 1], label='GNN predictions', alpha=0.5, c='orange')
    plt.ylabel('[diacid]')
    plt.xlabel('t [sec]')

    # plt.subplot(133)
    # plt.plot(t, y_val[:, 2], label='True dynamics')
    # plt.scatter(t, predictions[:, 1], label='GNN predictions', alpha=0.5, c='orange')
    # plt.ylabel('[glucose]')
    # plt.xlabel('t [sec]')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
    plt.tight_layout()
    plt.savefig('RAS.png')
    plt.show()

    return


if __name__ == '__main__':
    main()
