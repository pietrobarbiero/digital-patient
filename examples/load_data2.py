import pandas as pd
from scipy import interpolate
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def load_physiology(window_size=1000):
    # load simulated physiological data
    # x_ras = pd.read_csv('data/70/DKD_drug-5_glu-7_infection-0_renal-impaired.csv')
    # x_cardio = pd.read_csv('data/70/CARDIO_drug-5_glu-7_infection-0_renal-impaired.csv', index_col=0).iloc[:, :5]
    # x_ras = pd.read_csv('data/70/DKD_drug-0_glu-7_infection-0_renal-impaired.csv')
    # x_cardio = pd.read_csv('data/70/CARDIO_drug-0_glu-7_infection-0_renal-impaired.csv', index_col=0).iloc[:, :5]
    x_ras = pd.read_csv('data/70/DKD_drug-0_glu-7_infection-1_renal-impaired.csv')
    x_cardio = pd.read_csv('data/70/CARDIO_drug-0_glu-7_infection-1_renal-impaired.csv', index_col=0)
    x_cardio = x_cardio[['t', 'Pra', 'Prv', 'Pla', 'Plv', 'Ppap', 'Ppad', 'Ppa', 'Ppc', 'Ppv', 'Psa', 'Psap', 'Psc', 'Psv']]
    # x_ras = pd.read_csv('data/70/DKD_drug-5_glu-7_infection-1_renal-impaired.csv')
    # x_cardio = pd.read_csv('data/70/CARDIO_drug-5_glu-7_infection-1_renal-impaired.csv', index_col=0).iloc[:, :5]
    x_ras.drop(['angII_norm', 'IR'], axis=1, inplace=True)
    x_cardio.columns = ['t2', 'Pra', 'Prv', 'Pla', 'Plv', 'Ppap', 'Ppad', 'Ppa', 'Ppc', 'Ppv', 'Psa', 'Psap', 'Psc', 'Psv']
    tx_ras = x_ras['t']
    tx_cardio = x_cardio['t2']

    t1 = np.linspace(3, 4.99, 300)
    t2 = np.linspace(0.8, 4.6, 300)

    x_list = []
    for c in list(x_ras.columns) + list(x_cardio.columns):
        print(c)
        if c in x_ras.columns:
            f = interpolate.interp1d(tx_ras, x_ras[c].values)
            x_list.append(f(t1))
        elif c in x_cardio.columns:
            f = interpolate.interp1d(tx_cardio, x_cardio[c].values)
            x_list.append(f(t2))

    # x = np.vstack([x_angII, x_diacid, x_glu, x_diacid, x_glu]).T
    # x = np.vstack([x_angII, x_diacid, x_glu]).T
    x = np.vstack(x_list).T
    x = x.astype('float32')

    reps = 20
    x = np.tile(x.T, reps=reps).T
    t1 = np.arange(0, len(x)) / (np.max(t1) * reps)
    x[:, 0] = t1
    t2 = np.arange(0, len(x)) / (np.max(t2) * reps)
    x[:, 11] = t2

    # plt.figure()
    # plt.plot(t2, x[:, 0])
    # plt.show()

    scaler = StandardScaler()
    scaler = scaler.fit(x)
    x = scaler.transform(x)

    samples = []
    labels = []
    t_list_1 = []
    t_list_2 = []
    for batch in range(x.shape[0] - 2 * window_size + 1):
        print(f"{batch} - {batch + window_size - 2} -> {batch + window_size - 1} - {batch + 2 * window_size - 3}")
        samples.append(x[batch:batch + window_size - 2])
        labels.append(x[batch + window_size - 1:batch + 2 * window_size - 3])
        t_list_1.append(t1[batch + window_size - 1:batch + 2 * window_size - 3])
        t_list_2.append(t2[batch + window_size - 1:batch + 2 * window_size - 3])

    samples = np.array(samples)
    labels = np.array(labels)
    t_list_1 = np.array(t_list_1)
    t_list_2 = np.array(t_list_2)

    skf = KFold(n_splits=5, shuffle=True)
    trainval_index, test_index = [split for split in skf.split(samples)][0]
    skf2 = KFold(n_splits=5, shuffle=True)
    train_index, val_index = [split for split in skf2.split(np.arange(trainval_index.size))][0]
    x_train, x_val = samples[trainval_index[train_index]], samples[trainval_index[val_index]]
    y_train, y_val = labels[trainval_index[train_index]], labels[trainval_index[val_index]]
    x_test, y_test = samples[test_index], labels[test_index]
    t_list_1 = t_list_1[test_index]
    t_list_2 = t_list_2[test_index]

    addendum = {
        "RAS": [t_list_1, x_ras.columns],
        "CARDIO": [t_list_2, x_cardio.columns],
    }

    features = list(x_ras.columns) + list(x_cardio.columns)

    elist = [
        ('t', 't'), ('angI', 'angI'), ('Inhibition', 'Inhibition'), ('Renin', 'Renin'),
        ('AGT', 'AGT'), ('angII', 'angII'),
        ('diacid', 'diacid'), ('ang17', 'ang17'), ('at1r', 'at1r'), ('at2r', 'at2r'), ('ACE2', 'ACE2'),
        ('t', 'angI'), ('t', 'Inhibition'), ('t', 'Renin'), ('t', 'AGT'), ('t', 'angII'), ('t', 'diacid'),
        ('t', 'ang17'), ('t', 'at1r'), ('t', 'at2r'), ('t', 'ACE2'),
        ('AGT', 'angI'), ('Renin', 'angI'), ('angI', 'ang17'), ('angI', 'angII'), ('diacid', 'angII'),
        ('angII', 'Renin'),
        ('angII', 'ang17'), ('angII', 'at1r'), ('angII', 'at2r'), ('ACE2', 'ang17'), ('ACE2', 'angI'),

        ('t2', 't2'), ('t2', 'Pra'), ('t2', 'Prv'), ('t2', 'Pla'), ('t2', 'Plv'), ('t2', 'Ppap'), ('t2', 'Ppad'), ('t2', 'Ppa'), ('t2', 'Ppc'), ('t2', 'Ppv'), ('t2', 'Psa'),  ('t2', 'Psap'), ('t2', 'Psc'), ('t2', 'Psv'),
        ('Pra', 'Pra'), ('Pra', 'Prv'), ('Pra', 'Pla'), ('Pra', 'Plv'), ('Pra', 'Ppap'), ('Pra', 'Ppad'), ('Pra', 'Ppa'), ('Pra', 'Ppc'), ('Pra', 'Ppv'), ('Pra', 'Psa'),  ('Pra', 'Psap'), ('Pra', 'Psc'), ('Pra', 'Psv'),
        ('Prv', 'Prv'), ('Prv', 'Pla'), ('Prv', 'Plv'), ('Prv', 'Ppap'), ('Prv', 'Ppad'), ('Prv', 'Ppa'), ('Prv', 'Ppc'), ('Prv', 'Ppv'), ('Prv', 'Psa'),  ('Prv', 'Psap'), ('Prv', 'Psc'), ('Prv', 'Psv'),
        ('Pla', 'Pla'), ('Pla', 'Plv'), ('Pla', 'Ppap'), ('Pla', 'Ppad'), ('Pla', 'Ppa'), ('Pla', 'Ppc'), ('Pla', 'Ppv'), ('Pla', 'Psa'),  ('Pla', 'Psap'), ('Pla', 'Psc'), ('Pla', 'Psv'),
        ('Pra', 'Pra'), ('Pra', 'Ppap'), ('Pra', 'Ppad'), ('Pra', 'Ppa'), ('Pra', 'Ppc'), ('Pra', 'Ppv'), ('Pra', 'Psa'),  ('Pra', 'Psap'), ('Pra', 'Psc'), ('Pra', 'Psv'),
        ('Ppap', 'Ppap'), ('Ppap', 'Ppad'), ('Ppap', 'Ppa'), ('Ppap', 'Ppc'), ('Ppap', 'Ppv'), ('Ppap', 'Psa'), ('Ppap', 'Psap'), ('Ppap', 'Psc'), ('Ppap', 'Psv'),
        ('Ppad', 'Ppad'), ('Ppad', 'Ppa'), ('Ppad', 'Ppc'), ('Ppad', 'Ppv'), ('Ppad', 'Psa'), ('Ppad', 'Psap'), ('Ppad', 'Psc'), ('Ppad', 'Psv'),
        ('Ppa', 'Ppa'), ('Ppa', 'Ppc'), ('Ppa', 'Ppv'), ('Ppa', 'Psa'), ('Ppa', 'Psap'), ('Ppa', 'Psc'), ('Ppa', 'Psv'),
        ('Ppc', 'Ppc'), ('Ppc', 'Ppv'), ('Ppc', 'Psa'), ('Ppc', 'Psap'), ('Ppc', 'Psc'), ('Ppc', 'Psv'),
        ('Ppv', 'Ppv'), ('Ppv', 'Psa'), ('Ppv', 'Psap'), ('Ppv', 'Psc'), ('Ppv', 'Psv'),
        ('Psa', 'Psa'), ('Psa', 'Psap'), ('Psa', 'Psc'), ('Psa', 'Psv'),
        ('Psap', 'Psap'), ('Psap', 'Psc'), ('Psap', 'Psv'),
        ('Psc', 'Psc'), ('Psc', 'Psv'),
        ('Psv', 'Psv'),
    ]

    edge_list = []
    for edge in elist:
        i = features.index(edge[0])
        j = features.index(edge[1])
        edge_list.append((i, j))
        edge_list.append((j, i))

    # edge_list = []
    # for i in range(len(features)):
    #     for j in range(len(features)):
    #         edge_list.append((i, j))

    return x_train, y_train, x_val, y_val, x_test, y_test, edge_list, addendum, scaler
