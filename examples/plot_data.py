import glob
import gc

import joblib
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from scipy import interpolate
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    result_dir = './results/patient-old1/'
    # ['t', 'angI', 'Inhibition', 'Renin', 'AGT', 'angII', 'diacid', 'ang17', 'at1r', 'at2r', 'ACE2', 't2', 'Pra', 'Prv', 'Pla', 'Plv']
    scalerA = joblib.load(f'{result_dir}scaler.joblib')
    file_list = glob.glob(f'{result_dir}**.csv')
    f1, f2, f3, f4 = file_list[5], file_list[9], file_list[12], file_list[13]
    df1A = (pd.read_csv(f1, index_col=0) * scalerA.scale_[-2]) + scalerA.mean_[-2]
    df2A = (pd.read_csv(f2, index_col=0) * scalerA.scale_[-1]) + scalerA.mean_[-1]
    df3A = (pd.read_csv(f3, index_col=0) * scalerA.scale_[-4]) + scalerA.mean_[-4]
    df4A = (pd.read_csv(f4, index_col=0) * scalerA.scale_[-3]) + scalerA.mean_[-3]

    result_dir = './results/patient-old2/'
    scalerB = joblib.load(f'{result_dir}scaler.joblib')
    file_list = glob.glob(f'{result_dir}**.csv')
    f1, f2, f3, f4 = file_list[5], file_list[9], file_list[12], file_list[13]
    df1B = (pd.read_csv(f1, index_col=0) * scalerB.scale_[-2]) + scalerB.mean_[-2]
    df2B = (pd.read_csv(f2, index_col=0) * scalerB.scale_[-1]) + scalerB.mean_[-1]
    df3B = (pd.read_csv(f3, index_col=0) * scalerB.scale_[-4]) + scalerB.mean_[-4]
    df4B = (pd.read_csv(f4, index_col=0) * scalerB.scale_[-3]) + scalerB.mean_[-3]

    result_dir = './results/patient-old3/'
    scalerC = joblib.load(f'{result_dir}scaler.joblib')
    file_list = glob.glob(f'{result_dir}**.csv')
    f1, f2, f3, f4 = file_list[5], file_list[9], file_list[12], file_list[13]
    df1C = (pd.read_csv(f1, index_col=0) * scalerC.scale_[-2]) + scalerC.mean_[-2]
    df2C = (pd.read_csv(f2, index_col=0) * scalerC.scale_[-1]) + scalerC.mean_[-1]
    df3C = (pd.read_csv(f3, index_col=0) * scalerC.scale_[-4]) + scalerC.mean_[-4]
    df4C = (pd.read_csv(f4, index_col=0) * scalerC.scale_[-3]) + scalerC.mean_[-3]

    result_dir = './results/patient-old4/'
    scalerD = joblib.load(f'{result_dir}scaler.joblib')
    file_list = glob.glob(f'{result_dir}**.csv')
    f1, f2, f3, f4 = file_list[5], file_list[9], file_list[12], file_list[13]
    df1D = (pd.read_csv(f1, index_col=0) * scalerD.scale_[-2]) + scalerD.mean_[-2]
    df2D = (pd.read_csv(f2, index_col=0) * scalerD.scale_[-1]) + scalerD.mean_[-1]
    df3D = (pd.read_csv(f3, index_col=0) * scalerD.scale_[-4]) + scalerD.mean_[-4]
    df4D = (pd.read_csv(f4, index_col=0) * scalerD.scale_[-3]) + scalerD.mean_[-3]

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=[5, 3])
    sns.kdeplot(df1A.iloc[:, 1], df3A.iloc[:, 1], cmap="Greys_r", n_levels=5, shade=True, shade_lowest=False, alpha=0.2)
    # sns.kdeplot(np.hstack(df1A.iloc[:, 2:].values), np.hstack(df3A.iloc[:, 2:].values), cmap="Greens_r", n_levels=10)
    sns.kdeplot(df1B.iloc[:, 1], df3B.iloc[:, 1], cmap="Greys_r", n_levels=5, shade=True, shade_lowest=False, alpha=0.2)
    # sns.kdeplot(np.hstack(df1C.iloc[:, 2:].values), np.hstack(df3C.iloc[:, 2:].values), cmap="Reds_r", n_levels=100)
    sns.despine(left=True, bottom=True)
    ax.annotate("predicted",
                xy=(11.7, 6.5), xycoords='data',
                xytext=(10, 8), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3", color='black'),
                )
    ax.annotate("true",
                xy=(12.5, 4.5), xycoords='data',
                xytext=(13, 3), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3", color='black'),
                )
    ax.annotate("viral infection",
                xy=(9.5, 4), xycoords='data',
                xytext=(7.5, 2), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3", color='black'),
                )
    ax.annotate("healthy state",
                xy=(7.5, 5), xycoords='data',
                xytext=(7.6, 6), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3", color='black'),
                )
    plt.xlabel('left atrium (mmHg)')
    plt.ylabel('right atrium (mmHg)')
    plt.tight_layout()
    # plt.savefig(f"./results/pla_pra_healthy_infected.png")
    # plt.savefig(f"./results/pla_pra_healthy_infected.pdf")
    plt.show()

    fig, ax = plt.subplots(figsize=[5, 3])
    sns.kdeplot(df1A.iloc[:, 1], df3A.iloc[:, 1], cmap="Greys_r", n_levels=5, shade=True, shade_lowest=False, alpha=0.2)
    sns.kdeplot(np.hstack(df1A.iloc[:, 2:].values), np.hstack(df3A.iloc[:, 2:].values), cmap="Greens_r", n_levels=10)
    sns.kdeplot(df1C.iloc[:, 1], df3C.iloc[:, 1], cmap="Greys_r", n_levels=5, shade=True, shade_lowest=False, alpha=0.2)
    sns.kdeplot(np.hstack(df1C.iloc[:, 2:].values), np.hstack(df3C.iloc[:, 2:].values), cmap="Reds_r", n_levels=100)
    sns.kdeplot(df1D.iloc[:, 1], df3D.iloc[:, 1], cmap="Greys_r", n_levels=5, shade=True, shade_lowest=False, alpha=0.2)
    sns.kdeplot(np.hstack(df1D.iloc[:, 2:].values), np.hstack(df3D.iloc[:, 2:].values), cmap="Oranges_r", n_levels=20)
    sns.despine(left=True, bottom=True)
    style = "Simple,tail_width=0.5,head_width=4,head_length=8"
    kw = dict(arrowstyle=style)
    a1 = FancyArrowPatch((7, 5.5), (11, 8), connectionstyle="arc3,rad=-0.3", color='black', **kw)
    ax.add_patch(a1)
    plt.text(8.6, 8.6, 'viral infection', ha='left', rotation=0, wrap=True)
    a2 = FancyArrowPatch((11.5, 2.8), (8, 2), connectionstyle="arc3,rad=-0.3", color='black', **kw)
    ax.add_patch(a2)
    plt.text(8.8, 0.7, 'treatment', ha='left', rotation=0, wrap=True)
    plt.ylim([0.5, 9.5])
    plt.xlabel('left atrium (mmHg)')
    plt.ylabel('right atrium (mmHg)')
    plt.tight_layout()
    plt.savefig(f"./results/pla_pra_infected.png")
    plt.savefig(f"./results/pla_pra_infected.pdf")
    plt.show()

    return


if __name__ == '__main__':
    main()
