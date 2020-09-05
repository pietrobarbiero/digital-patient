import glob
import gc

import joblib
import pandas as pd
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

    sns.set_style("whitegrid")

    plt.figure(figsize=[5, 3])
    sns.kdeplot(df1A.iloc[:, 1], df3A.iloc[:, 1], cmap="Greens", shade=True, shade_lowest=False)
    # sns.kdeplot(df1B.iloc[:, 1], df3B.iloc[:, 1], cmap="Reds", shade=True, shade_lowest=False)
    sns.kdeplot(df1B.iloc[:, 1], df3B.iloc[:, 1], cmap="Reds", n_levels=5)
    sns.kdeplot(df1C.iloc[:, 1], df3C.iloc[:, 1], cmap="Reds", n_levels=5)
    sns.despine(left=True, bottom=True)
    plt.xlabel('left atrium (mmHg)')
    plt.ylabel('right atrium (mmHg)')
    plt.tight_layout()
    plt.savefig(f"./results/pla_pra.png")
    plt.savefig(f"./results/pla_pra.pdf")
    plt.show()

    plt.figure(figsize=[5, 3])
    sns.kdeplot(df2A.iloc[:, 1], df4A.iloc[:, 1], cmap="Greens", shade=True, shade_lowest=False)
    # sns.kdeplot(df2B.iloc[:, 1], df4B.iloc[:, 1], cmap="Reds", shade=True, shade_lowest=False)
    sns.kdeplot(df2B.iloc[:, 1], df4B.iloc[:, 1], cmap="Blues", n_levels=5)
    sns.kdeplot(df2C.iloc[:, 1], df4C.iloc[:, 1], cmap="Reds", n_levels=5)
    # plt.legend(["5mg", "0mg"], loc='center left', bbox_to_anchor=(1, 0.5))
    sns.despine(left=True, bottom=True)
    plt.xlabel('left ventricle (mmHg)')
    plt.ylabel('right ventricle (mmHg)')
    plt.tight_layout()
    plt.savefig(f"./results/plv_prv.png")
    plt.savefig(f"./results/plv_prv.pdf")
    plt.show()

    # plt.figure(figsize=[5, 3])
    # sns.lineplot(x=df2.iloc[:, 0], y=df1.iloc[:, 1], alpha=1)
    # sns.lineplot(x=df2.iloc[:, 0], y=df2.iloc[:, 1], alpha=1)
    # sns.lineplot(x=df2.iloc[:, 0], y=df3.iloc[:, 1], alpha=1)
    # sns.lineplot(x=df2.iloc[:, 0], y=df4.iloc[:, 1], alpha=1)
    # sns.despine(left=True, bottom=True)
    # plt.tight_layout()
    # # plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.png")
    # # plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.pdf")
    # plt.show()
    #
    # plt.figure(figsize=[5, 3])
    # # for i in range(df1.shape[1]):
    #     # if i == 2:
    #         # g = sns.scatterplot(x=df4.iloc[:, 0], y=df4.iloc[:, i], color='red', alpha=0.1)
    #         # sns.kdeplot(df2.iloc[:, i], df4.iloc[:, i], cmap="Reds", shade=True, shade_lowest=False)
    # g = sns.scatterplot(x=df2.iloc[:, 0], y=df2.iloc[:, 1], alpha=1)
    # # plt.legend(["H", "C+T", "V", "C+V", "C+V+T"], loc='center left', bbox_to_anchor=(1, 0.5))
    # # plt.title(f"{title}")
    # # plt.xlim([0, 150])
    # # plt.ylim([0, 40])
    # sns.despine(left=True, bottom=True)
    # # plt.xlabel(xlabel)
    # # plt.ylabel(ylabel)
    # plt.tight_layout()
    # # plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.png")
    # # plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.pdf")
    # plt.show()
    #
    #
    # plt.figure(figsize=[5, 3])
    # for i in range(df1.shape[1]):
    #     if i == 2:
    #         # g = sns.scatterplot(x=df4.iloc[:, 0], y=df4.iloc[:, i], color='red', alpha=0.1)
    #         sns.kdeplot(df1.iloc[:, i], df3.iloc[:, i], cmap="Reds", shade=True, shade_lowest=False)
    # # g = sns.scatterplot(x=df4.iloc[:, 1], y=df2.iloc[:, 1], alpha=1)
    # # plt.legend(["H", "C+T", "V", "C+V", "C+V+T"], loc='center left', bbox_to_anchor=(1, 0.5))
    # # plt.title(f"{title}")
    # # plt.xlim([min_y[0], max_y[0]])
    # # plt.ylim([min_y[1], max_y[1]])
    # sns.despine(left=True, bottom=True)
    # # plt.xlabel(xlabel)
    # # plt.ylabel(ylabel)
    # plt.tight_layout()
    # # plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.png")
    # # plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.pdf")
    # plt.show()
    #
    # plt.clf()
    # plt.close()
    # gc.collect()

    return


if __name__ == '__main__':
    main()
