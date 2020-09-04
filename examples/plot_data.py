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
    scaler = joblib.load(f'{result_dir}scaler.joblib')
    file_list = glob.glob(f'{result_dir}**.csv')

    f1, f2, f3, f4 = file_list[5], file_list[9], file_list[12], file_list[13]
    df = pd.read_csv(f1, index_col=0)
    df1 = (pd.read_csv(f1, index_col=0) * scaler.scale_[-2]) + scaler.mean_[-2]
    df2 = (pd.read_csv(f2, index_col=0) * scaler.scale_[-1]) + scaler.mean_[-1]
    df3 = (pd.read_csv(f3, index_col=0) * scaler.scale_[-4]) + scaler.mean_[-4]
    df4 = (pd.read_csv(f4, index_col=0) * scaler.scale_[-3]) + scaler.mean_[-3]

    sns.set_style("whitegrid")

    plt.figure()
    plt.plot(df1.iloc[:, 0], df1.iloc[:, 1])
    for i in df1.columns[2:]:
        plt.plot(df1.iloc[:, 0], df1.iloc[:, int(i)], c='r', alpha=0.2)
    # plt.title(name)
    # plt.legend()
    plt.tight_layout()
    # plt.savefig(f'{result_dir}/{name}_{j}.png')
    plt.show()

    plt.figure(figsize=[5, 3])
    sns.lineplot(x=df2.iloc[:, 0], y=df1.iloc[:, 1], alpha=1)
    sns.lineplot(x=df2.iloc[:, 0], y=df2.iloc[:, 1], alpha=1)
    sns.lineplot(x=df2.iloc[:, 0], y=df3.iloc[:, 1], alpha=1)
    sns.lineplot(x=df2.iloc[:, 0], y=df4.iloc[:, 1], alpha=1)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    # plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.png")
    # plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.pdf")
    plt.show()

    plt.figure(figsize=[5, 3])
    # for i in range(df1.shape[1]):
        # if i == 2:
            # g = sns.scatterplot(x=df4.iloc[:, 0], y=df4.iloc[:, i], color='red', alpha=0.1)
            # sns.kdeplot(df2.iloc[:, i], df4.iloc[:, i], cmap="Reds", shade=True, shade_lowest=False)
    g = sns.scatterplot(x=df2.iloc[:, 0], y=df2.iloc[:, 1], alpha=1)
    # plt.legend(["H", "C+T", "V", "C+V", "C+V+T"], loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.title(f"{title}")
    # plt.xlim([0, 150])
    # plt.ylim([0, 40])
    sns.despine(left=True, bottom=True)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    plt.tight_layout()
    # plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.png")
    # plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.pdf")
    plt.show()


    plt.figure(figsize=[5, 3])
    for i in range(df1.shape[1]):
        if i == 2:
            # g = sns.scatterplot(x=df4.iloc[:, 0], y=df4.iloc[:, i], color='red', alpha=0.1)
            sns.kdeplot(df1.iloc[:, i], df3.iloc[:, i], cmap="Reds", shade=True, shade_lowest=False)
    # g = sns.scatterplot(x=df4.iloc[:, 1], y=df2.iloc[:, 1], alpha=1)
    # plt.legend(["H", "C+T", "V", "C+V", "C+V+T"], loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.title(f"{title}")
    # plt.xlim([min_y[0], max_y[0]])
    # plt.ylim([min_y[1], max_y[1]])
    sns.despine(left=True, bottom=True)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    plt.tight_layout()
    # plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.png")
    # plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.pdf")
    plt.show()

    plt.clf()
    plt.close()
    gc.collect()

    return


if __name__ == '__main__':
    main()
