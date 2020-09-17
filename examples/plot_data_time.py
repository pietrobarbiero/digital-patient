import glob
import gc

import joblib
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from scipy import interpolate
import numpy as np
from sklearn.decomposition import PCA
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

    sns.set_style("whitegrid")

    plt.figure(figsize=[5, 3])
    sns.lineplot(np.repeat(df3A.iloc[:, 0]*3-27, repeats=5), np.hstack(df2A.iloc[:, 2:].values), label='left ventricle')
    sns.lineplot(np.repeat(df3A.iloc[:, 0]*3-27, repeats=5), np.hstack(df4A.iloc[:, 2:].values), label='right ventricle')
    sns.despine(left=True, bottom=True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.xlim([0, 14])
    plt.ylim([0, 150])
    plt.xlabel('time (s)')
    plt.ylabel('pressure (mmHg)')
    plt.tight_layout()
    plt.savefig(f"./results/ventricle_time.png")
    plt.savefig(f"./results/ventricle_time.pdf")
    plt.show()

    plt.figure(figsize=[5, 3])
    sns.lineplot(np.repeat(df3A.iloc[:, 0]*3-27, repeats=5), np.hstack(df1A.iloc[:, 2:].values), label='left atrium')
    sns.lineplot(np.repeat(df3A.iloc[:, 0]*3-27, repeats=5), np.hstack(df3A.iloc[:, 2:].values), label='right atrium')
    sns.despine(left=True, bottom=True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.xlim([0, 14])
    plt.ylim([1, 8])
    plt.xlabel('time (s)')
    plt.ylabel('pressure (mmHg)')
    plt.tight_layout()
    plt.savefig(f"./results/atrium_time.png")
    plt.savefig(f"./results/atrium_time.pdf")
    plt.show()

    return


if __name__ == '__main__':
    main()
