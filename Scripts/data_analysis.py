# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool

def read_data():
    with open("../Data/netflix_3m1k/matrix.pkl", "rb") as file:
        matrix = pickle.load(file)
    with open("../Data/netflix_3m1k/lil_matrix.pkl", "rb") as file:
        lil_matrix = pickle.load(file)
    return matrix, lil_matrix

def main():
    np.set_printoptions(linewidth=120, precision=2, threshold=np.nan)
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 1200
    sns.set(style="darkgrid")

    matrix, sparse_matrix = read_data()
    non_zero_values = sparse_matrix.data

    ax = sns.countplot(matrix[matrix > 0])
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.get_figure().savefig("../Results/Netflix/Plots/Data_distribution.png")
    plt.show()

    pool = Pool()
    results = pool.map_async(np.linalg.norm, non_zero_values)
    norms_by_row = results.get()
    pool.close()
    pool.join()

    ax = sns.distplot(norms_by_row, kde=False)
    ax.set_xlabel("Norms by row")
    ax.set_ylabel("Count")

    ax.get_figure().savefig("../Results/Netflix/Plots/Row_norms_distribution.png")
    plt.show()

    pool = Pool()
    results = pool.map_async(np.linalg.norm, sparse_matrix.transpose().data)
    norms_by_column = results.get()
    pool.close()
    pool.join()

    ax = sns.distplot(norms_by_column, kde=False)
    ax.set_xlabel("Norms by column")
    ax.set_ylabel("Count")

    ax.get_figure().savefig("../Results/Netflix/Plots/Column_norms_distribution.png")
    plt.show()

if __name__ == '__main__':
    main()
