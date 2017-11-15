# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
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

    pool = Pool()
    mean_by_row = pool.map(np.mean, non_zero_values)
    mean_by_row = np.array(mean_by_row)
    is_finite = np.isfinite(mean_by_row)
    mean_by_row = mean_by_row[is_finite]
    pool.close()
    pool.join()

    pool = Pool()
    var_by_row = pool.map(np.var, non_zero_values)
    var_by_row = np.array(var_by_row)
    var_by_row = var_by_row[is_finite]
    pool.close()
    pool.join()

    ax = sns.regplot(x=mean_by_row, y=var_by_row, fit_reg=False)
    ax.set_xlabel("Mean by row")
    ax.set_ylabel("Variance by row")
    ax.get_figure().savefig("../Results/Netflix/Plots/Mean_Var_Rows.png")
    plt.show()

    ax = sns.distplot(mean_by_row, kde=False)
    ax.set_title("Mean by row")
    ax.get_figure().savefig("../Results/Netflix/Plots/Hist_Mean_Rows.png")
    plt.show()

    ax = sns.distplot(var_by_row, kde=False)
    ax.set_title("Var by row")
    ax.get_figure().savefig("../Results/Netflix/Plots/Hist_Var_Rows.png")
    plt.show()

    pool = Pool()
    mean_by_col = pool.map(np.mean, sparse_matrix.transpose().data)
    mean_by_col = np.array(mean_by_col)
    is_finite = np.isfinite(mean_by_col)
    mean_by_col = mean_by_col[is_finite]
    pool.close()
    pool.join()

    pool = Pool()
    var_by_col = pool.map(np.var, sparse_matrix.transpose().data)
    var_by_col = np.array(var_by_col)
    var_by_col = var_by_col[is_finite]
    pool.close()
    pool.join()

    ax = sns.regplot(x=mean_by_col, y=var_by_col, fit_reg=False)
    ax.set_xlabel("Mean by col")
    ax.set_ylabel("Variance by col")
    ax.get_figure().savefig("../Results/Netflix/Plots/Mean_Var_Cols.png")
    plt.show()

    ax = sns.distplot(mean_by_col, kde=False)
    ax.set_title("Mean by col")
    ax.get_figure().savefig("../Results/Netflix/Plots/Hist_Mean_Cols.png")
    plt.show()

    ax = sns.distplot(var_by_col, kde=False)
    ax.set_title("Var by col")
    ax.get_figure().savefig("../Results/Netflix/Plots/Hist_Var_Cols.png")
    plt.show()


if __name__ == '__main__':
    main()
