# -*- coding: utf-8 -*-

import numpy as np
import sys
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import utils
import ASD

def main():
    matplotlib.rcParams["figure.dpi"] = 200
    matplotlib.rcParams["savefig.dpi"] = 600
    sns.set(style="darkgrid")

    if len(sys.argv) > 1:
        algorithm = sys.argv[1]
    else:
        algorithm = "ASD"

    data = utils.read_netflix_data()
    mask = (data != 0)

    print("data shape:", data.shape)
    print("data density:", mask.sum()/data.size)

    rank = 50
    iter_max = 10000
    norm_tol = 1e-4

    if algorithm == "sASD":
        minimize = ASD.scaled_alternating_steepest_descent
    else: # ASD
        minimize = ASD.alternating_steepest_descent

    results = minimize(data, rank, mask, iter_max, norm_tol, verbose=True)
    completed_data = results.matrix

    rmse = utils.rmse(data, completed_data, mask)

    print("RMSE:", rmse)

if __name__ == '__main__':
    main()
