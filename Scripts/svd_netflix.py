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
    mask = (data != 0).astype(np.uint8)

    data[~mask] = data[mask].mean() # Llenar la información desconocida con la media de la conocida

    rank = 800
    iter_max = 10000
    norm_tol = 1e-4

    if algorithm == "sASD":
        completed_data, residuals = ASD.scaled_alternating_steepest_descent(data, rank, mask, iter_max, norm_tol)
    else: # ASD
        completed_data, residuals = ASD.alternating_steepest_descent(data, rank, mask, iter_max, norm_tol)

    rmse = utils.rmse(data, completed_data, mask)
    print("RMSE:", rmse)

    sns.distplot(completed_data.flatten(), kde=False)
    plt.show()

if __name__ == '__main__':
    main()
