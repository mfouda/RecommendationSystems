# -*- coding: utf-8 -*-

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

from svd_methods import *
from evaluation_measures import EvaluationMeasures

def read_data():
    with open("../Data/netflix_3m1k/matrix.pkl", "rb") as file:
        matrix = pickle.load(file)
    return matrix

def main():
    np.random.seed(351243)
    np.set_printoptions(linewidth=120, precision=2, threshold=np.nan)

    # Parameters
    max_iter = int(1e4)
    norm_tol=1e-2
    step_size = 1.0
    regularization_parameter = 1.0
    rank = 50

    # Data
    use_netflix_data = False

    if use_netflix_data:
        data = read_data()
        real_data = data.copy()
        mask = np.int8(data > 0)
    else:
        m = 200
        n = 250
        prob = 0.2 # Probability of unobserved values
        data = np.random.randint(1, 6, (m, n))
        data = low_rank_approximation(data, rank)
        real_data = data.copy()
        mask = np.random.choice([0, 1], (m, n), p=[prob, 1-prob])
        data = mask*data

    m, n = data.shape

    # Initial values
    x0 = np.random.random_integers(0, 6, (m, rank)).astype("float64")
    y0 = np.random.random_integers(0, 6, (rank, n)).astype("float64")

    # Alternating Steepest Descent
    # set_data(data, mask)
    train_indexes, test_indexes = create_sets(data, mask, train_percentage=80, test_percentage=20)

    time1 = time.time()
    predicted_data, residuals = alternating_steepest_descent(x0, y0, data, mask, max_iter, norm_tol)
    print("(TIME) ASD:", time.time() - time1)

    rmse = EvaluationMeasures.root_mean_square_error(real_data, predicted_data, test_indexes)
    print("(RMSE) ASD:", rmse)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=150)
    ax.semilogy(residuals, linewidth=2.0, linestyle="-", marker="o")
    fig.tight_layout()
    plt.savefig("../Results/ASD_Plot.png", bbox_inches="tight", pad_inches=0)

    # Regularized SVD

    # time1 = time.time()
    # predicted_data, residuals = regularized_x0, y0, max_iter, norm_tol, step_size, regularization_parameter)
    # print("(TIME) Regularized:", time.time() - time1)
    #
    # rmse = EvaluationMeasures.root_mean_square_error(real_data, predicted_data, SVDMethods.test_indexes)
    # print("(RMSE) Regularized:", rmse)
    #
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=150)
    # ax.semilogy(residuals, linewidth=2.0, linestyle="-", marker="o")
    # fig.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
