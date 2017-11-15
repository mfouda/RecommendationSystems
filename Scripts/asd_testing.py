#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import cv2
import functools
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import ASD
from user_based import UserBased
from similarity_measures import SimilarityMeasures

def low_rank_approximation(X, rank):
    U, d, Vt = np.linalg.svd(X)
    d[rank:] = 0
    D = sp.linalg.diagsvd(d, X.shape[0], X.shape[1])
    return np.dot(np.dot(U, D), Vt)

def main():
    if len(sys.argv) > 1:
        algorithm = sys.argv[1]
    else:
        algorithm = "ASD"


    image = cv2.imread("../Data/Images/boat.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Cambiar tamaño
    image = cv2.resize(image, (512, 512))
    m, n = image.shape

    # Parámetros
    iter_max = 100000
    norm_tol = 1e-4

    ranks = [25, 50, 100, 150, 250]
    densities = [0.1*t for t in range(1, 9)]

    # Guardar resultados
    with open("../Results/Noisy_Images/Results.csv", "w") as csv_file:
        fieldnames = ["Algorithm", "Density", "Rank", "Shape", "Num_iterations", "Relative_error", "Time"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        results_dict = {}
        results_dict["Algorithm"] = algorithm
        results_dict["Shape"] = image.shape

        for rank, density in itertools.product(ranks, densities):
            # Aproximación de rango bajo
            low_rank_image = low_rank_approximation(image, rank)

            # Máscara
            mask = np.random.choice([0, 1], (m, n), p=[1-density, density])
            masked_image = mask*low_rank_image

            if algorithm == "sASD":
                minimize = ASD.scaled_alternating_steepest_descent
            else: # algorithm == "ASD":
                minimize = ASD.alternating_steepest_descent

            result = minimize(masked_image, rank, mask, iter_max, norm_tol, verbose=False)

            results_dict["Density"] = density
            results_dict["Rank"] = rank
            results_dict["Num_iterations"] = result.num_iterations
            results_dict["Relative_error"] = result.residual
            results_dict["Time"] = result.time

            writer.writerow(results_dict)

if __name__ == "__main__":
    main()


