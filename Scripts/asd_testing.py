#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import cv2
import functools
import itertools
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import ASD
import utils
from user_based import UserBased
from similarity_measures import SimilarityMeasures

def low_rank_approximation(X, rank):
    U, d, Vt = np.linalg.svd(X)
    d[rank:] = 0
    D = sp.linalg.diagsvd(d, X.shape[0], X.shape[1])
    return np.dot(np.dot(U, D), Vt)

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-a", "--algorithm", action="store", default="ASD")
    argparser.add_argument("-r", "--rank", action="store", type=int)
    argparser.add_argument("-n", "--netflix", action="store_true", default=False)
    argparser.add_argument("-d", "--data", action="store", default="boat.png")
    argparser.add_argument("-i", "--iter", action="store", type=int, default=10000)
    argparser.add_argument("-o", "--out", action="store", default="Results.csv")
    argparser.add_argument("-v", "--verbose", action="store_true", default=False)
    return argparser.parse_args()

def add_result(writer, results_dict, result, density, rank, rank_M=None):
    results_dict["Density"] = density
    results_dict["Rank_M"] = rank_M
    results_dict["Rank"] = rank
    results_dict["Num_iterations"] = result.num_iterations
    results_dict["Relative_error"] = result.residual
    results_dict["Time"] = result.time
    results_dict["RMSE"] = result.rmse

    writer.writerow(results_dict)

def main():
    args = parse_args()
    algorithm = args.algorithm

    if algorithm == "sASD":
        minimize = ASD.scaled_alternating_steepest_descent
    else: # algorithm == "ASD":
        minimize = ASD.alternating_steepest_descent

    iter_max = args.iter
    norm_tol = 1e-4
    densities = [0.1*t for t in range(1, 9)]
    ranks = [10, 25, 50, 100, 150, 250]

    # Guardar resultados
    results_prefix = "../Results/ASD/"
    with open(results_prefix + args.out, "w") as csv_file:
        fieldnames = ["Algorithm", "Density", "Rank", "Rank_M", "Shape", "Num_iterations", "Relative_error", "Time", "Data", "RMSE"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        results_dict = {}
        results_dict["Algorithm"] = algorithm
        results_dict["Data"] = "Netflix" if args.netflix else args.data

        if not args.netflix:
            image_prefix = "../Data/Images/"
            image = cv2.imread(image_prefix + args.data)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            image = cv2.resize(image, (512, 512))
            m, n = image.shape

            results_dict["Shape"] = image.shape
            if args.rank: # Fijar rango de la matriz, variar rango de la aproximación
                low_rank_image = low_rank_approximation(image, args.rank)

                for rank, density in itertools.product(ranks, densities):
                    mask = np.random.choice([0, 1], (m, n), p=[1-density, density])
                    masked_image = mask*low_rank_image
                    result = minimize(masked_image, rank, mask, iter_max, norm_tol, verbose=args.verbose)
                    add_result(writer, results_dict, result, density, rank, rank_M=args.rank)

            else: # Variar rango de la matriz y de la aproximación
                for rank, density in itertools.product(ranks, densities):
                    low_rank_image = low_rank_approximation(image, rank)
                    mask = np.random.choice([0, 1], (m, n), p=[1-density, density])
                    masked_image = mask*low_rank_image
                    result = minimize(masked_image, rank, mask, iter_max, norm_tol, verbose=args.verbose)
                    add_result(writer, results_dict, result, density, rank, rank_M=rank)
        else:
            data = utils.read_netflix_data()
            mask = (data != 0)
            density = 100*mask.sum()/data.size
            m, n = data.shape

            results_dict["Shape"] = data.shape

            if args.rank:
                ranks = [args.rank]

            for rank in ranks:
                result = minimize(data, rank, mask, iter_max, norm_tol, verbose=args.verbose)
                add_result(writer, results_dict, result, density, rank)

if __name__ == "__main__":
    main()


