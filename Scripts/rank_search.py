#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import cv2
import glob
import functools
import itertools
import os
import time

import numpy as np
import scipy as sp

import ASD
import utils

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from binary_search import binary_search

normalize = functools.partial(cv2.normalize, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

class constants:
    image_prefix = "../Data/Images/"
    results_prefix = "../Results/Rank_Search/"

    norm_tol = 1e-3

    densities = [0.5, 0.6, 0.7]
    real_ranks = np.random.randint(30, 200, 15)

def parse_args():
    default_image = "pexels-photo-685526.jpeg"
    argparser = argparse.ArgumentParser()

    argparser.add_argument("-a", "--algorithm", action="store", default="ASD")
    argparser.add_argument("-d", "--density", action="store", type=int)
    argparser.add_argument("-i", "--image", action="store", default=default_image)
    argparser.add_argument("-o", "--out", action="store", default="Results_ASD.csv")
    argparser.add_argument("-v", "--verbose", action="store_true", default=False)
    argparser.add_argument("-w", "--write", action="store_true", default=False)

    argparser.add_argument("--iter", action="store", type=int, default=5000)
    argparser.add_argument("--realrank", action="store", type=int)
    argparser.add_argument("--labels", action="store_true", default=False)

    argparser.add_argument("--netflix", action="store_true", default=False)
    argparser.add_argument("--allimages", action="store_true", default=False)

    return argparser.parse_args()

def add_result(writer, results_dict, result, density, real_rank, rank):
    results_dict["Density"] = density
    results_dict["Real_rank"] = real_rank
    results_dict["Best_rank"] = rank

    writer.writerow(results_dict)

def approximate_netflix_matrix(writer, results_dict):
    data = utils.read_netflix_data()
    mask = (data != 0)
    density = 100*mask.sum()/data.size

    results_dict["Shape"] = data.shape
    results_dict["Data"] = "Netflix"

    for rank in constants.ranks:
        result = constants.minimize(data, rank, mask, constants.iter_max, constants.norm_tol, verbose=constants.verbose)
        add_result(writer, results_dict, result, density, rank)

def approximate_image(image_name, writer, results_dict):
    minimize_result = None
    low_rank_image = None
    test_mask = None
    result = None

    def minimize_rank_aux(rank, z0, mask, max_iter, norm_tol, verbose):
        nonlocal result
        result = constants.minimize(z0, rank, mask, max_iter, norm_tol, verbose)
        test_error = utils.relative_error(result.matrix, low_rank_image, test_mask)
        return test_error
    
    image_path = constants.image_prefix + image_name

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    results_dict["Shape"] = image.shape
    results_dict["Data"] = image_name

    for real_rank in constants.real_ranks:
        path_prefix = constants.results_prefix + image_name[:-5] + "__" + str(real_rank) + "_"
        low_rank_image = utils.low_rank_approximation(image, real_rank)

        if constants.write:
            cv2.imwrite(path_prefix + "low_rank.png", normalize(low_rank_image))

        for density in constants.densities:
            mask = np.random.choice([0, 1], image.shape, p=[1-density, density])
            train_mask = np.random.choice([0, 1], image.shape, p=[0.2, 0.8]) * mask
            test_mask = np.random.choice([0, 1], image.shape, p=[0.8, 0.2]) * mask

            masked_image = train_mask*low_rank_image

            if constants.write:
                masked_image_green = cv2.cvtColor(masked_image.astype(np.float32), cv2.COLOR_GRAY2RGB)
                masked_image_green[:,:,1][masked_image_green[:,:,1] == 0] = 1.0
                cv2.imwrite(path_prefix + str(int(100*density)) + "_masked.png", normalize(masked_image_green))

            minimize_rank = functools.partial(minimize_rank_aux, z0=masked_image, mask=train_mask, max_iter=constants.iter_max, norm_tol=constants.norm_tol, verbose=constants.verbose)

            low, high = 30, 200

            start = time.time()
            best_rank, num_eval = binary_search(low, high, minimize_rank)
            results_dict["Time"] = time.time() - start

            results_dict["Num_eval"] = num_eval
            results_dict["Train_error"] = utils.relative_error(result.matrix, low_rank_image, train_mask)
            results_dict["Test_error"] = utils.relative_error(result.matrix, low_rank_image, test_mask)
            results_dict["Full_error"] = utils.relative_error(result.matrix, low_rank_image)
            results_dict["Relative_error"] = result.residual

            if(constants.verbose):
                print("Best rank:", best_rank)
                print("Train error:", results_dict["Train_error"])
                print("Test error:", results_dict["Test_error"])
                print("Full error:", results_dict["Full_error"])
                print("Relative error:", results_dict["Relative_error"])

            add_result(writer, results_dict, result, density, real_rank, best_rank)

def main():
    args = parse_args()

    constants.algorithm = args.algorithm
    constants.iter_max = args.iter
    constants.verbose = args.verbose
    constants.write = args.write
    constants.labels = args.labels

    if args.realrank:
        constants.real_ranks = [args.realrank]
    if args.density:
        constants.densities = [args.density/100]

    if constants.algorithm == "sASD":
        constants.minimize = ASD.scaled_alternating_steepest_descent
    else: # algorithm == "ASD":
        constants.minimize = ASD.alternating_steepest_descent

    with open(constants.results_prefix + args.out, "w", newline="") as csv_file:
        fieldnames = ["Algorithm", "Density", "Real_rank", "Best_rank", "Num_eval", "Shape", "Train_error", "Test_error", "Full_error", "Relative_error", "Time", "Data"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        results_dict = {}
        results_dict["Algorithm"] = constants.algorithm

        if args.netflix:
            approximate_netflix_matrix(writer, results_dict)
        elif args.allimages:
            image_names = [os.path.basename(file_path) for file_path in glob.glob(constants.image_prefix + "/*.jpeg")]
            for image_name in image_names:
                approximate_image(image_name, writer, results_dict)
        elif args.image:
            image_name = args.image
            approximate_image(image_name, writer, results_dict)

if __name__ == "__main__":
    main()

