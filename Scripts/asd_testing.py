#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import cv2
import glob
import functools
import itertools
import os

import numpy as np
import scipy as sp

import ASD
import utils

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

normalize = functools.partial(cv2.normalize, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

image_prefix = "../Data/Images/"
results_prefix = "../Results/ASD_Testing/"

# Default values
norm_tol = 1e-3

# densities = [0.1*t for t in range(1, 9)]
# ranks = [10, 25, 50, 100, 150, 250]

densities = [0.2, 0.3, 0.4, 0.5]
ranks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
real_ranks = [25, 50, 100, 200]

def parse_args():
    default_image = "pexels-photo-462358.jpeg"
    argparser = argparse.ArgumentParser()

    argparser.add_argument("-a", "--algorithm", action="store", default="ASD")
    argparser.add_argument("-r", "--rank", action="store", type=int)
    argparser.add_argument("-d", "--density", action="store", type=int)
    argparser.add_argument("-o", "--out", action="store", default="Results_ASD.csv")
    argparser.add_argument("-v", "--verbose", action="store_true", default=False)
    argparser.add_argument("-w", "--write", action="store_true", default=False)
    argparser.add_argument("-i", action="store", type=int, default=5000)

    argparser.add_argument("--realrank", action="store", type=int)
    argparser.add_argument("--labels", action="store_true", default=False)

    argparser.add_argument("--image", action="store", default=default_image)
    argparser.add_argument("--allimages", action="store_true", default=False)

    return argparser.parse_args()

def add_result(writer, results_dict, result, density, real_rank, rank):
    results_dict["Density"] = density
    results_dict["Real_rank"] = real_rank
    results_dict["Rank"] = rank
    results_dict["Num_iterations"] = result.num_iterations
    results_dict["Time"] = result.time

    writer.writerow(results_dict)

def approximate_image(image_name, writer, results_dict):
    image_path = image_prefix + image_name

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.resize(image, (512, 512))
    image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    results_dict["Shape"] = image.shape
    results_dict["Data"] = image_name

    for real_rank in real_ranks:
        path_prefix = results_prefix + image_name[:-5] + "__" + str(real_rank) + "_"
        low_rank_image = utils.low_rank_approximation(image, real_rank)

        if write:
            cv2.imwrite(path_prefix + "low_rank.png", normalize(low_rank_image))

        for density in densities:
            mask = np.random.choice([0, 1], image.shape, p=[1-density, density])
            train_mask = np.random.choice([0, 1], image.shape, p=[0.2, 0.8]) * mask
            test_mask = np.random.choice([0, 1], image.shape, p=[0.8, 0.2]) * mask

            masked_image = train_mask*low_rank_image

            if write:
                masked_image_green = cv2.cvtColor(masked_image.astype(np.float32), cv2.COLOR_GRAY2RGB)
                masked_image_green[:,:,1][masked_image_green[:,:,1] == 0] = 1.0
                cv2.imwrite(path_prefix + str(int(100*density)) + "_masked.png", normalize(masked_image_green))

            for rank in ranks:
                result = minimize(masked_image, rank, train_mask, iter_max, norm_tol, verbose=verbose)

                results_dict["Train_error"] = utils.relative_error(result.matrix, low_rank_image, train_mask)
                results_dict["Test_error"] = utils.relative_error(result.matrix, low_rank_image, test_mask)
                results_dict["Full_error"] = utils.relative_error(result.matrix, low_rank_image)
                results_dict["Relative_error"] = result.residual

                if(verbose):
                    print("Train error:", results_dict["Train_error"])
                    print("Test error:", results_dict["Test_error"])
                    print("Full error:", results_dict["Full_error"])
                    print("Relative error:", results_dict["Relative_error"])

                add_result(writer, results_dict, result, density, real_rank, rank)

                if write:
                    result_matrix = cv2.cvtColor(result.matrix.astype(np.float32), cv2.COLOR_GRAY2RGB)

                    if labels:
                        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                        label_text = "Rank: " + str(rank) + "   Error: " + str(round(results_dict["Test_error"], 3))
                        cv2.putText(result_matrix, label_text, (200, 500), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    cv2.imwrite(path_prefix + str(rank) + "_" + str(int(100*density)) + "_approx.png", normalize(np.clip(result_matrix, 0, 1)))

def main():
    args = parse_args()

    algorithm = args.algorithm
    iter_max = args.iter
    verbose = args.verbose
    write = args.write
    labels = args.labels

    if args.rank:
        ranks = [args.rank]
    if args.realrank:
        real_ranks = [args.realrank]
    if args.density:
        densities = [args.density/100]

    if algorithm == "sASD":
        minimize = ASD.scaled_alternating_steepest_descent
    else: # algorithm == "ASD":
        minimize = ASD.alternating_steepest_descent

    with open(results_prefix + args.out, "w", newline="") as csv_file:
        fieldnames = ["Algorithm", "Density", "Real_rank", "Rank", "Shape", "Num_iterations", "Train_error", "Test_error", "Full_error", "Relative_error", "Time", "Data"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        results_dict = {}
        results_dict["Algorithm"] = algorithm

        if args.netflix:
            approximate_netflix_matrix(writer, results_dict)
        elif args.allimages:
            image_names = [os.path.basename(file_path) for file_path in glob.glob(image_prefix + "/*.jpeg")]
            for image_name in image_names:
                approximate_image(image_name, writer, results_dict)
        elif args.image:
            image_name = args.image
            approximate_image(image_name, writer, results_dict)

if __name__ == "__main__":
    main()
