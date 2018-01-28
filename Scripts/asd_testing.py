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

class constants:
    image_prefix = "../Data/Images/"
    results_prefix = "../Results/ASD_Testing/"

    # Default values
    norm_tol = 1e-3

    # densities = [0.1*t for t in range(1, 9)]
    # ranks = [10, 25, 50, 100, 150, 250]

    densities = [0.4]
    ranks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    real_ranks = [50]

def parse_args():
    default_image = "pexels-photo-462358.jpeg"
    argparser = argparse.ArgumentParser()

    argparser.add_argument("-a", "--algorithm", action="store", default="ASD")
    argparser.add_argument("-r", "--rank", action="store", type=int)
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
    results_dict["Rank"] = rank
    results_dict["Num_iterations"] = result.num_iterations
    results_dict["Time"] = result.time

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
    image_path = constants.image_prefix + image_name

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.resize(image, (512, 512))
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

            for rank in constants.ranks:
                result = constants.minimize(masked_image, rank, train_mask, constants.iter_max, constants.norm_tol, verbose=constants.verbose)

                results_dict["Train_error"] = utils.relative_error(result.matrix, low_rank_image, train_mask)
                results_dict["Test_error"] = utils.relative_error(result.matrix, low_rank_image, test_mask)
                results_dict["Full_error"] = utils.relative_error(result.matrix, low_rank_image)

                add_result(writer, results_dict, result, density, real_rank, rank)

                if constants.write:
                    result_matrix = cv2.cvtColor(result.matrix.astype(np.float32), cv2.COLOR_GRAY2RGB)

                    if constants.labels:
                        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                        label_text = "Rank: " + str(rank) + "   Error: " + str(round(results_dict["Test_error"], 3))
                        cv2.putText(result_matrix, label_text, (200, 500), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    cv2.imwrite(path_prefix + str(rank) + "_" + str(int(100*density)) + "_approx.png", normalize(np.clip(result_matrix, 0, 1)))

                # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                # ax.set_aspect("equal")
                # plt.imshow(result.matrix, interpolation="nearest", cmap=matplotlib.cm.coolwarm)
                # plt.colorbar()
                # plt.show()


def main():
    args = parse_args()

    constants.algorithm = args.algorithm
    constants.iter_max = args.iter
    constants.verbose = args.verbose
    constants.write = args.write
    constants.labels = args.labels

    if args.rank:
        constants.ranks = [args.rank]
    if args.realrank:
        constants.real_ranks = [args.realrank]
    if args.density:
        constants.densities = [args.density/100]

    if constants.algorithm == "sASD":
        constants.minimize = ASD.scaled_alternating_steepest_descent
    else: # algorithm == "ASD":
        constants.minimize = ASD.alternating_steepest_descent

    with open(constants.results_prefix + args.out, "w", newline="") as csv_file:
        fieldnames = ["Algorithm", "Density", "Real_rank", "Rank", "Shape", "Num_iterations", "Train_error", "Test_error", "Full_error", "Time", "Data"]
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
