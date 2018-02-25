# -*- coding: utf-8 -*-

import argparse
import cv2
import functools
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import time
import csv
from matrix_completion import *
from utils import *
from scipy.linalg import svd, diagsvd
from collections import namedtuple

matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.dpi"] = 600

densities = [0.4, 0.5]
ranks = [50, 60]

delta_tol = 0.1
error_tol = 1e-4

image_prefix = "../Data/Images/"
results_prefix = "../Results/Binary/"
normalize = functools.partial(cv2.normalize, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
args = None

def parse_args():
    default_image = "pexels-photo-462358.jpeg"
    argparser = argparse.ArgumentParser()

    argparser.add_argument("-a", "--algorithm", action="store", default="nuclear")
    argparser.add_argument("-v", "--verbose", action="store_true", default=False)
    argparser.add_argument("-s", "--show", action="store_true", default=False)
    argparser.add_argument("-i", "--iter", action="store", type=int, default=100)
    argparser.add_argument("-w", "--write", action="store_true", default=False)
    argparser.add_argument("--image", action="store", default=default_image)

    global args
    args = argparser.parse_args()

def load_image(image_name):
    image_path = image_prefix + image_name
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = normalize(image)
    return image

def process_image(image_name, rank, density):
    image = load_image(image_name)
    low_rank_image = normalize(low_rank_approximation(image, rank))

    mask = np.random.choice([0, 1], image.shape, p=[1-density, density])
    train_mask = np.random.choice([0, 1], image.shape, p=[0.2, 0.8]) * mask
    test_mask = np.random.choice([0, 1], image.shape, p=[0.8, 0.2]) * mask

    start_time = time.time()
    result = matrix_completion(low_rank_image*train_mask, train_mask, args.iter, delta_tol, error_tol, args.algorithm, args.verbose)
    end_time = time.time()

    train_error = relative_error(result.X, low_rank_image, train_mask)
    test_error = relative_error(result.X, low_rank_image, test_mask)
    full_error = relative_error(result.X, low_rank_image)
    rank_result = np.linalg.matrix_rank(result.X)
    total_time = end_time - start_time

    result_dict = dict()
    result_dict["Algorithm"] = "Binary_" + args.algorithm
    result_dict["Density"] = density
    result_dict["Real_rank"] = rank
    result_dict["Approx_rank"] = rank_result
    result_dict["Train_error"] = train_error
    result_dict["Test_error"] = test_error
    result_dict["Full_error"] = full_error
    result_dict["Time"] = total_time
    result_dict["Image"] = image_name
    result_dict["Shape"] = image.shape

    if args.verbose:
        if args.algorithm == "nuclear":
            delta_real = sum(np.linalg.svd(low_rank_image, 0, 0))
        elif args.algorithm == "spectral":
            delta_real = np.max(np.linalg.svd(low_rank_image, False, False))

        print("Real delta:", delta_real)
        print("Best delta:", result.delta)
        print("Train Error:", train_error)
        print("Test Error:", test_error)
        print("Full Error:", full_error)
        print("Rank(result):", rank_result)
        print("Time:", total_time)

    if args.show:
        plt.figure(1)
        plt.imshow(low_rank_image*train_mask, interpolation="nearest", cmap=matplotlib.cm.gray)
        plt.colorbar()

        plt.figure(2)
        plt.imshow(result.X, interpolation="nearest", cmap=matplotlib.cm.gray)
        plt.colorbar()

        # plt.figure(3)
        # plt.imshow(result.X, interpolation="nearest", cmap=matplotlib.cm.coolwarm)
        # plt.colorbar()

        plt.show()

    if args.write:
        cv2.imwrite(results_prefix + "Images/result_" + str(rank) + "_" + str(density) + ".png", np.clip(result.X, 0, 255))
        cv2.imwrite(results_prefix + "Images/train_masked.png", normalize(low_rank_image*train_mask))

    return result_dict

def main():
    parse_args()

    fieldnames = ["Algorithm", "Density", "Real_rank", "Approx_rank", "Train_error", "Test_error", "Full_error", "Time", "Image", "Shape"]

    with open(results_prefix + "Binary_results.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for rank, density in itertools.product(ranks, densities):
            results_dict = process_image(args.image, rank, density)
            writer.writerow(results_dict)

if __name__ == '__main__':
    main()
