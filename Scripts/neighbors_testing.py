# -*- coding: utf-8 -*-

import numpy as np
import functools
import argparse
import time
import csv
import pickle
import itertools
import cv2

import utils

from user_based import UserBased
from item_based import ItemBased
from similarity_measures import SimilarityMeasures

normalize = functools.partial(cv2.normalize, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

class constants:
    image_prefix = "../Data/Images/"
    results_prefix = "../Results/Neighbors_Testing/"

    densities = [0.1*t for t in range(1, 9)]
    ranks = [10, 25, 50, 100, 150, 250]
    numbers_of_neighbors = [10, 25, 50, 100]

    train_percentage = 80
    test_percentage = 20

    similarity_measures = {
            "pearson_correlation" : SimilarityMeasures.pearson_correlation,
            "cosine_similarity" : SimilarityMeasures.cosine_similarity,
            "mean_squared_difference" : SimilarityMeasures.mean_squared_difference,
            "mean_absolute_difference" : SimilarityMeasures.mean_absolute_difference
            }

    methods = {
            "User_based" : UserBased,
            "Item_based" : ItemBased
            }

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.resize(image, (512, 512))
    image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return image

def approximate_image(image, writer, results_dict):
    results_dict["Shape"] = image.shape
    
    for rank, density in itertools.product(constants.ranks, constants.densities):
        results_dict["Density"] = density
        results_dict["Rank"] = rank
        low_rank_image = utils.low_rank_approximation(image, rank)

        for method_name in constants.methods:
            results_dict["Algorithm"] = method_name

            method = constants.methods[method_name]
            method.set_data(low_rank_image)
            method.create_sets(constants.train_percentage, constants.test_percentage)

            for similarity_name, number_of_neighbors in itertools.product(constants.similarity_measures, constants.numbers_of_neighbors):

                results_dict["Similarity"] = similarity_name
                results_dict["Number_of_neighbors"] = number_of_neighbors

                similarity = constants.similarity_measures[similarity_name]

                time1 = time.time()
                predicted_image = method.similarity_prediction(similarity, number_of_neighbors)
                time2 = time.time()
                results_dict["Time"] = time2 - time1

                results_dict["Train_error"] = utils.relative_error(predicted_image, low_rank_image, method.train_mask)
                results_dict["Test_error"] = utils.relative_error(predicted_image, low_rank_image, method.test_mask)
                results_dict["Full_error"] = utils.relative_error(predicted_image, low_rank_image)

                writer.writerow(results_dict)

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-a", "--algorithm", action="store", default="ASD")
    argparser.add_argument("-r", "--rank", action="store", type=int)
    argparser.add_argument("-d", "--data", action="store", default="boat.png")
    argparser.add_argument("-i", "--iter", action="store", type=int, default=10000)
    argparser.add_argument("-o", "--out", action="store", default="Results_neighbors.csv")
    argparser.add_argument("-v", "--verbose", action="store_true", default=False)
    argparser.add_argument("--netflix", action="store_true", default=False)
    argparser.add_argument("--allimages", action="store_true", default=False)
    return argparser.parse_args()

def main():
    args = parse_args()
    use_netflix_data = args.netflix

    train_percentage = constants.train_percentage
    test_percentage = constants.test_percentage

    results_dict = {}

    with open(constants.results_prefix + args.out, "w", newline="") as csv_file:
        fieldnames = ["Algorithm", "Density", "Rank", "Shape", "Num_iterations", "Train_error", "Test_error", "Full_error", "Time", "Data", "Similarity", "Number_of_neighbors"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        if use_netflix_data:
            data = utils.read_netflix_data()
            real_data = data.copy()
        elif args.allimages:
            image_names = [os.path.basename(file_path) for file_path in glob.glob(constants.image_prefix + "/*.jpeg")]
            for image_name in image_names:
                image_path = constants.image_prefix + image_name
                image = get_image(image_path)
                results_dict["Data"] = image_name
                approximate_image(image, writer, results_dict)
        else:
            image_name = args.data
            image_path = constants.image_prefix + image_name
            image = get_image(image_path)
            results_dict["Data"] = image_name
            approximate_image(image, writer, results_dict)

if __name__ == "__main__":
    main()
