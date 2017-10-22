# -*- coding: utf-8 -*-

import numpy as np
import time
import csv
import pickle
import itertools

from user_based import UserBased
from item_based import ItemBased
from similarity_measures import SimilarityMeasures
from evaluation_measures import EvaluationMeasures
from timing import Timing

def read_data():
    with open("../../Data/netflix_3m1k/lil_matrix.pkl", "rb") as file:
        lil_matrix = pickle.load(file)
    return lil_matrix

def main():
    # np.random.seed(351243)
    np.set_printoptions(linewidth=120, precision=2, threshold=np.nan)

    use_netflix_data = True

    if use_netflix_data:
        data = read_data()
        real_data = data.copy()
    else:
        m = 100
        n = 150
        prob = 0.2 # Probability of unobserved values
        data = np.random.randint(1, 6, (m, n))
        real_data = data.copy()
        mask = np.random.choice([0, 1], (m, n), p=[prob, 1-prob])
        data = mask*data

    similarity_measures = {
            "pearson_correlation" : SimilarityMeasures.pearson_correlation,
            "cosine_similarity" : SimilarityMeasures.cosine_similarity,
            "mean_squared_difference" : SimilarityMeasures.mean_squared_difference,
            "mean_absolute_difference" : SimilarityMeasures.mean_absolute_difference
            }

    evaluation_measures = {
            "RMSE" : EvaluationMeasures.root_mean_square_error,
            "MAE" : EvaluationMeasures.mean_absolute_error,
            }

    methods = {
            "User_based" : UserBased,
            "Item_based" : ItemBased
            }

    numbers_of_neighbors = [10, 25, 50, 100]
    train_percentages = [30, 50, 70, 90]
    test_percentages = [10, 20, 30, 50]

    with open("../Results/Netflix/Results2.csv", "w") as csv_file:
        fieldnames = ["Method", "Similarity_measure", "Number_of_neighbors", "Train_percentage", "Test_percentage", "Error_measure", "Error_value", "Time"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        results_dict = {}

        # Constant predictions
        for method_name, train_percentage, test_percentage in itertools.product(methods, train_percentages, test_percentages):
            method = methods[method_name]
            method.set_data(data)
            method.create_sets(train_percentage, test_percentage)

            results_dict["Train_percentage"] = train_percentage
            results_dict["Test_percentage"] = test_percentage
            results_dict["Method"] = method_name

            for prediction_method, prediction_method_name in \
                    zip([method.mean_prediction, method.row_mean_prediction], ["mean_prediction", "row_mean_prediction"]):
                time1 = time.time()
                predicted_data = prediction_method()
                time2 = time.time()

                results_dict["Time"] = time2 - time1
                results_dict["Similarity_measure"] = prediction_method_name

                for evaluation_name, evaluation_measure in evaluation_measures.items():
                    evaluation_value = evaluation_measure(real_data, predicted_data, method.test_indexes)

                    results_dict["Error_measure"] = evaluation_name
                    results_dict["Error_value"] = evaluation_value
                    writer.writerow(results_dict)

        # Based on neighborhoods
        evaluation_measures["Zero-one"] = EvaluationMeasures.zero_one_error

        for method_name, train_percentage, test_percentage in \
                itertools.product(methods, train_percentages, test_percentages):
            results_dict["Method"] = method_name

            method = methods[method_name]
            method.set_data(data)
            method.create_sets(train_percentage, test_percentage)

            for similarity_name, number_of_neighbors in itertools.product(similarity_measures, numbers_of_neighbors):

                results_dict["Similarity_measure"] = similarity_name
                results_dict["Number_of_neighbors"] = number_of_neighbors
                results_dict["Train_percentage"] = train_percentage
                results_dict["Test_percentage"] = test_percentage

                similarity = similarity_measures[similarity_name]

                time1 = time.time()
                predicted_data = method.similarity_prediction(similarity, number_of_neighbors)
                time2 = time.time()

                results_dict["Time"] = time2 - time1

                for evaluation_name, evaluation_measure in evaluation_measures.items():
                    evaluation_value = evaluation_measure(real_data, predicted_data, method.test_indexes)

                    results_dict["Error_measure"] = evaluation_name
                    results_dict["Error_value"] = evaluation_value
                    writer.writerow(results_dict)

if __name__ == "__main__":
    main()
