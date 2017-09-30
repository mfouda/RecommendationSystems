# -*- coding: utf-8 -*-

import numpy as np
import time
import arff
import csv
import pickle

from user_based import UserBased
from item_based import ItemBased
from similarity_measures import SimilarityMeasures
from evaluation_measures import EvaluationMeasures
from timing import Timing

def read_data():
    with open("../Data/netflix_3m1k/lil_matrix.pkl", "rb") as file:
        lil_matrix = pickle.load(file)
    return lil_matrix

def main():
    # np.random.seed(351243)
    np.set_printoptions(linewidth=120, precision=2, threshold=np.nan)

    # m = 100
    # n = 250
    # prob = 0.8 # Probability of unobserved values

    data = read_data()
    # data = np.random.randint(1, 6, (m, n))
    real_data = data.copy()

    # mask = np.random.choice([0, 1], (m, n), p=[prob, 1-prob])
    # data = mask*data

    results_dict = {}
    with open("../Results/Netflix/Results.csv", "w") as csv_file:
        fieldnames = ["Method", "Number_of_neighbors", "Train_percentage", "Test_percentage", "Error_measure", "Error_value", "Time"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for number_of_neighbors in [2, 5, 10, 25, 50]:
            results_dict["Number_of_neighbors"] = number_of_neighbors
            for train_percentage in [30, 50, 70, 90]:
                results_dict["Train_percentage"] = train_percentage
                for test_percentage in [10, 20, 30, 50]:
                    results_dict["Test_percentage"] = test_percentage

                    print("Number_of_neighbors", number_of_neighbors)
                    print("Train {}%, Test {}%".format(train_percentage, test_percentage))

                    ################
                    #  User based  #
                    ################

                    print("\n************** User Based ****************\n")

                    UserBased.set_data(data)
                    UserBased.create_sets(80, 20)

                    time1 = time.time()
                    predicted_data_correlation = UserBased.similarity_prediction(SimilarityMeasures.pearson_correlation, number_of_neighbors)
                    time2 = time.time()
                    mae_correlation = EvaluationMeasures.mean_absolute_error(real_data, predicted_data_correlation, UserBased.test_indexes_by_user)
                    rmse_correlation = EvaluationMeasures.root_mean_square_error(real_data, predicted_data_correlation, UserBased.test_indexes_by_user)
                    zero_one_correlation = EvaluationMeasures.zero_one_error(real_data, predicted_data_correlation, UserBased.test_indexes_by_user)

                    results_dict["Method"] = "User based, Pearson Correlation"
                    results_dict["Time"] = time2 - time1

                    results_dict["Error_measure"] = "MAE"
                    results_dict["Error_value"] = mae_correlation
                    writer.writerow(results_dict)

                    results_dict["Error_measure"] = "RMSE"
                    results_dict["Error_value"] = rmse_correlation
                    writer.writerow(results_dict)

                    results_dict["Error_measure"] = "Zero-one"
                    results_dict["Error_value"] = zero_one_correlation
                    writer.writerow(results_dict)

                    print("(TIME) Correlation prediction time", time2 - time1)

                    time1 = time.time()
                    predicted_data_cosine = UserBased.similarity_prediction(SimilarityMeasures.cosine_similarity, number_of_neighbors)
                    time2 = time.time()

                    print("(TIME) Cosine prediction", time2 - time1)

                    mae_cosine = EvaluationMeasures.mean_absolute_error(real_data, predicted_data_cosine, UserBased.test_indexes_by_user)
                    rmse_cosine = EvaluationMeasures.root_mean_square_error(real_data, predicted_data_cosine, UserBased.test_indexes_by_user)
                    zero_one_cosine = EvaluationMeasures.zero_one_error(real_data, predicted_data_cosine, UserBased.test_indexes_by_user)

                    results_dict["Method"] = "User based, Cosine"
                    results_dict["Time"] = time2 - time1

                    results_dict["Error_measure"] = "MAE"
                    results_dict["Error_value"] = mae_cosine
                    writer.writerow(results_dict)

                    results_dict["Error_measure"] = "RMSE"
                    results_dict["Error_value"] = rmse_cosine
                    writer.writerow(results_dict)

                    results_dict["Error_measure"] = "Zero-one"
                    results_dict["Error_value"] = zero_one_cosine
                    writer.writerow(results_dict)

                    mean_prediction = UserBased.mean_prediction()
                    mae_mean = EvaluationMeasures.mean_absolute_error(real_data, mean_prediction, UserBased.test_indexes_by_user)
                    rmse_mean = EvaluationMeasures.root_mean_square_error(real_data, mean_prediction, UserBased.test_indexes_by_user)

                    results_dict["Method"] = "Global mean"
                    results_dict["Time"] = time2 - time1

                    results_dict["Error_measure"] = "MAE"
                    results_dict["Error_value"] = mae_mean
                    writer.writerow(results_dict)

                    results_dict["Error_measure"] = "RMSE"
                    results_dict["Error_value"] = rmse_mean
                    writer.writerow(results_dict)

                    user_mean_prediction = UserBased.user_mean_prediction()
                    mae_user_mean = EvaluationMeasures.mean_absolute_error(real_data, user_mean_prediction, UserBased.test_indexes_by_user)
                    rmse_user_mean = EvaluationMeasures.root_mean_square_error(real_data, user_mean_prediction, UserBased.test_indexes_by_user)

                    results_dict["Method"] = "User mean"
                    results_dict["Time"] = time2 - time1

                    results_dict["Error_measure"] = "MAE"
                    results_dict["Error_value"] = mae_user_mean
                    writer.writerow(results_dict)

                    results_dict["Error_measure"] = "RMSE"
                    results_dict["Error_value"] = rmse_user_mean
                    writer.writerow(results_dict)

                    print("\nMAE Correlation:", mae_correlation)
                    print("MAE Cosine:", rmse_cosine)
                    print("MAE Mean:", rmse_mean)
                    print("MAE User Mean:", mae_user_mean)

                    print("\nRMSE Correlation:", rmse_correlation)
                    print("RMSE Cosine:", rmse_cosine)
                    print("RMSE Mean:", rmse_mean)
                    print("RMSE User Mean:", rmse_user_mean)

                    print("\nZero One Correlation:", zero_one_correlation)
                    print("Zero One Cosine:", zero_one_cosine)

                    ################
                    #  Item based  #
                    ################

                    print("\n************** Item Based ****************\n")

                    ItemBased.set_data(data)
                    ItemBased.create_sets(80, 20)

                    time1 = time.time()
                    predicted_data_correlation = ItemBased.similarity_prediction(SimilarityMeasures.pearson_correlation, number_of_neighbors)
                    time2 = time.time()

                    mae_correlation = EvaluationMeasures.mean_absolute_error(real_data, predicted_data_correlation, ItemBased.test_indexes_by_user)
                    rmse_correlation = EvaluationMeasures.root_mean_square_error(real_data, predicted_data_correlation, ItemBased.test_indexes_by_user)
                    zero_one_correlation = EvaluationMeasures.zero_one_error(real_data, predicted_data_correlation, ItemBased.test_indexes_by_user)

                    results_dict["Method"] = "Item based, Pearson Correlation"
                    results_dict["Time"] = time2 - time1

                    results_dict["Error_measure"] = "MAE"
                    results_dict["Error_value"] = mae_correlation
                    writer.writerow(results_dict)

                    results_dict["Error_measure"] = "RMSE"
                    results_dict["Error_value"] = rmse_correlation
                    writer.writerow(results_dict)

                    results_dict["Error_measure"] = "Zero-one"
                    results_dict["Error_value"] = zero_one_correlation
                    writer.writerow(results_dict)

                    print("(TIME) Correlation prediction time", time2 - time1)

                    time1 = time.time()
                    predicted_data_cosine = ItemBased.similarity_prediction(SimilarityMeasures.cosine_similarity, number_of_neighbors)
                    time2 = time.time()

                    mae_cosine = EvaluationMeasures.mean_absolute_error(real_data, predicted_data_cosine, ItemBased.test_indexes_by_user)
                    rmse_cosine = EvaluationMeasures.root_mean_square_error(real_data, predicted_data_cosine, ItemBased.test_indexes_by_user)
                    zero_one_cosine = EvaluationMeasures.zero_one_error(real_data, predicted_data_cosine, ItemBased.test_indexes_by_user)

                    print("(TIME) Cosine prediction", time2 - time1)

                    results_dict["Method"] = "Item based, Cosine"
                    results_dict["Time"] = time2 - time1

                    results_dict["Error_measure"] = "MAE"
                    results_dict["Error_value"] = mae_cosine
                    writer.writerow(results_dict)

                    results_dict["Error_measure"] = "RMSE"
                    results_dict["Error_value"] = rmse_cosine
                    writer.writerow(results_dict)

                    results_dict["Error_measure"] = "Zero-one"
                    results_dict["Error_value"] = zero_one_cosine
                    writer.writerow(results_dict)

                    item_mean_prediction = ItemBased.item_mean_prediction()
                    mae_item_mean = EvaluationMeasures.mean_absolute_error(real_data, item_mean_prediction, ItemBased.test_indexes_by_user)
                    rmse_item_mean = EvaluationMeasures.root_mean_square_error(real_data, item_mean_prediction, ItemBased.test_indexes_by_user)

                    results_dict["Method"] = "Item mean"
                    results_dict["Time"] = time2 - time1

                    results_dict["Error_measure"] = "MAE"
                    results_dict["Error_value"] = mae_item_mean
                    writer.writerow(results_dict)

                    results_dict["Error_measure"] = "RMSE"
                    results_dict["Error_value"] = rmse_item_mean
                    writer.writerow(results_dict)

                    print("\nMAE Correlation:", mae_correlation)
                    print("MAE Cosine:", rmse_cosine)
                    print("MAE Mean:", rmse_mean)
                    print("MAE User Mean:", mae_item_mean)

                    print("\nRMSE Correlation:", rmse_correlation)
                    print("RMSE Cosine:", rmse_cosine)
                    print("RMSE Mean:", rmse_mean)
                    print("RMSE Item Mean:", rmse_item_mean)

                    print("\nZero One Correlation:", zero_one_correlation)
                    print("Zero One Cosine:", zero_one_cosine)
                    print("\n**************************************************************************************************************\n")

if __name__ == "__main__":
    main()
