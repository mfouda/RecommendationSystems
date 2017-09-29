# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sparse
import time
import arff

from user_based import UserBased
from item_based import ItemBased
from similarity_measures import SimilarityMeasures
from evaluation_measures import EvaluationMeasures
from timing import Timing

def read_data():
    data = arff.load(open("../Data/netflix_3m1k/netflix_3m1k.arff", "r"))
    data = np.array(data["data"])
    data = data[:, 1:1001]
    return data.astype(float)

def main():
    np.random.seed(351243)
    np.set_printoptions(linewidth=120, precision=2, threshold=np.nan)

    m = 250
    n = 100
    prob = 0.8 # Probability of unobserved values

    # data = read_data()

    data = np.random.randint(1, 6, (m, n))
    real_data = data.copy()

    mask = np.random.choice([0, 1], (m, n), p=[prob, 1-prob])
    data = mask*data

    number_of_neighbors = 20

    ################
    #  User based  #
    ################
    print("************** User Based ****************\n")

    UserBased.set_data(data)
    UserBased.create_sets(80, 20)

    time1 = time.time()
    predicted_data_correlation = UserBased.similarity_prediction(SimilarityMeasures.pearson_correlation, number_of_neighbors)
    rmse_correlation = EvaluationMeasures.root_mean_square_error(real_data, predicted_data_correlation, UserBased.test_indexes_by_user)
    time2 = time.time()
    print("(TIME) Correlation prediction time", time2 - time1)
    print("RMSE Correlation:", rmse_correlation)

    time1 = time.time()
    predicted_data_cosine = UserBased.similarity_prediction(SimilarityMeasures.cosine_similarity, number_of_neighbors)
    rmse_cosine = EvaluationMeasures.root_mean_square_error(real_data, predicted_data_cosine, UserBased.test_indexes_by_user)
    time2 = time.time()
    print("(TIME) Cosine prediction", time2 - time1)
    print("RMSE Cosine:", rmse_cosine)

    mean_prediction = UserBased.mean_prediction()
    rmse_mean = EvaluationMeasures.root_mean_square_error(real_data, mean_prediction, UserBased.test_indexes_by_user)
    print("RMSE Mean:", rmse_mean)

    user_mean_prediction = UserBased.user_mean_prediction()
    rmse_user_mean = EvaluationMeasures.root_mean_square_error(real_data, user_mean_prediction, UserBased.test_indexes_by_user)
    print("RMSE User Mean:", rmse_user_mean)

    ################
    #  Item based  #
    ################

    print("\n************** Item Based ****************\n")

    ItemBased.set_data(data)
    ItemBased.create_sets(80, 20)

    time1 = time.time()
    predicted_data_correlation = ItemBased.similarity_prediction(SimilarityMeasures.pearson_correlation, number_of_neighbors)
    rmse_correlation = EvaluationMeasures.root_mean_square_error(real_data, predicted_data_correlation, ItemBased.test_indexes_by_user)
    time2 = time.time()
    print("(TIME) Correlation prediction time", time2 - time1)
    print("RMSE Correlation:", rmse_correlation)

    time1 = time.time()
    predicted_data_cosine = ItemBased.similarity_prediction(SimilarityMeasures.cosine_similarity, number_of_neighbors)
    rmse_cosine = EvaluationMeasures.root_mean_square_error(real_data, predicted_data_cosine, ItemBased.test_indexes_by_user)
    time2 = time.time()
    print("(TIME) Cosine prediction", time2 - time1)
    print("RMSE Cosine:", rmse_cosine)

    mean_prediction = ItemBased.mean_prediction()
    rmse_mean = EvaluationMeasures.root_mean_square_error(real_data, mean_prediction, ItemBased.test_indexes_by_user)
    print("RMSE Mean:", rmse_mean)

    item_mean_prediction = ItemBased.item_mean_prediction()
    rmse_user_mean = EvaluationMeasures.root_mean_square_error(real_data, item_mean_prediction, ItemBased.test_indexes_by_user)
    print("RMSE Item Mean:", rmse_user_mean)

if __name__ == "__main__":
    main()
