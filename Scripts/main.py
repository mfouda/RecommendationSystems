# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sparse

from user_based import UserBased
from similarity_measures import SimilarityMeasures
from evaluation_measures import EvaluationMeasures
from timing import Timing

def main():
    np.random.seed(351243)
    np.set_printoptions(linewidth=120, precision=2, threshold=np.nan)

    m = n = 250
    prob = 0.75 # Probabilidad de que haya ceros

    data = np.random.randint(1, 6, (m, n))
    real_data = data.copy()

    mask = np.random.choice([0, 1], (m, n), p=[prob, 1-prob])
    data = mask*data

    sparse_matrix = sparse.csr_matrix(data)
    UserBased.data = sparse_matrix
    UserBased.create_sets(80, 20)

    number_of_neighbors = 15

    predicted_data_correlation = UserBased.similarity_prediction(SimilarityMeasures.pearson_correlation, number_of_neighbors)
    rmse_correlation = EvaluationMeasures.root_mean_square_error(real_data, predicted_data_correlation, UserBased.test_indexes_by_user)
    print("RMSE Correlation")
    print(rmse_correlation)

    predicted_data_cosine = UserBased.similarity_prediction(SimilarityMeasures.cosine_similarity, number_of_neighbors)
    rmse_cosine = EvaluationMeasures.root_mean_square_error(real_data, predicted_data_cosine, UserBased.test_indexes_by_user)
    print("RMSE Cosine")
    print(rmse_cosine)

    mean_prediction = UserBased.mean_prediction()
    rmse_mean = EvaluationMeasures.root_mean_square_error(real_data, mean_prediction, UserBased.test_indexes_by_user)
    print("RMSE Mean")
    print(rmse_mean)

    user_mean_prediction = UserBased.user_mean_prediction()
    rmse_user_mean = EvaluationMeasures.root_mean_square_error(real_data, user_mean_prediction, UserBased.test_indexes_by_user)
    print("RMSE User Mean")
    print(rmse_user_mean)

    Timing.print_times()

if __name__ == "__main__":
    main()
