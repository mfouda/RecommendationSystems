# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sparse

from user_based import UserBased
from similarity_measures import SimilarityMeasures
from evaluation_measures import EvaluationMeasures

def main():
    np.random.seed(351243)
    np.set_printoptions(linewidth=120, precision=2, threshold=np.nan)

    m = n = 100
    prob = 0.5 # Probabilidad de que haya ceros

    data = np.random.randint(1, 6, (m, n))
    real_data = data.copy()

    mask = np.random.choice([0, 1], (m, n), p=[prob, 1-prob])
    data = mask*data

    sparse_matrix = sparse.csr_matrix(data)
    UserBased.data = sparse_matrix

    number_of_neighbors = 10

    predicted_data_correlation = UserBased.similarity_prediction(SimilarityMeasures.pearson_correlation, number_of_neighbors)
    predicted_data_cosine = UserBased.similarity_prediction(SimilarityMeasures.cosine_similarity, number_of_neighbors)
    mean_prediction = UserBased.mean_prediction()
    user_mean_prediction = UserBased.user_mean_prediction()

    rmse_correlation = EvaluationMeasures.root_mean_square_error(real_data, predicted_data_correlation, UserBased.indexes_by_user)
    rmse_cosine = EvaluationMeasures.root_mean_square_error(real_data, predicted_data_cosine, UserBased.indexes_by_user)
    rmse_mean = EvaluationMeasures.root_mean_square_error(real_data, mean_prediction, UserBased.indexes_by_user)
    rmse_user_mean = EvaluationMeasures.root_mean_square_error(real_data, user_mean_prediction, UserBased.indexes_by_user)

    print("RMSE Correlation")
    print(rmse_correlation)
    print("RMSE Cosine")
    print(rmse_cosine)
    print("RMSE Mean")
    print(rmse_mean)
    print("RMSE User Mean")
    print(rmse_user_mean)

if __name__ == "__main__":
    main()
