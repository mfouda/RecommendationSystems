#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import arff
from scipy.sparse import lil_matrix
from similarity_measures import *

class MatrixCompletion(object):
    def read_data():
        data = arff.load(open("../Data/netflix_3m1k/netflix_3m1k.arff", "r"))
        data = np.array(data["data"])
        data = data[:, 1:1001]
        return data

    def get_indexes_by_user(data):
        indexes = [[] for _ in range(data.shape[0])]
        for i, user in enumerate(data):
            for j, value in enumerate(user):
                if value > 0:
                    indexes[i].append(j)
        return np.array(indexes)

    def get_similarity_matrix(data, similarity):
        m = data.shape[0]
        similarity_matrix = np.empty((m, m))
        indexes_by_user = get_indexes_by_user(data)

        for i in range(m):
            similarity_matrix[i, i] = -1 # Ignore that each user is similar to itself
            indexes_set = set(indexes_by_user[i])
            for j in range(i + 1, m):
                common_indexes = list(indexes_set.intersection(indexes_by_user[j]))
                x = data[i, common_indexes]
                y = data[j, common_indexes]
                value = similarity(x, y)
                similarity_matrix[i, j] = value
                similarity_matrix[j, i] = value

        return similarity_matrix

    def predict_entry(i, j, data, similar_users, similarity_matrix):
        prediction = 0
        for user in similar_users:
            prediction += similarity_matrix[i, user]*data[user, j]
        prediction /= np.sum(similarity_matrix[i, similar_users])
        return prediction if not math.isnan(prediction) else 0

    def predict_data(data, indexes_by_user, similarity, k):
        shape = data.shape
        predicted_data = np.empty(shape)
        similarity_matrix = get_similarity_matrix(data, similarity)

        for i, indexes in zip(range(shape[0]), indexes_by_user):
            indexes = set(indexes)
            similar_users = np.argpartition(similarity_matrix[i], -k)[-k:]

            for j in range(shape[1]):
                predicted_data[i, j] = predict_entry(i, j, data, similar_users, similarity_matrix)

        return predicted_data

def main():
    np.random.seed(123)
    np.set_printoptions(linewidth=120, precision=2)

    # data = read_data()
    # data = data.astype(float)

    m = n = 50
    prob = 0.99 # Probabilidad de que haya ceros

    data = np.random.randint(1, 6, (m, n))
    mask = np.random.choice([0, 1], (m, n), p=[prob, 1-prob])
    original_data = data.copy()
    data = mask*data

    indexes_by_user = get_indexes_by_user(data)
    sparse_matrix = lil_matrix(data)

    similarity_matrix = get_similarity_matrix(data, indexes_by_user, pearson_correlation)

    # predicted_data = predict_data(sparse_matrix, indexes_by_user, pearson_correlation, 10)

    print(np.linalg.norm(data - predicted_data))

    #TODO: Evaluation measures

if __name__ == '__main__':
    main()
