# -*- coding: utf-8 -*-

import numpy as np
import math

from multiprocessing import Pool

from timing import *

class UserBased(object):
    data = None
    similarity_matrix = None
    train_indexes_by_user = None
    test_indexes_by_user = None

    @classmethod
    @timing
    def create_sets(cls, train_percentage=90, test_percentage=10):
        nonzero = cls.data.nonzero()
        n = len(nonzero[0])

        train_indexes = [[] for _ in range(cls.data.shape[0])]
        test_indexes = [[] for _ in range(cls.data.shape[0])]

        selected_indexes = np.random.choice(range(n), size=int(n*train_percentage/100), replace=False)

        for i, j in zip(nonzero[0][selected_indexes], nonzero[1][selected_indexes]):
            train_indexes[i].append(j)

        selected_indexes = np.random.choice(range(n), size=int(n*test_percentage/100), replace=False)
        for i, j in zip(nonzero[0][selected_indexes], nonzero[1][selected_indexes]):
            test_indexes[i].append(j)

        cls.train_indexes_by_user = train_indexes
        cls.test_indexes_by_user = test_indexes

    @classmethod
    @timing
    def create_similarity_matrix(cls, similarity):
        m = cls.data.shape[0]
        similarity_matrix = np.empty((m, m))

        for i in range(m):
            similarity_matrix[i, i] = 1
            train_indexes_set = set(cls.train_indexes_by_user[i])
            for j in range(i + 1, m):
                common_indexes = list(train_indexes_set.intersection(cls.train_indexes_by_user[j]))

                x = cls.data[i, common_indexes].toarray().flatten()
                y = cls.data[j, common_indexes].toarray().flatten()

                value = similarity(x, y)
                similarity_matrix[i, j] = value
                similarity_matrix[j, i] = value

        cls.similarity_matrix = similarity_matrix

    @classmethod
    @timing
    def predict_entry(cls, i, j, similar_users):
        prediction = 0
        den = 0
        for user in similar_users:
            if cls.data[user, j] > 0:
                prediction += cls.similarity_matrix[i, user]*cls.data[user, j]
                den += cls.similarity_matrix[i, user]
        if den != 0:
            prediction /= den
        if math.isnan(prediction):
            print("den", den)
        return prediction

    @classmethod
    @timing
    def similarity_prediction(cls, similarity, number_of_neighbors):
        shape = cls.data.shape
        predicted_data = np.empty(shape)
        cls.create_similarity_matrix(similarity)

        for i, indexes in enumerate(cls.train_indexes_by_user):
            indexes = set(indexes)
            similar_users = np.argpartition(cls.similarity_matrix[i], -number_of_neighbors)[-number_of_neighbors:]

            for j in range(shape[1]):
                predicted_data[i, j] = cls.predict_entry(i, j, similar_users)
        return predicted_data

    @classmethod
    @timing
    def mean_prediction(cls):
        value = cls.data.sum()/cls.data.count_nonzero()
        return value

    @classmethod
    @timing
    def user_mean_prediction(cls):
        shape = cls.data.shape
        predicted_data = np.empty(shape)

        for i in range(shape[0]):
            row = cls.data.getrow(i)
            value = row.sum()/row.count_nonzero()
            predicted_data[i] = np.repeat(value, shape[0])

        return predicted_data
