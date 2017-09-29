# -*- coding: utf-8 -*-

import numpy as np
from user_based import UserBased

class ItemBased(object):
    train_indexes_by_user = None
    test_indexes_by_user = None

    @classmethod
    def set_data(cls, data):
        UserBased.set_data(data.T)

    @classmethod
    def create_sets(cls, train_percentage=90, test_percentage=10):
        nonzero = UserBased.data.nonzero()
        n = len(nonzero[0])

        train_indexes = [[] for _ in range(UserBased.data.shape[0])]
        test_indexes = [[] for _ in range(UserBased.data.shape[1])]

        selected_indexes = np.random.choice(range(n), size=int(n*train_percentage/100), replace=False)
        for i, j in zip(nonzero[0][selected_indexes], nonzero[1][selected_indexes]):
            train_indexes[i].append(j)

        selected_indexes = np.random.choice(range(n), size=int(n*test_percentage/100), replace=False)
        for i, j in zip(nonzero[0][selected_indexes], nonzero[1][selected_indexes]):
            test_indexes[j].append(i)

        cls.train_indexes_by_user = train_indexes
        cls.test_indexes_by_user = test_indexes
        UserBased.train_indexes_by_user = train_indexes

    @classmethod
    def similarity_prediction(cls, similarity, number_of_neighbors):
        predicted_data = UserBased.similarity_prediction(similarity, number_of_neighbors)
        return predicted_data.T

    @classmethod
    def mean_prediction(cls):
        return UserBased.mean_prediction()

    @classmethod
    def item_mean_prediction(cls):
        return UserBased.user_mean_prediction().T
