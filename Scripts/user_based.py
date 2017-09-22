# -*- coding: utf-8 -*-

import numpy as np
import math

class UserBased(object):
    data = None
    similarity_matrix = None
    indexes_by_user = None

    @classmethod
    def create_indexes_by_user(cls):
        if cls.indexes_by_user is not None:
            return

        indexes = [[] for _ in range(cls.data.shape[0])]
        nonzero = cls.data.nonzero()
        for i, j in zip(*nonzero):
            indexes[i].append(j)
        cls.indexes_by_user = indexes

    @classmethod
    def create_similarity_matrix(cls, similarity):
        m = cls.data.shape[0]
        similarity_matrix = np.empty((m, m))

        for i in range(m):
            similarity_matrix[i, i] = 1
            indexes_set = set(cls.indexes_by_user[i])
            for j in range(i + 1, m):
                common_indexes = list(indexes_set.intersection(cls.indexes_by_user[j]))

                x = cls.data[i, common_indexes].toarray().flatten()
                y = cls.data[j, common_indexes].toarray().flatten()

                value = similarity(x, y)
                similarity_matrix[i, j] = value
                similarity_matrix[j, i] = value

        cls.similarity_matrix = similarity_matrix

    @classmethod
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
    def similarity_prediction(cls, similarity, number_of_neighbors):
        shape = cls.data.shape
        predicted_data = np.empty(shape)

        cls.create_indexes_by_user()
        cls.create_similarity_matrix(similarity)

        for i, indexes in enumerate(cls.indexes_by_user):
            indexes = set(indexes)
            similar_users = np.argpartition(cls.similarity_matrix[i], -number_of_neighbors)[-number_of_neighbors:]

            for j in range(shape[1]):
                predicted_data[i, j] = cls.predict_entry(i, j, similar_users)
        return predicted_data

    @classmethod
    def mean_prediction(cls):
        value = cls.data.sum()/cls.data.count_nonzero()
        return value

    @classmethod
    def user_mean_prediction(cls):
        shape = cls.data.shape
        predicted_data = np.empty(shape)

        for i in range(shape[0]):
            row = cls.data.getrow(i)
            value = row.sum()/row.count_nonzero()
            predicted_data[i] = np.repeat(value, shape[0])

        return predicted_data
