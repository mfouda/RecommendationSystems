# -*- coding: utf-8 -*-

import numpy as np
import math
import time
import scipy.sparse as sparse

from multiprocessing import Pool

class UserBased(object):
    data = None
    similarity_matrix = None
    train_indexes = None
    test_indexes = None

    @classmethod
    def set_data(cls, data):
        if type(data) is sparse.lil.lil_matrix:
            cls.data = data
        else:
            cls.data = sparse.lil_matrix(data)

    @classmethod
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

        cls.train_indexes = train_indexes
        cls.test_indexes = test_indexes

    @classmethod
    def calculate_similarity_entry(cls, i, j, indexes_set, similarity):
        common_indexes = list(indexes_set.intersection(cls.train_indexes[j]))
        if not common_indexes:
            return None
        x = cls.data.getrow(i).toarray()[0, common_indexes]
        y = cls.data.getrow(j).toarray()[0, common_indexes]
        value = similarity(x, y)
        return i, j, value

    @classmethod
    def create_similarity_matrix(cls, similarity):
        m = cls.data.shape[0]
        similarity_matrix = np.zeros((m, m))

        args = []
        for i in range(m):
            similarity_matrix[i, i] = 1
            train_indexes_set = set(cls.train_indexes[i])
            for j in range(i + 1, m):
                args.append((i, j, train_indexes_set, similarity))

        pool = Pool()
        map_result = pool.starmap_async(cls.calculate_similarity_entry, args)
        results = map_result.get()
        pool.close()
        pool.join()

        for result in results:
            if result is not None:
                i, j, value = result
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
        return i, j, prediction

    @classmethod
    def similarity_prediction(cls, similarity, number_of_neighbors):
        shape = cls.data.shape

        time1 = time.time()
        cls.create_similarity_matrix(similarity)
        print("(TIME) Create similarity matrix:", time.time() - time1)

        time1 = time.time()

        args = []
        for i, indexes in enumerate(cls.train_indexes):
            indexes = set(indexes)
            similar_users = np.argpartition(cls.similarity_matrix[i], -number_of_neighbors)[-number_of_neighbors:]
            args.extend([(i, j, similar_users) for j in range(shape[1])])

        pool = Pool()
        map_result = pool.starmap_async(cls.predict_entry, args)
        results = map_result.get()
        pool.close()
        pool.join()

        predicted_data = np.empty(shape)
        for i, j, prediction in results:
            predicted_data[i, j] = prediction

        print("(TIME) Predict data", time.time() - time1)

        return predicted_data

    @classmethod
    def mean_prediction(cls):
        value = 0
        n = 0
        for i, indexes in enumerate(cls.train_indexes):
            row = cls.data.getrow(i).toarray()[0, indexes]
            value += row.sum()
            n += len(row)
        return value/n

    @classmethod
    def row_mean_prediction(cls):
        shape = cls.data.shape
        predicted_data = np.empty(shape)

        for i, indexes in enumerate(cls.train_indexes):
            row = cls.data.getrow(i).toarray()[0, indexes]
            value = row.sum()/len(row)
            value = value if not math.isnan(value) else 0
            predicted_data[i] = np.repeat(value, shape[1])

        return predicted_data
