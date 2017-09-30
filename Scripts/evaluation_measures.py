# -*- coding: utf-8 -*-

import numpy as np
import math
import itertools

class EvaluationMeasures(object):

    @staticmethod
    def mean_absolute_error(real_data, predicted_data, indexes_by_user):
        value = 0
        n = 0
        for i, indexes in enumerate(indexes_by_user):
            for j in indexes:
                if len(np.shape(predicted_data)) == 2:
                    value += math.fabs(real_data[i, j] - predicted_data[i, j])
                else:
                    value += math.fabs(real_data[i, j] - predicted_data)
                n += 1
        return value/n

    @staticmethod
    def root_mean_square_error(real_data, predicted_data, indexes_by_user):
        value = 0
        n = 0
        for i, indexes in enumerate(indexes_by_user):
            for j in indexes:
                if len(np.shape(predicted_data)) == 2:
                    value += (real_data[i, j] - predicted_data[i, j])**2
                else:
                    value += (real_data[i, j] - predicted_data)**2
                n += 1
        return math.sqrt(value/n)

    @staticmethod
    def zero_one_error(real_data, predicted_data, indexes_by_user):
        value = 0
        n = 0
        for i, indexes in enumerate(indexes_by_user):
            for u, v in itertools.combinations(indexes, 2):
                value += int((real_data[i, u] - real_data[i, v]) * (predicted_data[i, u] - predicted_data[i, v]) < 0)
                n += 1
        return value/n

