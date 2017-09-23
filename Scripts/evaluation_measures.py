# -*- coding: utf-8 -*-

import numpy as np
import math

from timing import *

class EvaluationMeasures(object):

    @staticmethod
    @timing
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
    @timing
    def mean_absolute_error(real_data, predicted_data, indexes_by_user):
        value = 0
        n = 0
        for i, indexes in enumerate(indexes_by_user):
            for j in indexes:
                value += math.fabs(real_data[i, j] - predicted_data[i, j])
                n += 1
        return value/n
