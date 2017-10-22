# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.stats import pearsonr

class SimilarityMeasures(object):

    @staticmethod
    def pearson_correlation(x, y):
        value = np.fabs(pearsonr(x, y)[0])
        value = value if not math.isnan(value) else 0
        return value

    @staticmethod
    def cosine_similarity(x, y):
        value = np.dot(x, y)
        value /= (np.linalg.norm(x)*np.linalg.norm(y))
        value = value if not math.isnan(value) else 0
        return value

    @staticmethod
    def mean_squared_difference(x, y):
        value = np.sum((x - y)**2)/len(x)
        value = 2/(1 + math.exp(value)) # [0, inf] -> [0, 1] Decreasing
        value = value if not math.isnan(value) else 0
        return value

    @staticmethod
    def mean_absolute_difference(x, y):
        value = np.sum(np.fabs(x - y))/len(x)
        value = 2/(1 + math.exp(value)) # [0, inf] -> [0, 1] Decreasing
        value = value if not math.isnan(value) else 0
        return value
