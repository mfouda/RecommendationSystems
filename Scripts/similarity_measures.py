# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.stats import pearsonr

class SimilarityMeasures(object):

    @staticmethod
    def pearson_correlation(i, j, x, y):
        if not len(x):
            value = 0
        else:
            value = np.fabs(pearsonr(x, y)[0])
            value = value if not math.isnan(value) else 0
        return i, j, value

    @staticmethod
    def cosine_similarity(i, j, x, y):
        value = np.dot(x, y)
        value /= (np.linalg.norm(x)*np.linalg.norm(y))
        value = (value + 1)/2
        value = value if not math.isnan(value) else 0
        return i, j, value
