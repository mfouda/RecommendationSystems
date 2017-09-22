# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.stats import pearsonr

class SimilarityMeasures(object):

    @staticmethod
    def pearson_correlation(x, y):
        if x.size == 0:
            return 0
        value = np.fabs(pearsonr(x, y)[0])
        return value if not math.isnan(value) else 0

    @staticmethod
    def cosine_similarity(x, y):
        value = np.dot(x, y)
        value /= (np.linalg.norm(x)*np.linalg.norm(y))
        value = (value + 1)/2
        return value if not math.isnan(value) else 0
