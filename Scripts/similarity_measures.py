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
