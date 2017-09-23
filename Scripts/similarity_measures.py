# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.stats import pearsonr

from timing import *

class SimilarityMeasures(object):

    @staticmethod
    @timing
    def pearson_correlation(x, y):
        if not x:
            return 0
        value = np.fabs(pearsonr(x, y)[0])
        return value if not math.isnan(value) else 0

    @staticmethod
    @timing
    def cosine_similarity(x, y):
        value = np.dot(x, y)
        value /= (np.linalg.norm(x)*np.linalg.norm(y))
        value = (value + 1)/2
        return value if not math.isnan(value) else 0
