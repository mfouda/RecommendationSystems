# -*- coding: utf-8 -*-

import scipy as sp
import numpy as np
import math
import pickle

from scipy.linalg import svd, diagsvd

def low_rank_approximation(X, rank):
    U, d, Vt = svd(X)
    d[rank:] = 0
    D = diagsvd(d, X.shape[0], X.shape[1])
    return np.dot(np.dot(U, D), Vt)

def rmse(M, M_approx, mask):
    mask = mask.astype(np.bool)
    return math.sqrt(((M - M_approx)**2)[mask].mean())

def read_netflix_data(format=None):
    # Rank = 883
    if format == "lil":
        file_path = "../../Data/netflix_3m1k/lil_matrix.pkl"
    else:
        file_path = "../../Data/netflix_3m1k/matrix.pkl"
    with open(file_path, "rb") as file:
        matrix = pickle.load(file)
    return matrix
