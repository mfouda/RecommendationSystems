# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import functools

from scipy.optimize import minimize
from utils import *

def error(V, M, U, mask):
    m, n = M.shape
    m, r = U.shape
    V = V.reshape(n, r)
    return 0.5 * np.linalg.norm((M - U@V.T)[mask])**2

def der_U(M, U, V, mask):
    return (mask*(U @ V.T - M))@V

def der_V(V, M, U, mask):
    m, n = M.shape
    m, r = U.shape
    V = V.reshape(n, r)
    return ((mask.T *(V@U.T - M.T))@U).flatten()

def optimize(M, V, mask):
    m, n = M.shape
    n, r = V.shape
    U_opt = np.empty((m, r))

    for i in range(M.shape[0]):
        W = mask[i]
        Z = V.T @ (W.reshape(-1, 1) * V)
        try:
            Z = np.linalg.inv(Z)
        except:
            Z = np.linalg.pinv(Z)
        U_opt[i] = Z @ V.T @ (W * M[i])

    der = der_U(M, U_opt, V, mask)
    der_norm = np.linalg.norm(der)
    print("norma de la derivada:", der_norm)

    if der_norm > 0.5:
        print("der(U*) != 0")
        return None

    gradient = functools.partial(der_V, M=M, U=U_opt, mask=mask)
    error_func = functools.partial(error, M=M, U=U_opt, mask=mask)
    V = V.flatten()
    result = minimize(error_func, x0=V, jac=gradient)
    print("convergencia:", result.message)

    return U_opt @ result.x.reshape(n, r).T if result.success else None
