# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy as sp
import scipy.linalg as linalg

from numpy.linalg import norm
from numba import jit

def set_data(data, mask):
    data = data.astype("float64").copy()
    mask = mask.astype("float64").copy()

def create_sets(data, mask, train_percentage=90, test_percentage=10):
    nonzero = data.nonzero()
    n = len(nonzero[0])

    train_indexes = [[] for _ in range(data.shape[0])]
    test_indexes = [[] for _ in range(data.shape[0])]

    selected_indexes = np.random.choice(range(n), size=int(n*train_percentage/100), replace=False)
    for i, j in zip(nonzero[0][selected_indexes], nonzero[1][selected_indexes]):
        train_indexes[i].append(j)

    selected_indexes = set(selected_indexes)
    for k in range(n):
        if k not in selected_indexes:
            i, j  = nonzero[0][k], nonzero[1][k]
            data[i, j] = 0
            mask[i, j] = 0

    selected_indexes = np.random.choice(range(n), size=int(n*test_percentage/100), replace=False)
    for i, j in zip(nonzero[0][selected_indexes], nonzero[1][selected_indexes]):
        test_indexes[i].append(j)

    return train_indexes, test_indexes

"""
Aproxima la matriz X, por otra matriz de rango 'rank', utilizando el teorema
de Eckart–Young–Mirsky. El output es la matriz de aproximación.
"""
def low_rank_approximation(X, rank):
    U, d, Vt = linalg.svd(X)
    d[rank:] = 0
    D = linalg.diagsvd(d, X.shape[0], X.shape[1])
    return np.dot(np.dot(U, D), Vt)

@jit
def regularized_SVD(x0, y0, max_iter, norm_tol, alpha, sigma):
    z0 = data
    mask = mask

    x = x0
    y = y0
    xy = x@y

    residuals = []

    diff = mask*(z0 - xy)
    norm_z0 = norm(mask*z0)

    for num_iter in range(max_iter):
        grad_x = -diff @ y.T + sigma*x
        grad_y = -x.T @ diff + sigma*y

        x -= alpha*grad_x
        y -= alpha*grad_y

        xy = x@y
        diff = mask*(z0 - xy)

        residual = norm(diff)/norm_z0

        if math.isnan(residual):
            print("NAN")
            break
        
        print("Residual", residual)
        if num_iter % 1000 == 0:
            residuals.append(residual)

        if residual < norm_tol:
            print("Tolerancia de la norma alcanzada", residual)
            break

    print("Maximo numero de iteraciones alcanzado")
    return x@y, residuals

"""
Reconstruye una matriz Z a partir de algunas de sus entradas, minimizando ||P(Z0 - XY)||^2
donde P representa un muestreo en el conjunto de entradas conocidas.

Input:
x0, y0: Matrices para inicializar el algoritmo
z0: Matriz que se quiere reconstruir con las entradas conocidas
mask: Matriz cuyas entradas son 1 si el valor de z0 es conocido en esa entrada y 0 en otro caso
max_iter: Número máximo de iteraciones del algorimo
norm_tol: Tolerancia de los residuos relativos
Output:
Z: Matriz de aproximación Z = XY
"""

@jit(nopython=True, parallel=False)
def alternating_steepest_descent(x0, y0, z0, mask, max_iter, norm_tol):
    x = x0
    y = y0

    xy = x@y
    diff = mask*(z0 - xy)

    residuals = []
    norm_z0 = norm(mask*z0)

    for num_iter in range(max_iter):
        print(num_iter)

        grad_x = -diff @ y.T

        delta_xy = mask*(grad_x@y)
        tx = (norm(grad_x)/norm(delta_xy))**2
        x = x - tx*grad_x

        diff = diff + tx*delta_xy
        grad_y = -x.T @ diff

        delta_xy = mask*(x@grad_y)
        ty = (norm(grad_y)/norm(delta_xy))**2
        y = y - ty*grad_y

        diff = diff + ty*delta_xy
        residual = norm(diff)/norm_z0

        if num_iter % 1000 == 0:
            residuals.append(residual)

        if residual < norm_tol:
            print("Tolerancia de la norma alcanzada", residual)
            break

    print("Maximo numero de iteraciones alcanzado")
    return x@y, residuals
