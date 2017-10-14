#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import misc
from numpy.linalg import norm


"""
Aproxima la matriz X, por otra matriz de rango 'rank', utilizando el teorema
de Eckart–Young–Mirsky. El output es la matriz de aproximación.
"""
def low_rank_approximation(X, rank):
    U, d, Vt = sp.linalg.svd(X)
    d[rank:] = 0
    D = sp.linalg.diagsvd(d, X.shape[0], X.shape[1])
    return np.dot(np.dot(U, D), Vt)

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
  residuals: Vector con muestras de los residuos cada 1000 iteraciones
"""
def alternating_steepest_descent(x0, y0, z0, mask, max_iter, norm_tol):
    x = x0
    y = y0

    xy = x@y
    diff = mask*(z0 - xy)

    residuals = []
    norm_z0 = norm(mask*z0)

    tenPowers = [10**k for k in range(10)]

    for num_iter in range(max_iter):
        print(num_iter)

        grad_x = -diff @ y.T

        delta_xy = mask*(grad_x@y)
        tx = norm(grad_x)**2/norm(delta_xy)**2
        x = x - tx*grad_x

        diff = diff + tx*delta_xy
        grad_y = -x.T @ diff

        delta_xy = mask*(x@grad_y)
        ty = norm(grad_y)**2/norm(delta_xy)**2
        y = y - ty*grad_y

        diff = diff + ty*delta_xy
        residual = norm(diff)/norm_z0

        if num_iter in tenPowers:
            cv2.imwrite("../Results/asd_iter_" + str(num_iter) + ".png", x@y)

        if num_iter % 1000 == 0:
            residuals.append(residual)

        if residual < norm_tol:
            print("norm_tol alcanzada", residual)
            break

    return x@y, residuals


"""
Es una modificación del algoritmo anterior. Los inputs y outputs son los mismos.
"""
def scaled_alternating_steepest_descent(x0, y0, z0, mask, max_iter, norm_tol):
    x = x0
    y = y0

    xy = x@y
    diff = mask*(z0 - xy)

    residuals = []
    norm_z0 = norm(mask*z0)

    for num_iter in range(max_iter):
        print(num_iter)

        grad_x = diff @ y.T

        scale = np.linalg.inv(y @ y.T)
        dx = grad_x @ scale

        delta_xy = mask*(dx @ y)
        tx = np.trace(dx.T @ grad_x)/norm(delta_xy)**2
        
        x = x + tx*dx
        diff = diff - tx*delta_xy

        grad_y = x.T @ diff

        scale = np.linalg.inv(x.T @ x)
        dy = scale @ grad_y

        delta_xy = mask*(x @ dy)
        ty = np.trace(dy @ grad_y.T)/norm(delta_xy)**2

        y = y + ty*dy
        diff = diff - ty*delta_xy

        residual = norm(diff)/norm_z0

        if num_iter % 1000 == 0:
            residuals.append(residual)

        if residual < norm_tol:
            print("norm_tol alcanzada", residual)
            break

    return x@y, residuals
