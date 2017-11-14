#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

from scipy import misc
from numpy.linalg import norm

def alternating_steepest_descent(z0, rank, mask, max_iter, norm_tol):
    begin = time.time()

    # Initialize
    U, s, V = np.linalg.svd(mask*z0, full_matrices=False)
    s[rank:] = 0
    x = (U @ np.diag(s))[:,:rank]
    y = V[:rank,:]

    xy = x@y
    diff = mask*(z0 - xy)

    residuals = []
    norm_z0 = norm(mask*z0)

    tenPowers = [10**k for k in range(10)]

    for num_iter in range(max_iter):
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
        # print(num_iter, residual)

        if num_iter % 1000 == 0:
            residuals.append(residual)

        if residual < norm_tol:
            break

    print("Algoritmo: ASD")
    print("Tiempo:", time.time() - begin)
    print("Iteraciones:", num_iter)
    print("Error relativo:", residual)

    return x@y, residuals


def scaled_alternating_steepest_descent(z0, rank, mask, max_iter, norm_tol):
    begin = time.time()

    # Initialize
    U, s, V = np.linalg.svd(mask*z0, full_matrices=False)
    s = np.sqrt(s)
    s[rank:] = 0
    x = (U @ np.diag(s))[:,:rank]
    y = (np.diag(s) @ V)[:rank,:]

    xy = x@y
    diff = mask*(z0 - xy)

    residuals = []
    norm_z0 = norm(mask*z0)

    for num_iter in range(max_iter):
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

        # print(num_iter, residual)

        if num_iter % 1000 == 0:
            residuals.append(residual)

        if residual < norm_tol:
            break

    print("Algoritmo: sASD")
    print("Tiempo:", time.time() - begin)
    print("Iteraciones:", num_iter)
    print("Error relativo:", residual)

    return x@y, residuals
