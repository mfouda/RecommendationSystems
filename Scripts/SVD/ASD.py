#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import misc
from numpy.linalg import norm

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
