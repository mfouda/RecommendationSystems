# -*- coding: utf-8 -*-

import numpy as np
import cvxopt
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple

cvxopt.solvers.options['show_progress'] = False

def best_approximation_spectral(X, delta):
    u, s, v = np.linalg.svd(X, full_matrices=0)
    if s[0] < delta:
        return X
    s[s > delta] = delta
    return u @ np.diag(s) @ v

# ref: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
def best_approximation_nuclear(X, delta):
    u, s, v = np.linalg.svd(X, full_matrices = 0)
    if np.sum(s) < delta:
        return X

    n = s.size
    P = np.eye(n)
    q = -s.T
    G = np.vstack([np.ones((1, n)), -np.eye(n)])
    h = np.hstack([np.array([[delta]]), np.zeros((1,n))]).T
    sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G), cvxopt.matrix(h))
    x = sol['x']
    sol = u.dot(np.diagflat(x)).dot(v)
    return sol

def matrix_approximation(X, X0, mask, delta, iter_max, error_tol, algorithm):
    if algorithm == "spectral":
        best_approximation = best_approximation_spectral
    elif algorithm == "nuclear":
        best_approximation = best_approximation_nuclear

    error_frobenius = float("inf")
    Xn = X0
    num_iter = 0
    error_prev = 0

    while num_iter < iter_max and error_frobenius > error_tol:
        Xn = (1 - mask)*Xn + mask*X
        Xn = best_approximation(np.array(Xn), delta)

        error_frobenius = np.linalg.norm(mask*(Xn - X))/np.sum(mask)

        if num_iter > 1:
            diff = abs(error_frobenius - error_prev)
            if 1000*diff < error_tol:
                return Xn, error_frobenius

        error_prev = error_frobenius
        num_iter += 1

    return Xn, error_frobenius

def matrix_completion(X, mask, iter_max, delta_tol, error_tol, algorithm="spectral", verbose=False):
    if algorithm == "nuclear":
        delta_max = sum(np.linalg.svd(X, 0, 0))
    elif algorithm == "spectral":
        delta_max = np.max(np.linalg.svd(X, False, False))
    else:
        print("Invalid mode")
        return

    Result = namedtuple("Result", ["X", "delta", "binary_iter", "error"])

    if verbose:
        print("Delta masked:", delta_max)

    delta_max = delta_max * 1.2

    if verbose:
        print("Delta max:", delta_max)

    delta_min = 0
    delta = float("inf")
    delta_prev = 0
    error_frobenius = float("inf")
    binary_iter = 0
    close_deltas = 0

    Xn = X

    while error_frobenius > error_tol or abs(delta - delta_prev) > delta_tol:
        binary_iter += 1
        delta_prev = delta
        delta = (delta_min + delta_max)/2.0

        Xn, error_frobenius = matrix_approximation(mask*X, Xn, mask, delta, iter_max, error_tol, algorithm)

        if verbose:
            print("Binary search iter:", binary_iter, "Error:", error_frobenius, "Delta:", delta)

        if error_frobenius > error_tol:
            delta_min = delta
        else:
            delta_max = delta

        if abs(delta - delta_prev) < delta_tol:
            close_deltas +=1
        if close_deltas >= 5:
            result = Result(X=Xn, delta=delta, binary_iter=binary_iter, error=error_frobenius)
            return result

    result = Result(X=Xn, delta=delta, binary_iter=binary_iter, error=error_frobenius)
    return result
