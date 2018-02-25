# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:11:15 2015

@author: gil
"""

import numpy as np
import mc
import time

# N = 1000    
# A = np.random.normal(0,1, (N,1))
# A = A.dot(A.T)
# Mask = np.random.uniform(0,1, (N,N))<0.9
# start = time.time()
# out=mc.MatrixCompletion(A*Mask, Mask, 50, 'nuclear', 1e-3, 1e-6, 1)
# total = time.time()-start
# print("total time =", total)

# -*- coding: utf-8 -*-

import numpy as np
import functools
import cv2
import time
import matplotlib
import matplotlib.pyplot as plt
from matrix_completion import *
from scipy.linalg import svd, diagsvd

matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.dpi"] = 600

image_prefix = "../Data/Images/"
results_prefix = "../Results/ASD_Testing/"
normalize = functools.partial(cv2.normalize, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def low_rank_approximation(X, rank):
    U, d, Vt = svd(X)
    d[rank:] = 0
    D = diagsvd(d, X.shape[0], X.shape[1])
    return np.dot(np.dot(U, D), Vt)

def main():
    image_path = image_prefix + "boat.png"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = normalize(image)
    low_rank_image = normalize(low_rank_approximation(image, 150))

    density = 0.9
    mask = np.random.choice([0, 1], image.shape, p=[1-density, density])

    delta_real = np.max(np.linalg.svd(low_rank_image, False, False))
    print("Real delta:", delta_real)

    delta_masked = np.max(np.linalg.svd(low_rank_image * mask, False, False))
    print("Delta masked", delta_masked)

    result = mc.MatrixCompletion(mask*low_rank_image, mask, 50, 'nuclear', 1e-2, 1e-2, 1)
    result = result.NewMAT

    print("Error: ", np.linalg.norm(mask*(low_rank_image - result)))
    print("Full error: ", np.linalg.norm(low_rank_image - result))
    print("rank(result)", np.linalg.matrix_rank(result))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_aspect("equal")
    plt.imshow(result, interpolation="nearest", cmap=matplotlib.cm.coolwarm)
    plt.colorbar()
    plt.show()

    cv2.imwrite("result.png", np.clip(result, 0, 255))
    cv2.imwrite("masked.png", normalize(low_rank_image*mask))

if __name__ == '__main__':
    main()
