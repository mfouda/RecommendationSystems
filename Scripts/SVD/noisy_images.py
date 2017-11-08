#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from ASD import *
from numpy.linalg import svd
from scipy.linalg import diagsvd

def low_rank_approximation(X, rank):
    U, d, Vt = svd(X)
    d[rank:] = 0
    D = diagsvd(d, X.shape[0], X.shape[1])
    return np.dot(np.dot(U, D), Vt)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "scaled":
        algorithm = "scaled"
    else:
        algorithm = "asd"

    print("algorithm:", algorithm)

    image = cv2.imread("../../Data/Images/boat.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Cambiar tamaño
    # image = cv2.resize(image, (256, 256))
    m, n = image.shape

    # Aproximación de rango bajo
    rank = 50
    low_rank_image = low_rank_approximation(image, rank)

    # Máscara
    density = 0.30
    mask = np.random.choice([0, 1], (m, n), p=[1-density, density])
    masked_image = mask*low_rank_image

    # Optimizar
    iter_max = 20000
    norm_tol=1e-4

    if algorithm == "asd":
        minimize = alternating_steepest_descent
    elif algorithm == "scaled":
        minimize = scaled_alternating_steepest_descent

    asd_image, residuals = minimize(masked_image, rank, mask, iter_max, norm_tol)

    # Mostrar resultados
    # plt.figure(dpi=150); plt.imshow(image, cmap="gray")
    # plt.figure(dpi=150); plt.imshow(mask, cmap="gray")
    # plt.figure(dpi=150); plt.imshow(low_rank_image, cmap="gray")
    # plt.figure(dpi=150); plt.imshow(masked_image, cmap="gray")

    masked_image = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    masked_image[:,:,1][masked_image[:,:,1] == 0] = 255

    plt.figure(dpi=150); plt.axis("off"); plt.imshow(masked_image)
    plt.figure(dpi=150); plt.axis("off"); plt.imshow(asd_image, cmap="gray")
    plt.show()

    # Gráfica de residuos
    path_prefix = "../../Results/Noisy_Images/"
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=150)
    ax.semilogy(residuals, linewidth=2.0, linestyle="-", marker="o")

    fig.tight_layout()
    plt.savefig(path_prefix + "plot.png", bbox_inches="tight", pad_inches=0)
    # plt.show()
    cv2.imwrite(path_prefix + "image.png", image)
    cv2.imwrite(path_prefix + "mask.png", 255*mask)
    cv2.imwrite(path_prefix + "low_rank.png", low_rank_image)
    cv2.imwrite(path_prefix + "masked.png", masked_image)
    cv2.imwrite(path_prefix + "asd.png", asd_image)

if __name__ == "__main__":
    main()

