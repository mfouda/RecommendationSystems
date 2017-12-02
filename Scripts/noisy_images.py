#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import ASD
import utils

from user_based import UserBased
from similarity_measures import SimilarityMeasures

def main():
    if len(sys.argv) > 1:
        algorithm = sys.argv[1]
    else:
        algorithm = "ASD"

    image = cv2.imread("../Data/Images/pexels-photo-688018.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Cambiar tamaño
    image = cv2.resize(image, (512, 512))
    m, n = image.shape

    # Aproximación de rango bajo
    rank = 50
    low_rank_image = utils.low_rank_approximation(image, rank)

    # Máscara
    density = 0.3
    mask = np.random.choice([0, 1], (m, n), p=[1-density, density])
    masked_image = mask*low_rank_image

    # Optimizar
    iter_max = 20000
    norm_tol = 1e-4

    if algorithm == "sASD":
        minimize = ASD.scaled_alternating_steepest_descent
        result = minimize(masked_image, rank, mask, iter_max, norm_tol)
        asd_image = result.matrix
    else: # algorithm == "ASD":
        minimize = ASD.alternating_steepest_descent
        result = minimize(masked_image, rank, mask, iter_max, norm_tol)

    masked_image = cv2.cvtColor(masked_image.astype(np.float32), cv2.COLOR_GRAY2RGB)
    masked_image[:,:,1][masked_image[:,:,1] == 0] = 1.0

    path_prefix = "../Results/Noisy_Images/"

    # Calcular error con imagen original
    print("Error relativo", utils.relative_error(asd_image, image))

    # Normalizar imagenes
    normalize = functools.partial(cv2.normalize, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = normalize(image)
    low_rank_image = normalize(low_rank_image)
    mask = normalize(mask)
    masked_image = normalize(masked_image)
    asd_image = normalize(asd_image)

    # Mostrar resultados
    cv2.imshow("masked", masked_image)
    cv2.imshow(algorithm, asd_image)
    cv2.waitKey()

    # Guardar resultados
    cv2.imwrite(path_prefix + "image.png", image)
    cv2.imwrite(path_prefix + "mask.png", mask)
    cv2.imwrite(path_prefix + "low_rank.png", low_rank_image)
    cv2.imwrite(path_prefix + "masked.png", masked_image)
    cv2.imwrite(path_prefix + algorithm + ".png", asd_image)

if __name__ == "__main__":
    main()

