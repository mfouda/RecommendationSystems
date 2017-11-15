# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import functools

import ASD
import utils

from image_mask import ImageMask
from descent import Descent

def main():
    if len(sys.argv) > 1:
        algorithm = sys.argv[1]
    else:
        algorithm = "ASD"

    image = cv2.imread("../Data/Images/balloons.jpeg", cv2.IMREAD_COLOR)
    image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = cv2.resize(image, (256, 256))
    # image = cv2.resize(image, (512, 512))
    m, n = image.shape[:2]
    rank = 50
    iter_max = 20000
    norm_tol = 1e-4

    for k in range(image.shape[2]):
        layer = image[:, :, k]
        layer = utils.low_rank_approximation(layer, rank)
        image[:, :, k] = layer

    ImageMask(image)
    mask = ImageMask.mask

    image_approx = np.empty_like(image)

    for k in range(image_approx.shape[2]):
        layer = mask*ImageMask.image[:,:,k]

        cv2.imshow("Layer", layer)
        cv2.waitKey()

        if algorithm == "sASD":
            layer_approx, residuals = ASD.scaled_alternating_steepest_descent(layer, rank, mask, iter_max, norm_tol)
        else: # ASD
            layer_approx, residuals = ASD.alternating_steepest_descent(layer, rank, mask, iter_max, norm_tol)

        if layer_approx is None:
            return

        cv2.imshow("Layer approx", layer_approx)
        cv2.waitKey()

        image_approx[:,:,k] = layer_approx

    
    normalize = functools.partial(cv2.normalize, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = normalize(image)
    image_approx = normalize(image_approx)
    # ImageMask.image = normalize(ImageMask.image)

    cv2.imshow("image", image)
    cv2.imshow("masked", ImageMask.image)
    cv2.imshow("approx", image_approx)
    cv2.waitKey()

    path_prefix = "../Results/Inpainting/"
    path_posfix = "_" + algorithm + ".png"
    cv2.imwrite(path_prefix + "original" + path_posfix, image)
    cv2.imwrite(path_prefix + "masked" + path_posfix, ImageMask.image)
    cv2.imwrite(path_prefix + "approximated" + path_posfix, image_approx)

if __name__ == '__main__':
    main()
