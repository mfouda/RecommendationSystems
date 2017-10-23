# -*- coding: utf-8 -*-

import numpy as np
import cv2

from image_mask import ImageMask
from descent import Descent
from utils import *

def main():
    image = cv2.imread("../../Data/Images/house2.tiff", cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    m, n = image.shape[:2]
    rank = 50

    for k in range(image.shape[2]):
        layer = image[:, :, k]
        layer = low_rank_approximation(layer, rank)
        layer *= 255.0/layer.max()
        image[:, :, k] = layer

    ImageMask(image)
    mask = ImageMask.mask

    image_approx = np.empty_like(image)
    for k in range(image.shape[2]):
        V0 = np.random.rand(n, rank)
        layer = ImageMask.image[:,:,k]/255.0
        layer_approx = Descent.optimize(layer, V0, mask)*255.0

        if layer_approx is None:
            return

        image_approx[:,:,k] = layer_approx

    cv2.imshow("original image", image)
    cv2.imshow("masked image", ImageMask.image)
    cv2.imshow("approximated image", image_approx)
    cv2.waitKey()

    cv2.imwrite("../../Results/Inpainting/original.png", image)
    cv2.imwrite("../../Results/Inpainting/masked.png", ImageMask.image)
    cv2.imwrite("../../Results/Inpainting/approximated.png", image_approx)

if __name__ == '__main__':
    main()
