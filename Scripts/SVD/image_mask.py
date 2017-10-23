# -*- coding: utf-8 -*-

import numpy as np
import cv2

class ImageMask(object):
    image = None
    mask = None
    drawing = False # True if mouse is pressed
    mask_color = (0, 255, 0)
    radius = 5
    window_name = "ImageMask"

    @classmethod
    def __init__(cls, image):
        cls.image = image.copy()
        cv2.namedWindow(cls.window_name)
        cv2.setMouseCallback(cls.window_name, cls.draw_circle)

        cv2.imshow(cls.window_name, cls.image)
        cv2.waitKey()
        cv2.destroyWindow(cls.window_name)

        cls.mask = ~((cls.mask_color == cls.image).all(axis=2))

    @classmethod # Mouse callback function
    def draw_circle(cls, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cls.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if cls.drawing == True:
                cv2.circle(cls.image, (x, y), cls.radius, cls.mask_color, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            cls.drawing = False
            cv2.circle(cls.image, (x, y), cls.radius, cls.mask_color, -1)

        cv2.imshow(cls.window_name, cls.image)
