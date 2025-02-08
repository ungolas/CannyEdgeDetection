#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Edge detection with the Sobel filter

Author: Nicolas Ung
MatrNr: 11912380
"""

import cv2
import numpy as np

# I changed the return type to tuple[np.array, np.array] because of a warning:
# Tuple expression not allowed in type annotation

def sobel(img: np.array) -> tuple[np.array, np.array]:
#def sobel(img: np.array) -> (np.array, np.array):
    """ Apply the Sobel filter to the input image and return the gradient and the orientation.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    :return: (gradient, orientation): gradient: edge strength of the image in range [0.,1.],
                                      orientation: angle of gradient in range [-np.pi, np.pi]
    :rtype: (np.array, np.array)
    """
    ######################################################
    # Write your own code here

    # Sobel kernels
    G_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1] ])
    
    G_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1] ])

    # check if np.float32
    if (img.dtype != np.float32):
        raise ValueError("Input image must be np.float32")
    
    # rewrite below 0 values to 0 and
    # above 1 values to 1
    img = np.clip(img, 0.0, 1.0)

    # Apply Sobel kernels
    G_x_erg = cv2.filter2D(img, -1, G_x)
    G_y_erg = cv2.filter2D(img, -1, G_y)

    # Edge strength and angle of gradient
    orientation = np.arctan2(G_y_erg, G_x_erg)

    gradient = np.sqrt(G_x_erg**2 + G_y_erg**2)
    gradient /= np.max(gradient) # normalized to [0, 1]

    ######################################################
    return gradient, orientation
