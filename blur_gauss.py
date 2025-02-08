#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Blur the input image with Gaussian filter kernel

Author: Nicolas Ung
MatrNr: 11912380
"""

import cv2
import numpy as np
from math import ceil

def blur_gauss(img: np.array, sigma: float) -> np.array:
    """ Blur the input image with a Gaussian filter with standard deviation of sigma.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma: The standard deviation of the Gaussian kernel
    :type sigma: float

    :return: Blurred image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0.,1.]
    """
    ######################################################
    # kernel width from exercise description
    kernel_width = np.round( 2 * np.ceil(3 * sigma) + 1).astype(int)

    # create anonymous function for creation of Gaussian kernel
    #(x - kernel_width//2 and y - kernel_width//2, to make middle entry 0,0)
    G = lambda x, y: 1/(2*np.pi*sigma**2) * np.exp(-((x - kernel_width//2)**2 + (y - kernel_width//2)**2) / (2*sigma**2))

    # create kernel
    kernel = np.fromfunction(G, (kernel_width, kernel_width))
    kernel /= np.sum(kernel) # sum must be one
    
    # apply kernel
    img_blur = cv2.filter2D(img, -1, kernel)

    ######################################################
    return img_blur
