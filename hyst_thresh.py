#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Hysteresis thresholding

Author: Nicolas Ung
MatrNr: 11912380
"""

import cv2
import numpy as np


def hyst_thresh(edges_in: np.array, low: float, high: float) -> np.array:
    """ Apply hysteresis thresholding.

    Apply hysteresis thresholding to return the edges as a binary image. All connected pixels with value > low are
    considered a valid edge if at least one pixel has a value > high.

    :param edges_in: Edge strength of the image in range [0.,1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low: Value below all edges are filtered out
    :type low: float in range [0., 1.]

    :param high: Value which a connected element has to contain to not be filtered out
    :type high: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################
    # Write your own code here

    #print(f"low: {low}, high: {high}")
    
    # create binary image with edges > low as 1
    hyst_out = (edges_in > low).astype(np.uint8)
    
    # use connected components to find connected pixels
    _, label_im = cv2.connectedComponents(hyst_out, connectivity=8)
    
    # find connected components with at least one high value pixel
    labels_high = np.unique(label_im[edges_in > high])
    
    # create binary image
    bitwise_img = np.isin(label_im, labels_high).astype(np.float32)

    ######################################################
    return bitwise_img
