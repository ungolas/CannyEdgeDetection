#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Automatic hysteresis thresholding

Author: Nicolas Ung
MatrNr: 11912380
"""

import cv2
import numpy as np

from hyst_thresh import hyst_thresh


def hyst_thresh_auto(edges_in: np.array, low_prop: float, high_prop: float) -> np.array:
    """ Apply automatic hysteresis thresholding.

    Apply automatic hysteresis thresholding by automatically choosing the high and low thresholds of standard
    hysteresis threshold. low_prop is the proportion of edge pixels which are above the low threshold and high_prop is
    the proportion of edgepixels above the high threshold.

    :param edges_in: Edge strength of the image in range [0., 1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low_prop: Proportion of pixels which should lie above the low threshold
    :type low_prop: float in range [0., 1.]

    :param high_prop: Proportion of pixels which should lie above the high threshold
    :type high_prop: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################
    # Write your own code here

    # make close to 0 values to zero
    edges_in[edges_in < 1e-6] = 0
    
    # sorted and >0 edges
    sorted_non_zero_edges = np.sort(edges_in[edges_in > 0])
    
    # number of >0 edges
    total_non_zero_edges = len(sorted_non_zero_edges)
    
    # low/high thresholds based on proportions
    low_threshold = sorted_non_zero_edges[int(total_non_zero_edges * (1 - low_prop))]
    high_threshold = sorted_non_zero_edges[int(total_non_zero_edges * (1 - high_prop))]
    
    # hysteresis thresholding
    hyst_out = hyst_thresh(edges_in, low_threshold, high_threshold)

    ######################################################
    return hyst_out
