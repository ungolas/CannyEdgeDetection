#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Non-Maxima Suppression

Author: Nicolas Ung
MatrNr: 11912380
"""

import cv2
import numpy as np
import random


def non_max(gradients: np.array, orientations: np.array) -> np.array:
    """ Apply Non-Maxima Suppression and return an edge image.

    Filter out all the values of the gradients array which are not local maxima.
    The orientations are used to check for larger pixel values in the direction of orientation.

    :param gradients: Edge strength of the image in range [0.,1.]
    :type gradients: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param orientations: angle of gradient in range [-np.pi, np.pi]
    :type orientations: np.array with shape (height, width) with dtype = np.float32 and values in the range [-pi, pi]

    :return: Non-Maxima suppressed gradients
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    """
    ######################################################
    # Write your own code here

    
    # make all angles positive and convert to degrees
    angle = np.rad2deg(orientations) % 180

    # quantize angles into 4 directions (as stated in the assignment)
    # using bins where the first bin is [0, 22.5) = 1, the second bin is [22.5, 67.5) = 2, ...
    bins = [0, 22.5, 67.5, 112.5, 157.5, 180]
    # make an index-shift bc of the 0-indexing
    directions = np.digitize(angle, bins) - 1
    # have the last intervall be the same as the first bc both are horizontal
    directions = np.where(directions == 4, 0, directions)

    # pad gradients to let the vectorized algorithm work and handle the borders
    grd_pd = np.pad(gradients, pad_width=1, constant_values=0, mode='constant')
    edges = np.zeros_like(grd_pd, dtype=np.float32)

    # the vectrized algorithm works like that:
    # 1. create masks for each direction
    # 2. apply the mask to the gradients and compare the current pixel with the pixels in the direction by using the same index on different slices of the gradients
    # 3. set the current pixel to 0 if it is not the maximum in the direction

    # create masks
    m0 = (directions == 0)
    m45 = (directions == 1)
    m90 = (directions == 2)
    m135 = (directions == 3)

    # apply masks with the shifted slices and compare
    edges[1:-1, 1:-1][m0] = np.where(
        # middle matrix and right matrix
        (grd_pd[1:-1, 1:-1][m0] > grd_pd[1:-1, 2:][m0]) &

        # middle matrix and left matrix
        (grd_pd[1:-1, 1:-1][m0] >= grd_pd[1:-1, :-2][m0]),

        # if the values are greater or equal, keep the middle value, else set to 0
        grd_pd[1:-1, 1:-1][m0], 0)
    
    # analog to the first mask for the other directions
    edges[1:-1, 1:-1][m45] = np.where(
        (grd_pd[1:-1, 1:-1][m45] > grd_pd[2:, 2:][m45]) &
        (grd_pd[1:-1, 1:-1][m45] >= grd_pd[:-2, :-2][m45]),
        grd_pd[1:-1, 1:-1][m45], 0)

    edges[1:-1, 1:-1][m90] = np.where(
        (grd_pd[1:-1, 1:-1][m90] > grd_pd[2:, 1:-1][m90]) &
        (grd_pd[1:-1, 1:-1][m90] >= grd_pd[:-2, 1:-1][m90]),
        grd_pd[1:-1, 1:-1][m90], 0)

    edges[1:-1, 1:-1][m135] = np.where(
        (grd_pd[1:-1, 1:-1][m135] > grd_pd[2:, :-2][m135]) &
        (grd_pd[1:-1, 1:-1][m135] >= grd_pd[:-2, 2:][m135]),
        grd_pd[1:-1, 1:-1][m135], 0)

    # Remove padding
    edges = edges[1:-1, 1:-1]

    ######################################################

    return edges
