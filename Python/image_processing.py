#! /usr/bin/env python

"""
Collection of image processing functions in Python

Author: Martin Noblía
License: MIT
"""
#*************************************************************************
# Imports
import numpy as np
#*************************************************************************

def salt_and_pepper(image, n):
    """
    Inputs:
    ------
    image: image(numpy.array)
    n: number of noise samples(Int)

    Output:
    ------
    image: image + noise (numpy.array)

    """
    a, b = image.shape
    for i in xrange(0,n):
        num_row = np.random.randint(a)
        num_col = np.random.randint(b)
        image[num_row,num_col]=255

    return image

