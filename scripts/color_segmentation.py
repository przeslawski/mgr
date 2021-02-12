#!/usr/bin/env python

'''
Simple example of color segmentation of RGB image.
Usage:
    ESC or q - quit
    h, l - navigate through images <left> <right>

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import utils

def meanshiftfilter(img):

    return img

def idkyet(img):

    return img

def color_segmentation(img1, img2):
    
    original = img1
    cv.imshow('original', original)

    filtered = meanshiftfilter(original)
    cv.imshow('filtered', filtered)


    segmented = idkyet(filtered)
    cv.imshow('segmented', segmented)






    return


if __name__ == '__main__':
    print(__doc__)
    utils.loop(color_segmentation)
    cv.destroyAllWindows()
