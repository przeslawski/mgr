#!/usr/bin/env python

'''
Simple example of color segmentation of RGB image.
Usage:
    ESC or q - quit
    h, l - navigate through images <left> <right>

'''

# Python 2/3 compatibility
from __future__ import print_function

import os
import numpy as np
import cv2 as cv
import utils

def meanshiftfilter(img):

    # spatial window radius
    sp = 4
    # color window radius
    sr = 32

    ter_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    dst = cv.pyrMeanShiftFiltering(img, sp, sr, termcrit=ter_criteria)


    # k means



    
    return dst

def idkyet(img):

    return img

def color_segmentation(img1, img2):
    
    original = img1
    cv.imshow('original', original)

    filtered = meanshiftfilter(original)
    cv.imshow('filtered', filtered)

    # cv.imshow('8, 8', cv.pyrMeanShiftFiltering(img, 8, 8))
    # cv.imshow('8, 16 ', cv.pyrMeanShiftFiltering(img, 8, 16 ))
    # cv.imshow('16, 8 ', cv.pyrMeanShiftFiltering(img, 16, 8 ))
    # cv.imshow('16, 16', cv.pyrMeanShiftFiltering(img, 16, 16))
    # cv.imshow('32, 8 ', cv.pyrMeanShiftFiltering(img, 32, 8 ))
    # cv.imshow('8, 32 ', cv.pyrMeanShiftFiltering(img, 8, 32 ))

    # segmented = idkyet(filtered)
    # cv.imshow('segmented', segmented)






    return


if __name__ == '__main__':
    print(__doc__)

    import sys
    if len(sys.argv) > 1:
        img = cv.imread(sys.argv[1])
        color_segmentation(img, None)
        cv.waitKey()
    else:
        utils.loop(color_segmentation)
        cv.destroyAllWindows()
