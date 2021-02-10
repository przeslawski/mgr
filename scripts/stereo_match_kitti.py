#!/usr/bin/env python

'''
Simple example of stereo image matching using subset of KITTI 2015 stereo dataset.
Usage:
    ESC or q - quit
    h, l - navigate through images <left> <right>
    j, k - adjust block size for semi-global block matching algorithm <up> <down>
    r - reset to default parameters

Default params:
    blockSize = 11
    minDisparity = 0
    numDisparities = 112
    speckleWindowSize = 100
    speckleRange = 2
    uniquenessRatio = 10
    disp12MaxDiff = -1
    window_size = 3
    P1 = 8*3*window_size**2
    P2 = 32*3*window_size**2
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv


def main():

    window_size = 3
    min_disp = 0
    num_disp = 112-min_disp
    block_size = 11

    '''
    Update StereoSGBM algorithm object with current algorithm parameters.
    '''
    def updateStereo():
        stereo = cv.StereoSGBM_create(minDisparity = min_disp,
                numDisparities = num_disp,
                blockSize = block_size,
                P1 = 8*3*window_size**2,
                P2 = 32*3*window_size**2,
                disp12MaxDiff = -1,
                uniquenessRatio = 10,
                speckleWindowSize = 100,
                speckleRange = 2
                )
        return stereo

    def printParams():
        print('blockSize = %d' % block_size)
        print('numDisparities = %d' % num_disp)
        print('window_size = %d' % window_size)

    stereo = updateStereo()


    img_no = 0;
    done = False
    while not done:

        imLname = 'data/kitti/image2/{:06d}_10.png'.format(img_no)
        imRname = 'data/kitti/image3/{:06d}_10.png'.format(img_no)

        print("open file %s" % imLname)

        imgL = cv.pyrDown(cv.imread(imLname))
        imgR = cv.pyrDown(cv.imread(imRname))

        disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

        printParams()
        # print("numDisparities = %d, disp.max = %d, disp.min = %d" % (num_disp, disp.max(), disp.min()))

        cv.imshow('left', imgL)
        cv.imshow('right', imgR)
        cv.imshow('disparity', (disp-min_disp)/num_disp)

        key = cv.waitKey()
        # print(key)
        if key == 27 or key == ord('q'): # ESC
            done = True
        elif key == ord('l'):
            img_no += 1
            img_no %= 10
        elif key == ord('h'):
            if img_no == 0:
                img_no = 10
            img_no -= 1
            img_no %= 10
        elif key == ord('j'):
            block_size = max(block_size-1, 1)
            stereo = updateStereo()
        elif key == ord('k'):
            block_size = min(block_size+1, 33)
            stereo = updateStereo()



if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
