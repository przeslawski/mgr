#!/usr/bin/env python

'''
Simple example of stereo image matching.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

def main():
    print('loading images...')
    # imgL = cv.pyrDown(cv.imread(cv.samples.findFile('aloeL.jpg')))  # downscale images for faster processing
    # imgR = cv.pyrDown(cv.imread(cv.samples.findFile('aloeR.jpg')))
    # imgGT = cv.pyrDown(cv.imread(cv.samples.findFile('aloeGT.png'), cv.IMREAD_GRAYSCALE))

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 0
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 11,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = -1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 2
    )


    # print("numDisparities == %d" % num_disp)
    # print("disp.min() == %d" % disp.min())
    # print("disp.max() == %d" % disp.max())

    next_key = 83
    prev_key = 81
    esc_key = 27

    print('press ESC to exit')
    img_no = 0;
    done = False
    while not done:

        imLname = 'data/kitti/image2/{:06d}_10.png'.format(img_no)
        imRname = 'data/kitti/image3/{:06d}_10.png'.format(img_no)

        print("open file %s" % imLname)

        imgL = cv.pyrDown(cv.imread(imLname))
        imgR = cv.pyrDown(cv.imread(imRname))


        disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

        cv.imshow('left', imgL)
        cv.imshow('right', imgR)
        cv.imshow('disparity', (disp-min_disp)/num_disp)

        key = cv.waitKey()
        # print(key)
        if key == esc_key:
            done = True
        elif key == next_key:
            img_no += 1
            img_no %= 10
        elif key == prev_key:
            if img_no == 0:
                img_no = 10
            img_no -= 1
            img_no %= 10





if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
