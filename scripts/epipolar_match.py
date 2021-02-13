#!/usr/bin/env python

'''
Simple example of finding and matching correspondences using SIFT features.
Usage:
    ESC or q - quit
    h, l - navigate through images <left> <right>

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv


def drawLines(img1, img2, lines, pts1, pts2, showlines=True):
    '''
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines
    '''
    r, c, _ = img1.shape
    # img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    # img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        if showlines:
            img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def computeEpilines(img1, img2, pts1, pts2, F):
    # find epilines corresponding to points in right image (second image) and
    # draw its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    imgL, _ = drawLines(img1, img2, lines1, pts1, pts2)

    # same for other image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    imgR, _ = drawLines(img2, img1, lines2, pts2, pts1)

    return imgL, imgR

def detectAndMatchFeatures(img1, img2):
    sift = cv.SIFT_create()

    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    pts1 = []
    pts2 = []

    for idx, desc in enumerate([desc1, desc2]):
        print("img%d detected %d features" % (idx+1, len(desc)))
    print("found %d matches" % len(matches))

    """Filter matches using the Lowe's ratio test.

    The ratio test checks if matches are ambiguous and should be
    removed by checking that the two distances are sufficiently
    different. If they are not, then the match at that keypoint is
    ignored.

    https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
    """
    for idx, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    print("Lowe's test passed for %d points" % len(pts1))

    return pts1, pts2

# def rectify(img, F):
    # rectified = cv.stereoRectify
    # return rectified

def main():

    imLname = 'data/sceauxcastle/images/100_7100.JPG'
    imRname = 'data/sceauxcastle/images/100_7101.JPG'
    img1 = cv.pyrDown(cv.imread(imLname))
    img2 = cv.pyrDown(cv.imread(imRname))

    pts1, pts2 = detectAndMatchFeatures(img1, img2)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

    print("fundamental matrix F found")
    # print(F)

    # select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    print("inlier points img1 %d, img2 %d" % (len(pts1), len(pts2)))

    h1, w1, _ = img1.shape
    
    # rectify
    _, H1, H2 = cv.stereoRectifyUncalibrated(
            np.float32(pts1), np.float32(pts2), F,
            imgSize=(w1, h1), threshold=0)

    img1_undist = cv.warpPerspective(img1, H1, (w1, h1))
    img2_undist = cv.warpPerspective(img2, H2, (w1, h1))

    cv.imshow('img1_undistorted', img1_undist)
    cv.imshow('img2_undistorted', img2_undist)


    # stereo semi-global block matching on rectified images
    window_size = 3
    min_disp = 0
    num_disp = 128-min_disp
    block_size = 11
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
    disp = stereo.compute(img1_undist, img2_undist).astype(np.float32) / 16.0
    cv.imshow('disparity_undistorted', disp)


    imgEpiL, imgEpiR = computeEpilines(img1, img2, pts1, pts2, F)

    cv.imshow('out_epilines_left.png', imgEpiL)
    cv.imshow('out_epilines_right.png', imgEpiR)

    key = cv.waitKey()


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
