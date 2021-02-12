import cv2 as cv

#TODO: add dataset as argument
def loop(func):
    '''
    main loop opens input images, performs processing function given in argument and accepts user input
    @param func - processing function which takes img1 and img2 as arguments
    '''

    done = False
    img_no = 0
    while not done:
        imLname = 'data/kitti/image2/{:06d}_10.png'.format(img_no)
        imRname = 'data/kitti/image3/{:06d}_10.png'.format(img_no)

        print("open file %s" % imLname)

        img1 = cv.pyrDown(cv.imread(imLname))
        img2 = cv.pyrDown(cv.imread(imRname))

        # process images
        func(img1, img2)

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

