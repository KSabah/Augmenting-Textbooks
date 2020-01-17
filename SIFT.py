#Good tutorial! -> https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt

groundtruth = cv.imread('images/Ground-Truth-pg69.jpg')
test1 = cv.imread('images/Test1-pg69.jpg')
test2 = cv.imread('images/Test2-pg69.jpg')

groundtruth = cv.cvtColor(groundtruth,cv.COLOR_BGR2GRAY)
test1 = cv.cvtColor(test1,cv.COLOR_BGR2GRAY)
test2 = cv.cvtColor(test2,cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()

groundtruth_kp, groundtruth_d = sift.detectAndCompute(groundtruth,None)
test1_kp, test1_d = sift.detectAndCompute(test1,None)
test2_kp, test2_d = sift.detectAndCompute(test2,None)

len(groundtruth_kp), len(test1_kp), len(test2_kp)

bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

matches = bf.match(groundtruth_d, test1_d)
matches = sorted(matches, key = lambda x:x.distance)
result1 = cv.drawMatches(groundtruth, groundtruth_kp, test1, test1_kp, matches[:50], test1, flags=2)
cv.namedWindow('Result 1', cv.WINDOW_NORMAL)
cv.resizeWindow('Result 1', 1800,900)
cv.imshow('Result 1',result1)
#cv.imwrite('images/result1.jpg',result1)

matches = bf.match(groundtruth_d, test2_d)
matches = sorted(matches, key = lambda x:x.distance)
result2 = cv.drawMatches(groundtruth, groundtruth_kp, test2, test2_kp, matches[:50], test1, flags=2)
cv.namedWindow('Result 2', cv.WINDOW_NORMAL)
cv.resizeWindow('Result 2', 1800,900)
cv.imshow('Result 2',result2)
#cv.imwrite('images/result2.jpg',result2)

cv.waitKey(0)
cv.destroyAllWindows()

