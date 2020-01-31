import numpy as np
import cv2 as cv
import os
import pickle

#From https://www.isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python
def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])     
        i+=1
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

groundtruth = cv.imread('images/Ground-Truth-pg69.jpg')
test1 = cv.imread('images/Test1-pg69.jpg')

groundtruth = cv.cvtColor(groundtruth,cv.COLOR_BGR2GRAY)
test1 = cv.cvtColor(test1,cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()

if not os.path.exists("files/keypoints69.txt"):
    groundtruth_kp, groundtruth_d = sift.detectAndCompute(groundtruth,None)
    temp_array = []
    temp = pickle_keypoints(groundtruth_kp, groundtruth_d)
    temp_array.append(temp)
    pickle.dump(temp_array, open("files/keypoints69.txt", "wb"))

else:
    keypoints_database = pickle.load(open("files/keypoints69.txt", "rb"))
    groundtruth_kp, groundtruth_d = unpickle_keypoints(keypoints_database[0])

test1_kp, test1_d = sift.detectAndCompute(test1,None)

#***Brute-Force Matcher for SIFT***
#bf = cv.BFMatcher()
#matches = bf.knnMatch(groundtruth_d,test1_d,k=2)
# Apply ratio test
#good = []
#for m,n in matches:
#    if m.distance < 0.75*n.distance:
#        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
#result1 = cv.drawMatchesKnn(groundtruth, groundtruth_kp, test1, test1_kp,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(groundtruth_d,test1_d,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.5*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
result1 = cv.drawMatchesKnn(groundtruth, groundtruth_kp, test1, test1_kp, matches, None, **draw_params)

cv.namedWindow('Result 1', cv.WINDOW_NORMAL)
cv.resizeWindow('Result 1', 1800,900)
cv.imshow('Result 1',result1)
#cv.imwrite('images/surfResult1.jpg',result1)

cv.waitKey(0)
cv.destroyAllWindows()

