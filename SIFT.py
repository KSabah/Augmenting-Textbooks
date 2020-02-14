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

test1 = cv.imread('images/Test1-pg94.jpg')
#test2 = cv.imread('images/Test2-pg69.jpg')

test1 = cv.cvtColor(test1,cv.COLOR_BGR2GRAY)
#test2 = cv.cvtColor(test2,cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()

test1_kp, test1_d = sift.detectAndCompute(test1,None)
#test2_kp, test2_d = sift.detectAndCompute(test2,None)

#***Flann Matcher***
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)  
flann = cv.FlannBasedMatcher(index_params,search_params)

good_matches = 0
old_good_matches = 0
result = ""
matches_result = None 
result_kp = None
result_d = None  

for image in os.listdir('groundtruth/'):
    str_arr = image.split("-")
    good_matches = 0
    groundtruth = cv.imread('groundtruth/'+image) 
        
    if not os.path.exists("files/keypoints"+str_arr[1]+".txt"):
        groundtruth_kp, groundtruth_d = sift.detectAndCompute(groundtruth,None)
        temp_array = []
        temp = pickle_keypoints(groundtruth_kp, groundtruth_d)
        temp_array.append(temp)
        pickle.dump(temp_array, open("files/keypoints"+str_arr[1]+".txt", "wb"))
    else:
     keypoints_database = pickle.load(open("files/keypoints"+str_arr[1]+".txt", "rb"))
     groundtruth_kp, groundtruth_d = unpickle_keypoints(keypoints_database[0])
    
    matches = flann.knnMatch(groundtruth_d,test1_d,k=2)
    #Mask for good matches
    matchesMask = [[0,0] for i in range(len(matches))]
    #Ratio test 
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            matchesMask[i]=[1,0]
            good_matches += 1
    
    if good_matches > old_good_matches:
        old_good_matches = good_matches
        result_final = groundtruth
        matches_result = matches
        result_kp = groundtruth_kp
        result_d = groundtruth_d
    
    print(old_good_matches)
    print(good_matches)
    
matchesMask = [[0,0] for i in range(len(matches_result))]
for i,(m,n) in enumerate(matches_result):
    if m.distance < 0.5*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
res = cv.drawMatchesKnn(result_final, result_kp, test1, test1_kp, matches_result, None, **draw_params)

cv.namedWindow('Result 1', cv.WINDOW_NORMAL)
cv.resizeWindow('Result 1', 1800,900)
cv.imshow('Result 1',res)
#Change filename below to save any new images
#cv.imwrite('images/siftResult1.jpg',result1)

cv.waitKey(0)
cv.destroyAllWindows()

