import numpy as np
import cv2 as cv
import argparse
import os
import pickle
import time
import threading

class SIFTThread():
    def __init__(self, frame):
        self.frame = frame
    
    def start(self):
        threading.Thread(target=self.run).start()
        return self
    
    def run(self):
        print("I've been called!")
        sift = cv.xfeatures2d.SIFT_create()
        frame_kp, frame_d = sift.detectAndCompute(self.frame, None)

        #***Flann Matcher***
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)  
        flann = cv.FlannBasedMatcher(index_params,search_params)

        good_matches = 0
        old_good_matches = 0
        result = ""
        result_final = None
        matches_result = None 
        result_kp = None
        result_d = None  

        for image in os.listdir('groundtruth/'):
            str_arr = image.split("-")
            good_matches = 0
            groundtruth = cv.imread('groundtruth/'+image) 
                        
            if not os.path.exists("files/keypoints"+str_arr[1]+".txt"):
                print ("Cannot find keypoints for groundtruth image, needs to be added to database")
                exit()
            else:
                keypoints_database = pickle.load(open("files/keypoints"+str_arr[1]+".txt", "rb"))
                groundtruth_kp, groundtruth_d = unpickle_keypoints(keypoints_database[0])

            if (not frame_kp or not groundtruth_kp):
                print("No descriptors found.")
                break

            matches = flann.knnMatch(groundtruth_d,frame_d,k=2)
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
                    
        if(result_final is not None):
            print("Result found!")
            #cv.imshow("Matched Result", result_final)
            cv.imwrite('Test/res.jpg', result_final) 
    
        threading.Timer(20.0, lambda: self.run()).start()
  


def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)
