import numpy as np
import cv2 as cv
import argparse
import os
import pickle
import time

from CountsPerSec import CountsPerSec

first_frame = bool(False)
book_found = 0
frame_counter = 0

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """
    cv.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

def find_book(ndarray):
    #Convert to Greyscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #Threshold using Otsu
    thresh, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    #Erosion to keep the letters and other small contours that may be lost
    kernel = np.ones((3,3),np.uint8)
    morph = cv.erode(binary,kernel,iterations = 3)
    x = 0
    y = 0
    w = 0
    h = 0
    #Find contours
    image, contours, hierarchy = cv.findContours(morph,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)  
    if len(contours) != 0:
        #Find biggest countour (c) by the area
        c = max(contours, key = cv.contourArea)
        x,y,w,h = cv.boundingRect(c)
        #print (cv.contourArea(c))
        #Draw the biggest contour (c) in green
        #cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi = frame[y:y+h, x:x+w]
        print("ROI found!")
        cv.imshow('Region of Interest', roi)
        return roi
    return None

#https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/
#Fuck you huge helpfuls link
#Fix this function please and thanks bby
def sift_operation(ndarray):
    sift = cv.xfeatures2d.SIFT_create()
    frame_kp, frame_d = sift.detectAndCompute(frame, None)

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
            print ("Cannot find keypoints for groundtruth image, needs to be added to database")
        else:
            keypoints_database = pickle.load(open("files/keypoints"+str_arr[1]+".txt", "rb"))
            groundtruth_kp, groundtruth_d = unpickle_keypoints(keypoints_database[0])
            
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
            
        #print(old_good_matches)
        #print(good_matches)
        cv.imshow('matched image', result_final)


#If using a video file, videos captured by the webcam and MP4 files have been tested and work
cap = cv.VideoCapture(1) #Capture Video from the connected USB, \dev\video\0 is the laptop's webcam
if not cap.isOpened():
    print("Cannot open camera")
    exit()
cps = CountsPerSec().start()
while True:
    
    #cap.set(cv.CAP_PROP_FPS, 25)
    #Capture each frame
    ret, frame = cap.read()
    #Correct read of frame -> ret == True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame_counter += 1
    if frame_counter == 20:
        sift_operation(frame)
        frame_counter = 0

    #Display the result
    resFrame = putIterationsPerSec(frame, cps.countsPerSec())
    cv.imshow('frame', resFrame)
    cps.increment()
    #Quit when q is pressed
    if cv.waitKey(1) == ord('q'):
        break
#When everything is done, release the capture
cap.release() 
cv.destroyAllWindows()