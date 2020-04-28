import numpy as np
import cv2 as cv
import argparse
import os
import pickle
import time
import threading

from pageLinks import zone

#load in zones for page 68
img = cv.imread('groundtruth/GroundTruth-68-halved.jpg')
data = np.load('zones/68-zones.npy')
page68 = zone(img, data)
links = ["https://www.cse.iitm.ac.in/~vplab/courses/CV_DIP/PDF/NEIGH_CONN.pdf", 
        "http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/connect.html",
        "https://en.wikipedia.org/wiki/Connected-component_labeling", 
        "https://nodepit.com/node/org.knime.knip.base.nodes.seg.ConnectedCompAnalysisNodeFactory%23Connected%20Component%20Analysis",
        "https://aishack.in/tutorials/connected-component-labelling/",
        "https://homepages.inf.ed.ac.uk/rbf/HIPR2/label.htm"]
page68.insert(links) 

#Load in zones for page 69
img = cv.imread('groundtruth/GroundTruth-69-halved.jpg')
data = np.load('zones/69-zones.npy')
page69 = zone(img, data)
links = ["https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html", 
        "https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html",
        "https://www.youtube.com/watch?v=hMIrQdX4BkE", "https://en.wikipedia.org/wiki/Connected-component_labeling"]
page69.insert(links) 

class SIFTThread():
    def __init__(self, frame):
        self.frame = frame
        self.pagenum = 0   
        self.zone = None
        self.newCoords = None

    def start(self):
        threading.Thread(target=self.run()).start()
        return self

    def run(self):
        print("Running SIFT page recognition now...")
        sift = cv.xfeatures2d.SIFT_create()
        frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        frame_kp, frame_d = sift.detectAndCompute(frame, None)

        #***Flann Matcher***
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)  
        flann = cv.FlannBasedMatcher(index_params,search_params)
        
        MIN_MATCH_COUNT = 10
        good_matches = 0
        old_good_matches = 0
        result = ""
        result_final = None
        matches_result = None 
        result_kp = None
        result_d = None  
        img2 = None
        Matrix = None

        for image in os.listdir('groundtruth/'):
            str_arr = image.split("-")
            good_matches = 0
            groundtruth = cv.imread('groundtruth/'+image, 0) 
                        
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
            good = []
            #Ratio test 
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.5*n.distance:
                    good_matches += 1
                    good.append(m)
           
            if len(good)>MIN_MATCH_COUNT:
                src_pts = np.float32([ groundtruth_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ frame_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                Matrix, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
                
                if (Matrix is not None):
                    h, w = groundtruth.shape
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst = cv.perspectiveTransform(pts,Matrix)
                    img2 = cv.polylines(frame,[np.int32(dst)],True,255,3, cv.LINE_AA)

            else:
                print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
                matchesMask = None
                    
            if good_matches > old_good_matches:
                pagenum = str_arr[1]
                old_good_matches = good_matches
                result_final = groundtruth
                matches_result = matches
                result_kp = groundtruth_kp
                result_d = groundtruth_d
                    
        if(result_final is not None):
            self.pagenum = pagenum
            print("Identified page: "+self.pagenum)
            cv.imwrite('SIFT_results/result.jpg', result_final)

            if(pagenum == "68"): 
                self.zone = page68
            else: self.zone = page69
                       
            try:
                for i in range(len(self.zone.coords)):
                    zone = np.vstack(self.zone.coords[i]).squeeze()
                    zone = np.float32([zone])
                    temp = cv.perspectiveTransform(zone, Matrix)
                    self.zone.coords[i] = temp                  
            except:
                print("Something went wrong with transforming zones")
                                   
            #if (img2 is not None):
                #img3 = cv.drawContours(img2, [self.zone.coords.astype(int)], -1, (0,255,0), 3)
                #cv.imwrite('SIFT_results/perspective.jpg', img3)

        threading.Timer(20.0, lambda: self.start()).start()

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)
