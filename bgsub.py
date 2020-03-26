import numpy as np
import cv2 as cv
import argparse
import os
import pickle
import time
import threading

from CountsPerSec import CountsPerSec
from VideoGet import VideoGet
from VideoShow import VideoShow
from SIFTThread import SIFTThread

def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv.pow(cv.subtract(x, cx), 2)
        yp = cv.pow(cv.subtract(y, cy), 2)
        dist = cv.sqrt(cv.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None

def centroid(max_contour):
    moment = cv.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """
    cv.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

def threadBoth(source=1):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """
    video_getter = VideoGet(source).start()
    SIFT = SIFTThread(video_getter.frame).start()
    video_shower = VideoShow(video_getter.frame).start()
    fgbg = cv.createBackgroundSubtractorMOG2() 
    cps = CountsPerSec().start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        SIFT.frame = frame
        # apply mask to extract forgeound object
        fgmask = fgbg.apply(frame) 
        # preprocessing: erode to remove small finnicky bits
        kernel = np.ones((5,5),np.uint8)
        morph = cv.erode(fgmask,kernel,iterations = 3)
        # get the contours
        _, contours, _ = cv.findContours(morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            # ** Method One **
            # find the biggest countour by area
            c = max(contours, key = cv.contourArea)
            # determine the top-most extreme points along the contour 
            # -> https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            # draw a circle at the top-most point
            cv.circle(frame, extTop, 8, (100, 0, 255), -1)
           
            # ** Method Two **
            # -> https://dev.to/amarlearning/finger-detection-and-tracking-using-opencv-and-python-586m
            #c = max(contours, key = cv.contourArea)
            #hull = cv.convexHull(c, returnPoints=False)
            #defects = cv.convexityDefects(c, hull)
            #far_point = farthest_point(defects, c, (centroid(c)))
            #cv.circle(frame, far_point, 5, [100, 0, 255], -1)
            
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame
        cps.increment()

threadBoth(1)
