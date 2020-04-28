import numpy as np
import cv2 as cv
import os
import webbrowser
import threading

from VideoGet import VideoGet
from VideoShow import VideoShow
from SIFTThread import SIFTThread
from pageLinks import zone

threshold_area = 8000     #threshold area for contours

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

    frameCount = 0
    oldPoint = None

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        SIFT.frame = frame      
        # apply mask to extract forgeound object
        fgmask = fgbg.apply(frame) 
        # preprocessing: blurring and eroding to remove small finnicky bits
        kernel = np.ones((5,5),np.uint8)
        morph = cv.medianBlur(fgmask, 13)
        morph = cv.erode(morph,kernel,iterations = 3)
  
        # get the contours
        _, contours, _ = cv.findContours(morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            # find the biggest countour by area
            c = max(contours, key = cv.contourArea)
            area = cv.contourArea(c)
            # make sure contour area is bigger than threshold, don't want little contours
            if area > threshold_area:
                # determine the top-most extreme points along the contour 
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                # assign first value 
                if (oldPoint is None):
                    oldPoint = extTop
                # if the finger stays for a few frames, we know someone is pointing    
                if ((oldPoint[0]-5 <= extTop[0] <= oldPoint[0]+5) and (oldPoint[1]-5 <= extTop[1] <= oldPoint[1]+5)):
                    if (extTop[0] != 0 and extTop[1] != 0):
                        frameCount += 1
                        if frameCount == 20:
                            frameCount = 0
                            SIFT.run()
                            #print([SIFT.newCoords.astype(int)])
                            for i in range(len(SIFT.zone.coords)):
                                res = cv.pointPolygonTest(SIFT.zone.coords[i], extTop, measureDist = False)
                                print(res)
                                if (res == 1):
                                    webbrowser.open(SIFT.zone.zones[i][1], new=2)
                    oldPoint = extTop
                else: 
                    frameCount = 0
                    oldPoint = extTop
                # draw a circle at the top-most point
                cv.circle(frame, extTop, 8, (100, 0, 255), -1)
                      
        video_shower.frame = frame

threadBoth(1)
