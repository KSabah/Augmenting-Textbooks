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

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """
    cv.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

def threadBoth(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """
    frame_counter = 0
    video_getter = VideoGet(source).start()
    SIFT = SIFTThread(video_getter.frame).start()
    video_shower = VideoShow(video_getter.frame).start()
    cps = CountsPerSec().start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame
        #SIFT.run()
        #SIFT is running, cannot show result due to issues acquiring mutex with imshow due to video. 
        #threading.Timer(5.0, lambda: SIFT.run())
        cps.increment()

threadBoth(0)
