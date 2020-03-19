import threading
import numpy as np
import cv2 as cv
import argparse
import os
import pickle
import time

from RepeatedTimer import RepeatingTimer

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        threading.Thread(target=self.show).start()
        return self

    def show(self):
        while not self.stopped:
            cv.imshow("Video", self.frame)
            if cv.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True
    


