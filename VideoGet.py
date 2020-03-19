from threading import Thread
import cv2 as cv

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src):
        self.cap = cv.VideoCapture(src)
        (self.grabbed, self.frame) = self.cap.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.cap.read()

    def stop(self):
        self.stopped = True