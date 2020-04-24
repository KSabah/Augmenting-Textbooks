import numpy as np
import cv2 as cv
import os

class zone:
    """
    Zone object that corresponds to regions allocated on pages.
    Each zone has a list of coordinates which correspond to a link.
    """
    def __init__(self, image, coords):
        self.image = image
        self.coords = coords
        self.zones = dict()

    def insert(self, links):
        #Ensure the same number of links as there are zones
        if (len(self.coords) == len(links)):
            for i in range(len(self.coords)):
                #zone[index] = ([coords numpy array], "link string")
                #zone[index][0] = [coords numpy array], zone[index][1] = "link string"
                self.zones[i] = (self.coords[i], links[i])
        else: print("Need an equal number of zones and links")
        

img = cv.imread('groundtruth/GroundTruth-69-halved.jpg')
data = np.load('zones/pg69-zones.npy')
test = zone(img, data)
links = ["https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html", 
        "https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html",
        "https://www.youtube.com/watch?v=hMIrQdX4BkE", "https://en.wikipedia.org/wiki/Connected-component_labeling"]
test.insert(links)        