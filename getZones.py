import numpy as np
import cv2 as cv

#Convert image to grayscale
originalImg = cv.imread('GroundTruth-68-zones.jpg')
grayscale = cv.cvtColor(originalImg, cv.COLOR_BGR2GRAY)

#Threshold using Otsu
thresh, binary = cv.threshold(grayscale, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

#Find contours
image, contours, hierarchy = cv.findContours(binary,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)  

#Want the outer contours
externals = []
for i in range(len(contours)):
    if hierarchy[0,i,3] == 1:
        externals.append(contours[i]) 

#threshold for contours
threshold = 3000

zones = list()
#Keep only the contours with 4 sides
for cnt in externals:
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.04*peri, True)
    if len(approx) == 4:
        (x, y, w, h) = cv.boundingRect(approx)
        #Only want large contours
        area = cv.contourArea(cnt)       
        if area > threshold : 
            zones.append(cnt)

#Draw back onto the real image instead of the one with pre-drawn zones
realImg = cv.imread('groundtruth/GroundTruth-68-halved.jpg')
cv.drawContours(realImg, zones, -1, (0,255,0), 3)
cv.namedWindow('Result', cv.WINDOW_NORMAL)
cv.resizeWindow('Result', 1800,900)
cv.imshow('Result', realImg)

cv.waitKey(0)
cv.destroyAllWindows()


     
