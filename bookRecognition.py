import numpy as np
import cv2 as cv

#Convert image to grayscale
originalImg = cv.imread('images/compressed-images/BR-Test1-halved.jpg')
grayscale = cv.cvtColor(originalImg, cv.COLOR_BGR2GRAY)

#Threshold using Otsu
thresh, binary = cv.threshold(grayscale, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#print (thresh)

#Thresh using Canny
#threshold = 100
#binary = cv.Canny(grayscale, threshold, threshold * 2)

#Erosion to keep the letters and other small contours that may be lost
kernel = np.ones((3,3),np.uint8)
morph = cv.erode(binary,kernel,iterations = 3)
cv.namedWindow('Morphology', cv.WINDOW_NORMAL)
cv.resizeWindow('Morphology', 1800,900)
cv.imshow('Morphology', morph)

#Find contours
image, contours, hierarchy = cv.findContours(morph,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)  
if len(contours) != 0:
     # find the biggest countour (c) by the area
    c = max(contours, key = cv.contourArea)
    x,y,w,h = cv.boundingRect(c)
    print (cv.contourArea(c))
    # draw the biggest contour (c) in green
    cv.rectangle(originalImg,(x,y),(x+w,y+h),(0,255,0),2)

#Draw contours and display
#cv.drawContours(originalImg, contours, -1, (0,255,0), 3)
cv.namedWindow('Result', cv.WINDOW_NORMAL)
cv.resizeWindow('Result', 1800,900)
cv.imshow('Result', originalImg)

cv.waitKey(0)
cv.destroyAllWindows()