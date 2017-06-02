#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:01:40 2017

@author: abe
"""

import cv2 as cv
import numpy as np
from scipy.misc import imresize

from mlutil.image import pyramid
import posenet

width, height = 640,480

colors = np.array([(255,0,0),
                   (0,255,0),
                   (0,0,255),
                   (255,255,0),
                   (0,255,255),
                   (255,0,255),
                   (255,127,127),
                   (127,255,127),
                   (127,127,255),
                   (255,127,0),
                   (0,255,127),
                   (127,0,255),
                   (127,255,0),
                   (0,127,255),
                   (255,0,127),
                   (127,0,0),
                   (0,127,0),
                   (0,0,127),
                   (127,127,0),
                   (0,127,127),
                   (127,0,127),
                   (127,127,127),
                   (255,255,127),
                   (127,255,255),
                   (255,127,255),
                   (0,0,0),
                   (0,0,0)])

class_labels = ('A','B','C','D','E','F','G',
                'H','I','J','K','L','M','N',
                'O','P','Q','R','S','T','U',
                'V','W','X','Y','Z','0')

net = posenet.load()

capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)

hand_position = 0,0,1,1
pyramid_iterations = 2

hand_detected = False

while True:
    ## Read a frame from the capture and flip it horizontally 
    ## and along the color channel to convert from BGR to RGB
    _, frame = capture.read()
    frame = np.flip(frame, 1)
    frame = np.flip(frame, 2)
    
    if not hand_detected:
        ## Make prediction map from a scaled image pyramid
        total_prediction = np.zeros_like(frame, dtype=float)
        
        for scaled_image in pyramid(frame, iterations=pyramid_iterations):
            prediction = posenet.forward(net, scaled_image / 255)
            prediction = colors[np.argmax(prediction, axis=2)]
            prediction = imresize(prediction, (height, width))
            
            total_prediction += prediction
        
        prediction = (total_prediction / pyramid_iterations).astype(np.uint8)
        
        threshold = (np.sum(prediction, axis=2) > 127).astype(np.uint8) * 255
        threshold = cv.dilate(threshold, np.ones((24,24)), iterations=3)
        
        ## Find contours and select contour containing a hand
        _, contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        largest_area = 0
        
        for index, contour in enumerate(contours):
            x,y,w,h = cv.boundingRect(contour)
            
            if w is 0 or h is 0: continue
            
            image = imresize(frame[y:y+h,x:x+w], (64,64)) / 255
            output = posenet.forward(net, image)  
            
            ## If the output is classified as a hand pose, check whether the area is the largest
            if np.argmax(output) < 26:
                if w*h > largest_area:
                    largest_area = w*h
                    
                    #cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    hand_position = x,y,w,h
                    hand_detected = True
    
    x,y,w,h = hand_position
    hand = imresize(frame[y:y+h,x:x+w], (64,64))
    
    output = posenet.forward(net, hand / 255)
    cv.putText(hand, class_labels[np.argmax(output)], (0,16), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
    if np.argmax(output) >= 26:
        hand_detected = False
    
    frame = np.flip(frame, 2)
    hand = np.flip(hand, 2)
    
    cv.imshow("Frame", frame)
    #cv.imshow("Prediction", prediction)
    #cv.imshow("Threshold", threshold)
    cv.imshow("Hand", hand)
    
    key = cv.waitKey(1)
    
    if key is ord('q'):
        break
    
capture.release()
cv.destroyAllWindows()