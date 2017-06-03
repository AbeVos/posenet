#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:27:04 2017

@author: abe
"""
import cv2 as cv
import numpy as np
from scipy.misc import imresize

from mlutil.image import pyramid
import posenet

width, height = 640,480

class_labels = ('A','B','C','D','E','F','G',
                'H','I','J','K','L','M','N',
                'O','P','Q','R','S','T','U',
                'V','W','X','Y','Z','0')

pyramid_iterations = 3

net = posenet.load()

capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)

while True:
    ## Read a frame from the capture and flip it horizontally 
    ## and along the color channel to convert from BGR to RGB
    _, frame = capture.read()
    frame = np.flip(frame, 1)
    frame = np.flip(frame, 2)
    
    total_prediction = np.zeros_like(frame, dtype=float)
    
    for layer in pyramid(frame, iterations=pyramid_iterations, ratio=1.3):
            prediction = posenet.forward(net, layer / 255)
            prediction = 1 - prediction[:,:,-1]
            prediction = np.tile(prediction[...,None], 3)
            prediction = imresize(prediction, (height, width), interp='bilinear')
            
            total_prediction += prediction
    
    prediction = total_prediction / pyramid_iterations / 255
    
    cv.imshow("Frame", prediction * np.flip(frame, 2) / 255)
    cv.imshow("Prediction", prediction)
    
    key = cv.waitKey(1)
    
    if key is ord('q'):
        break
    
capture.release()
cv.destroyAllWindows()