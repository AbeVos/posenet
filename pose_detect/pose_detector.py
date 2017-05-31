#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:01:40 2017

@author: abe
"""

import cv2 as cv
import numpy as np
from scipy.misc import imresize

import posenet

width, height = 640, 480

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

net = posenet.load()

capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)

while True:
    _, frame = capture.read()
    frame = np.flip(frame, 1)
    frame = np.flip(frame, 2)
    
    prediction = posenet.forward(net, frame / 255)
    prediction = colors[np.argmax(prediction, axis=2)]
    prediction = imresize(prediction, (height, width), interp='nearest')
    
    threshold = (np.sum(prediction, axis=2) > 0).astype(float)
    
    #cv.rectangle(frame, (0, 0), (64, 64), (0,255,0))
    cv.imshow("Frame", np.flip(frame, 2))
    cv.imshow("Prediction", prediction)
    cv.imshow("Threshold", threshold)
    
    key = cv.waitKey(1)
    
    if key is ord('q'):
        break
    
capture.release()
cv.destroyAllWindows()