#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:36:49 2017

@author: abe
"""

import numpy as np
import cv2 as cv
import detector
from mlutil import image, misc
import scipy.misc

cap = cv.VideoCapture(0)
frame_width, frame_height = 320, 240
cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv.CAP_PROP_FPS, 60)

stride = 8
iterations = 5

while True:
    _, frame = cap.read()
    frame = cv.flip(frame, 1)
    frame = np.flip(frame, axis=2) / 256
    
    frame = cv.blur(frame, (3,3))
    
    frames = np.array([i for i in image.pyramid(frame, iterations=iterations, ratio=1.2)])
    
    detection = np.zeros(frame.shape)
    for f in frames:
        windows, dims = image.conv_preprocess(f, (32,32), stride=stride)
        out = detector.evaluate(windows)
        #out = misc.one_hot(out, max_value=2).reshape((dims[0], dims[1], 3))
        out = out.reshape((dims[0], dims[1], 3))
        out = scipy.misc.imresize(out, frame.shape).astype(float)
        detection += out
    
    detection /= iterations
    detection = np.clip(detection[:,:,2] - detection[:,:,1], 0, 255)
    detection = detection.astype(np.uint8)
    detection = cv.dilate(detection, np.ones((5,5)).astype('uint8'))
    ret, proposal = cv.threshold(detection, 64, 255, 0)
    proposal, contours, hierarchy = cv.findContours(proposal, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    
    for contour in contours:
        if cv.contourArea(contour) < 1:
            continue
        
        x,y,w,h = cv.boundingRect(contour)
        img = frame[y:y+h,x:x+w]
        img = scipy.misc.imresize(img, (32,32)) / 255
        img = np.expand_dims(img, axis=0)
        if detector.evaluate(img)[0][2] > 0.9:
            cv.imshow("?", img[0])
            cv.rectangle(frame, (x,y), (x+w,y+h), (1,1,1))
    
    #cv.drawContours(frame, contours, -1, (0,255,0), 3)
    cv.imshow("Frame", frame)
    cv.imshow("Detection", detection)
    cv.imshow("Proposal", proposal)
    
    cv.waitKey(1)

cap.release()
cv.destroyAllWindows()