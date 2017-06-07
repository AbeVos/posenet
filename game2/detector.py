#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:27:04 2017

@author: abe
"""

import cv2 as cv
import numpy as np
from scipy.misc import imresize

import game_manager as game
from mlutil.image import pyramid
import posenet

def init():
    global width, height, class_labels, pyramid_iterations, net, capture, hand, score_threshold, best_rect, hand_frame
    
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
    
    game.update.subscribe(update)
    game.game_quit.subscribe(game_quit)
    
    score_threshold = 25
    best_rect = 0,0,1,1
    hand_frame = np.zeros((64,64,3))

def update(delta):
    global width, height, class_labels, pyramid_iterations, net, capture, hand, score_threshold, best_rect, frame, hand_frame
    
    ## Read a frame from the capture and flip it horizontally 
    ## and along the color channel to convert from BGR to RGB.
    _, frame = capture.read()
    frame = np.flip(frame, 1)
    frame = np.flip(frame, 2).copy()
    
    ## Take the average of the frame prediction at different scales.
    total_prediction = np.zeros_like(frame, dtype=float)
    
    for layer in pyramid(frame, iterations=pyramid_iterations, ratio=1.3):
            prediction = posenet.forward(net, layer / 255)
            
            # Invert the probability of prediction not being a hand to get 
            # map of general hand prediction.
            prediction = 1 - prediction[:,:,-1] 
            
            # Broadcast to three channels for upsampling.
            prediction = np.tile(prediction[...,None], 3)
            prediction = imresize(prediction, (height, width), interp='bilinear')
            
            total_prediction += prediction
    
    prediction = total_prediction / pyramid_iterations / 255
    
    ## Find contours and select contour containing a hand.
    threshold = (prediction[:,:,0] > 0.3).astype(np.uint8) * 255
    threshold = cv.dilate(threshold, np.ones((16, 16)), iterations=5)
    _, contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    best_score = 0
    
    for index, contour in enumerate(contours):
        x,y,w,h = cv.boundingRect(contour)
        
        #(x,y),r = cv.minEnclosingCircle(contour)
        #print(x,y,r)
        #x,y,w,h = np.max([x-r,y-r,2*r,2*r], dtype=int)
        #print(x,y,w,h)
        
        if w is 0 or h is 0: continue
    
        image = imresize(frame[y:y+h,x:x+w], (64,64))
        output = posenet.forward(net, image / 255)
        
        if np.argmax(output) < 26:
            score = (output.max() + np.sum(prediction[y:y+h,x:x+w]) / (w + h) -
                     output[:,:,-1]**2)
            if score > best_score:
                best_score = score
                best_rect = x,y,w,h
    
    if best_score > score_threshold:
        x,y,w,h = best_rect
        #cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        hand_frame = imresize(frame[y:y+h,x:x+w], (64,64))
    
    #cv.imshow("Frame", np.flip(frame, 2) / 255)
    #cv.imshow("Prediction", prediction)
    #cv.imshow("Hand", np.flip(hand_frame, 2))
    
    cv.waitKey(1)

def current_position():
    global width, height, best_rect
    x,y,w,h = best_rect
    
    x = (2 * x + w) / 2 / width * game.get_screen_size()[0]
    y = (2 * y + h) / 2 / height * game.get_screen_size()[1]
    
    return np.array([x, y])

def current_pose():
    return ''

def get_hand_frame(size):
    global frame, best_rect
    x,y,w,h = best_rect
    hand = imresize(frame[y:y+h,x:x+w], size)
    return hand.astype(np.uint8).transpose(1,0,2)

def game_quit():
    print("Release capture")
    capture.release()
    cv.destroyAllWindows()