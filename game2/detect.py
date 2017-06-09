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

class_labels = ('A','B','C','D','E','F','G',
                'H','I','J','K','L','M','N',
                'O','P','Q','R','S','T','U',
                'V','W','X','Y','Z','?')

class Detector():
    def __init__(self):
        super(Detector, self).__init__()
        
        self.screen_size = np.array((640, 480))
        
        self.pyramid_iterations = 3
        
        self.net = posenet.load()
        
        self.capture = cv.VideoCapture(0)
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, self.screen_size[0])
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, self.screen_size[1])
    
        game.update.subscribe(self.update)
        game.game_quit.subscribe(self.game_quit)
    
        self.score_threshold = 25
        self.positional_rect = 0,0,1,1
        self.frame = np.zeros((self.screen_size[0], self.screen_size[1], 3))
        self.hand_frame = np.zeros((64,64,3))
        self.current_pose = 26
        
        self.fixed_frame = None
        self.classification_scores = []
        self.wma_n = 10
        self.classification_moving_average = 0
    
    def update(self, delta):
        ## Read a frame from the capture and flip it horizontally 
        ## and along the color channel to convert from BGR to RGB.
        _, self.frame = self.capture.read()
        self.frame = np.flip(self.frame, 1)
        self.frame = np.flip(self.frame, 2).copy()
        
        ## Take the average of the frame prediction at different scales.
        total_prediction = np.zeros_like(self.frame, dtype=float)
        
        for layer in pyramid(self.frame, iterations=self.pyramid_iterations, ratio=1.3):
                prediction = posenet.forward(self.net, layer / 255)
                
                # Invert the probability of prediction not being a hand to get 
                # map of general hand prediction.
                prediction = 1 - prediction[:,:,-1] 
                
                # Broadcast to three channels for upsampling.
                prediction = np.tile(prediction[...,None], 3)
                prediction = imresize(prediction, (self.screen_size[1], self.screen_size[0]), interp='bilinear')
                total_prediction += prediction
        
        prediction = total_prediction / self.pyramid_iterations / 255
        
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
        
            image = imresize(self.frame[y:y+h,x:x+w], (64,64))
                
            output = posenet.forward(self.net, image / 255)
            
            largest_output = np.argmax(output)
            
            if largest_output < 26:
                score = (output.max() + np.sum(prediction[y:y+h,x:x+w]) / (w + h) -
                         output[:,:,-1])
                if score > best_score:
                    best_score = score
                    self.positional_rect = x,y,w,h
                    self.current_pose = largest_output
        
        if best_score > self.score_threshold:
            x,y,w,h = self.positional_rect
            self.hand_frame = imresize(self.frame[y:y+h,x:x+w], (64,64))
        
        if not self.fixed_frame is None:
            self.fixed_frame = imresize(self.fixed_frame, (64,64))
            output = posenet.forward(self.net, self.fixed_frame)
            
            self.classification_scores.append(output)
            
            if len(self.classification_scores) > self.wma_n:
                self.classification_scores.pop(0)
                
            weights = np.arange(len(self.classification_scores))
            self.classification_moving_average = np.sum(weights[::-1].reshape(-1,1,1,1) * np.array(self.classification_scores), axis=0)
            self.classification_moving_average = self.classification_moving_average / np.sum(weights)
            
            self.fixed_frame = None
        
        cv.imshow("Frame", np.flip(self.frame, 2) / 255)
        #cv.imshow("Prediction", prediction)
        #cv.imshow("Hand", np.flip(self.hand_frame, 2))
        
        cv.waitKey(1)
    
    def get_current_position(self):
        x,y,w,h = self.positional_rect
        
        x = (2 * x + w) / 2 / self.screen_size[0] * game.get_screen_size()[0]
        y = (2 * y + h) / 2 / self.screen_size[1] * game.get_screen_size()[1]
        
        return np.array([x, y])
    
    def get_current_pose(self):
        return class_labels[np.argmax(self.classification_moving_average)].lower()
    
    def get_positional_frame(self, output_size):
        x,y,w,h = self.positional_rect
        hand = imresize(self.frame[y:y+h,x:x+w], output_size)
        return hand.astype(np.uint8).transpose(1,0,2)
        
    def get_fixed_frame(self, frame_size, frame_position, output_size):
        frame_position = np.array(frame_position) / game.get_screen_size() * self.screen_size
        frame_position -= np.array(frame_size) / 2
        x,y = frame_position.astype(int)
        w,h = frame_size

        self.fixed_frame = self.frame[x:x+w,y:y+h]
        frame = imresize(self.fixed_frame, output_size)
        return frame.astype(np.uint8).transpose(1,0,2)     
    
    def game_quit(self):
        print("Release capture")
        self.capture.release()
        cv.destroyAllWindows()
        
def init():
    global detector
    
    detector = Detector()