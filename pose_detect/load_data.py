#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:25:18 2017

@author: abe
"""

import numpy as np
from scipy.io import loadmat
from scipy.misc import imread
import matplotlib.pyplot as plt
import mlutil.draw as draw

datasets = [["Buffy", 551], ["Inria", 561], ["Poselet", 390], ["Skin", 302], ["VOC2007", 820], ["VOC2010", 1445]]

def crop_image(image, threshold=0):
    grayscale = np.average(image, axis=2)
    mask = grayscale > threshold
    return image[np.ix_(mask.any(1), mask.any(0))]

def crop_indices(image, threshold=0):
    grayscale = np.average(image, axis=2)
    mask = grayscale > threshold
    return np.ix_(mask.any(1), mask.any(0))

def load_dataset(name, n):
    for i in range(n):
        a = loadmat("data/hand_dataset/training_dataset/training_data/alternate_annotations/%s_%i.mat"%(name, i+1))
        image = imread("data/hand_dataset/training_dataset/training_data/images/%s_%i.jpg"%(name, i+1))
        
        boxes = a['new_boxes']
        target = np.zeros(image.shape[:2], dtype=np.int)
        
        for b in range(len(boxes)):
            box = boxes[b].astype(int)
            
            center = np.mean(box, axis=0, dtype=np.int)
            
            target = draw.circle(target, center, 48)
        
        idx = crop_indices(image)
        image = image[idx].astype(float) / 255
        target = target[idx]

        yield image, target

def load_all_data():
    for dataset, dataset_size in datasets:
        for image, target in load_dataset(dataset, dataset_size):
            yield image, target

def data_length(dataset):
    return datasets[dataset][1]

def total_length():
    sum = 0
    
    for d in datasets:
        sum += d[1]
    
    return sum

#for image, target in load_dataset(datasets[0][0], 5):
#    plt.imshow(target)

