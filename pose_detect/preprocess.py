#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:43:17 2017

@author: abe
"""

import os
import numpy as np
import matplotlib.pyplot as plt

class_labels = ('A','B','C','D','E','F','G',
                'H','I','J','K','L','M','N',
                'O','P','Q','R','S','T','U',
                'V','W','X','Y','Z','0')

def initialize():
    data, label = combine()
    np.save("data/pose_data", data)
    np.save("data/pose_label", label.astype(int))

def combine():
    data_total = np.empty((0,64,64,3))
    label_total = np.empty((0))
    
    i = 0
    while os.path.isfile("data/pose_%02i"%i):
        data = np.load("data/pose_%02i_data.npy"%i)
        label = np.load("data/pose_%02i_labels.npy"%i)
        
        data_total = np.concatenate((data_total, data))
        label_total = np.concatenate((label_total, label))
        
        i += 1
        
    return data_total, label_total

def load(index=-1):
    if index < 0:
        data = np.load("data/pose_data.npy")
        label = np.load("data/pose_label.npy")
    else:
        data = np.load("data/pose_%02i_data.npy"%index)
        label = np.load("data/pose_%02i_labels.npy"%index)
    
    return data, label

def mirror(dataset):
    data, label = dataset
    
    data = np.flip(data, axis=2)
    
    return data, label

def concatenate(A, B):
    data_A, label_A = A
    data_B, label_B = B
    
    data = np.concatenate((data_A, data_B), axis=0)
    label = np.concatenate((label_A, label_B))
    
    return data, label

def sort(dataset):
    data, label = dataset

def shuffle(dataset):
    data, label = dataset
    
    index = np.arange(len(data))
    np.random.shuffle(index)
    
    data = data[index]
    label = label[index]
    
    return data, label

def count_data(dataset):
    data, label = dataset
    
    for index, class_label in enumerate(class_labels):
        class_instances = np.equal(label, index)
        print("%s: %i"%(class_label, np.sum(class_instances)))
    print(len(label))

''' 
initialize()
dataset = load()
print(len(dataset[0]))
dataset = concatenate(dataset, mirror(dataset))
dataset = shuffle(dataset)

offset = 0

plt.figure(0)
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.title(class_labels[dataset[1][i+offset]], fontsize=10)
    plt.imshow(dataset[0][i+offset])
    plt.axis('off')
'''

#plt.tight_layout()

dataset = load()
count_data(dataset)
