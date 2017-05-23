#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:32:41 2017

@author: abe
"""

import os
import numpy as np
from scipy.misc import imread, imresize

images = []

for path, dirs, files in os.walk('data/VOCdevkit/VOC2012/JPEGImages'):
    for file in files:
        image = imread(path + '/' + file)
        image = imresize(image, (64, 64))
        
        images.append(image)
        
data = np.array(images)

print(data.shape)