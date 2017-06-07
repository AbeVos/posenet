#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:23:56 2017

@author: abe
"""

import numpy as np
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt

n_frames = 15
offset = 15
frame_size = 256

sprite = np.empty((frame_size, n_frames * frame_size, 4))

for i in range(n_frames):
    sprite[:,i*frame_size:(i+1)*frame_size] = imread('sprites/%.4i.png'%(i+1 + offset))

plt.imshow(1 - sprite[:,:,:3])
imsave('sprites/sprite_b.png', sprite)