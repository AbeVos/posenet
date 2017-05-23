#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:27:04 2017

@author: abe
"""

import numpy as np
import torch
from torch.autograd import Variable
from mlutil import neuralnet

net = neuralnet.load("models/hand_detector_04.model")

def evaluate(x):
    x = x.reshape((-1, 32, 32, 3))
    x = x.transpose(0, 3, 1, 2)
    x = Variable(torch.from_numpy(x).type(neuralnet.dtype), volatile=True)
    x = net.forward(x).data.cpu().numpy()
    #return np.argmax(x, axis=1)
    return x