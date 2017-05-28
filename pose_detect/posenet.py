#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:40:27 2017

@author: abe
"""

import torch.nn as nn
import torch.nn.functional as F

class PoseNet(nn.Module):
    def __init__(self, deep_net):
        super(PoseNet, self).__init__()
        
        self.conv1 = deep_net.conv1
        self.pool1 = deep_net.pool1
        
        self.conv2 = deep_net.conv2
        self.pool2 = deep_net.pool2
        
        self.conv3 = deep_net.conv3
        self.pool3 = deep_net.pool3
        
        self.fc1 = deep_net.fc1
        self.fc2 = deep_net.fc2
        self.fc3 = deep_net.fc3
        
        self.softmax = nn.LogSoftmax()
                
    def forward(self, x):
        x, _ = self.pool1(F.leaky_relu(self.conv1(x)))
        x, _ = self.pool2(F.leaky_relu(self.conv2(x)))
        x, _ = self.pool3(F.leaky_relu(self.conv3(x)))
        
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        
        x = x.view(-1, 27)
        
        x = self.softmax(x)
        
        return x