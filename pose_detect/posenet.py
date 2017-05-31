#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:40:27 2017

@author: abe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

def load():
    model = torch.load("models/posenet.model")
    return model

def save(model):
    torch.save(model, "models/posenet.model")
    
def forward(model, x):
    if len(x.shape) is 3:
        x = np.expand_dims(x, 0)
    
    x = x.transpose(0,3,1,2)
    x = Variable(torch.from_numpy(x).type(torch.cuda.FloatTensor), requires_grad=True)
    
    model.eval()
    y, dims = model(x)
    y = y.view(dims[0], dims[1], -1)
    
    y = y.data.cpu().numpy()
    return np.exp(y)

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
        
        x = x.transpose(1,2)
        x = x.transpose(2,3)
        x = x.contiguous()
        output_size = x.size()[1:3]
        x = x.view(-1, 27)
        
        x = self.softmax(x)
        
        return x, output_size