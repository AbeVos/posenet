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

def untrained_net():
    layers = [PoolAE(3, 40, 7, stride=3, padding=6),
              PoolAE(40, 80, 5),
              PoolAE(80, 160, 3),
              AE(160, 256, 1),
              AE(256, 256, 1),
              AE(256, 27, 1)]
    
    deep_net = DeepAE(layers)
    deep_net.cuda()
    
    posenet = PoseNet(deep_net)
    
    return posenet

class AE(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride=1, padding=0):
        super(AE, self).__init__()
        
        self.encoder = nn.Conv2d(in_features, out_features, kernel, stride=stride, padding=padding)
        self.decoder = nn.ConvTranspose2d(out_features, in_features, kernel, stride=stride, padding=padding)
    
    def forward(self, x):
        x = self.encoder(x)
        x = F.dropout(x, training=self.training)
        x = self.decoder(x)
        
        return x
    
    def encode(self, x):
        x = F.leaky_relu(self.encoder(x))
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x
        

class PoolAE(AE):
    def __init__(self, in_features, out_features, kernel, stride=1, padding=0):
        super(PoolAE, self).__init__(in_features, out_features, kernel, stride, padding)
        
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
    
    def forward(self, x):
        x = self.encode(x)
        x = F.dropout(x, training=self.training)
        x = self.decode(x)
        
        return x
    
    def encode(self, x):
        x, i = self.pool(F.leaky_relu(self.encoder(x)))
        self.index = i
        return x
    
    def decode(self, x):
        x = self.decoder(self.unpool(x, self.index))
        return x

class DeepAE(nn.Module):
    def __init__(self, layers):
        super(DeepAE, self).__init__()
        
        self.conv1 = layers[0].encoder
        self.pool1 = layers[0].pool
        
        self.conv2 = layers[1].encoder
        self.pool2 = layers[1].pool
        
        self.conv3 = layers[2].encoder
        self.pool3 = layers[2].pool
        
        self.fc1 = layers[3].encoder
        self.fc2 = layers[4].encoder
        self.fc3 = layers[5].encoder
        
        self.t_fc3 = layers[5].decoder
        self.t_fc2 = layers[4].decoder
        self.t_fc1 = layers[3].decoder
        
        self.unpool3 = layers[2].unpool
        self.t_conv3 = layers[2].decoder
        
        self.unpool2 = layers[1].unpool
        self.t_conv2 = layers[1].decoder
        
        self.unpool1 = layers[0].unpool
        self.t_conv1 = layers[0].decoder
        
    def forward(self, x):
        x, index1 = self.pool1(F.leaky_relu(self.conv1(x)))
        x, index2 = self.pool2(F.leaky_relu(self.conv2(x)))
        x, index3 = self.pool3(F.leaky_relu(self.conv3(x)))
        
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        
        x = self.t_fc3(x)
        x = self.t_fc2(x)
        x = self.t_fc1(x)
        
        x = self.t_conv3(self.unpool3(x, index3))
        x = self.t_conv2(self.unpool2(x, index2))
        x = self.t_conv1(self.unpool1(x, index1))
        
        return x

class PoseNet(nn.Module):
    def __init__(self, deep_net):
        super(PoseNet, self).__init__()
        
        self.conv1 = deep_net.conv1
        self.pool1 = deep_net.pool1
        
        self.conv2 = deep_net.conv2
        self.pool2 = deep_net.pool2
        
        self.dropout2d = nn.Dropout2d(0.3)
        
        self.conv3 = deep_net.conv3
        self.pool3 = deep_net.pool3
        
        self.fc1 = deep_net.fc1
        self.fc2 = deep_net.fc2
        self.fc3 = deep_net.fc3
        
        self.softmax = nn.LogSoftmax()
                
    def forward(self, x):
        x, _ = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.dropout2d(x)
        x, _ = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.dropout2d(x)
        x, _ = self.pool3(F.leaky_relu(self.conv3(x)))
        x = self.dropout2d(x)
        
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

class GateNet(nn.Module):
    def __init__(self, num_experts):
        super(GateNet, self).__init__()
    
        self.conv1 = nn.Conv2d(3, 10, 7, stride=3, padding=6)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, 5, stride=3, padding=0)
        self.conv3 = nn.Conv2d(20, 40, 3, stride=3)
        
        self.fc1 = nn.Conv2d(40, 80, 1)
        self.fc2 = nn.Conv2d(80, num_experts, 1)
        
        self.softmax = nn.LogSoftmax()
        
        self.num_experts = num_experts
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        
        x = x.transpose(1,2)
        x = x.transpose(2,3)
        x = x.contiguous()
        x = x.view(-1, self.num_experts)
        
        x = self.softmax(x)
        
        return x

class PoseNetMixture(nn.Module):
    def __init__(self, net1, net2):
        super(PoseNetMixture, self).__init__()
        
        #self.gatenet = GateNet(2)
        #self.gatenet.cuda()
        self.nets = nn.ModuleList([net1, net2])
    
    def forward(self, x):
        output_sum = None
        
        '''
        p = torch.exp(self.gatenet(x))
        
        for index, net in enumerate(self.nets):
            output, _ = net(x)
            print(torch.exp(p[:,index]).size(), output.size())
            print(torch.exp(p[:,index]).expand_as(output).size())
            output = p[:,index].expand_as(output) * output
            
            if output_sum is None:
                output_sum = output
            else:
                output_sum = output_sum + output
        
        x = output_sum
        
        return x, _'''
        
        for net in self.nets:
            output, _ = net(x)
            
            if output_sum is None:
                output_sum = output
            else:
                output_sum = output_sum + output
        
        x = output_sum / len(self.nets)
        
        return x, _

'''
dtype = torch.cuda.FloatTensor

image = np.random.randn(1,3,64,64)
image = Variable(torch.from_numpy(image).type(dtype), requires_grad=False)

net = GateNet(2)
net.cuda()

y = net(image)

print(y.size())
'''
