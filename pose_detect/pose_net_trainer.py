#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 11:36:55 2017

@author: abe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

class AE(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride=1, padding=0):
        super(AE, self).__init__()
        
        self.encoder = nn.Conv2d(in_features, out_features, kernel, stride=stride, padding=padding)
        self.decoder = nn.ConvTranspose2d(out_features, in_features, kernel, stride=stride, padding=padding)
    
    def forward(self, x):
        x = F.relu(self.encoder(x))
        x = F.dropout(x, training=self.training)
        x = self.decoder(x)
        
        return x
    
    def encode(self, x):
        x = F.relu(self.encoder(x))
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

class PoseNet(nn.Module):
    def __init__(self, layers):
        super(PoseNet, self).__init__()
        
        for layer in layers:
            pass

dtype = torch.cuda.FloatTensor

layers = [PoolAE(3, 20, 7, stride=3, padding=6),
          PoolAE(20, 40, 5),
          PoolAE(40, 60, 3),
          AE(60, 80, 1),
          AE(80, 100, 1),
          AE(100, 26, 1)]

voc_data = np.load('data/pretrain_data.npy')
pose_data = np.load('data/pose_data.npy')
google_data = np.load('data/google_data.npy')

data = np.concatenate((voc_data, pose_data, google_data))
data = np.concatenate((data, np.flip(data, 2)))
data /= 255
np.random.shuffle(data)

data_cv = data[-10:]
data = data[:-10]
print(data.shape)

epochs = 10

learning_graphs = []

for index, layer in enumerate(layers):
    print("Training layer %i"%(index+1))
    
    layer.cuda()
    layer.train()
    
    optimizer = torch.optim.SGD(layer.parameters(), 0.01, 0.9)
    criterion = torch.nn.MSELoss()
    
    training_loss = []

    for epoch in range(epochs):
        total_loss = 0
        
        for datum in data:
            X = torch.from_numpy(datum.transpose(2,0,1)).unsqueeze(0).type(dtype)
            X = Variable(X, requires_grad=False)
        
            for i in range(index):
                X = layers[i].encode(X)
                X = Variable(X.data, requires_grad=False)
        
            corruption = Variable(torch.randn(X.size()).type(dtype), requires_grad=False)
            
            y = layer(X + 0.1 * corruption)
            
            loss = criterion(y, X)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.data[0]
        
        total_loss /= len(data)
        print("Epoch %i | Training loss: %f"%(epoch+1, total_loss))
        
        training_loss.append(total_loss)
    
    learning_graphs.append(np.array(training_loss))

training_loss = []

print("Finetune deep AE")
for epoch in range(epochs):
    
    total_loss = 0
    
    for datum in data:
        X = torch.from_numpy(datum.transpose(2,0,1)).unsqueeze(0).type(dtype)
        X = Variable(X, requires_grad=False)
        
        corruption = Variable(torch.randn(X.size()).type(dtype), requires_grad=False)
        h = X + 0.1 * corruption
        for layer in layers:
            h = layer.encode(h)
        
        for layer in reversed(layers):
            h = layer.decode(h)
        y = h
        
        loss = criterion(y, X)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.data[0]
    
    total_loss /= len(data)
    print("Epoch %i | Training loss: %f"%(epoch, total_loss))
    
    training_loss.append(total_loss)

learning_graphs.append(np.array(training_loss))

plt.figure(0)
for index, graph in enumerate(learning_graphs):
    plt.subplot(1, 1, 1)
    plt.plot(graph)

X = torch.from_numpy(data_cv.transpose(0,3,1,2)).type(dtype)
X = Variable(X, requires_grad=False)
print(X.size())

h = X
for layer in layers:
    h = layer.encode(h)
    
print(h.size()) 

for layer in reversed(layers):
    h = layer.decode(h)

y = h.data.cpu().numpy().transpose(0,2,3,1)
X = X.data.cpu().numpy().transpose(0,2,3,1)

plt.figure(1)
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(X[i])
    plt.axis('off')
    
    plt.subplot(2,10,i+1+10)
    plt.imshow((y[i] - y[i].min()) / (y[i].max() - y[i].min()))
    plt.axis('off')
    
params = list(layers[0].parameters())[0].data.cpu().numpy().transpose(0,2,3,1)
params = (params - params.min()) / (params.max() - params.min())

plt.figure(2)
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.imshow(params[i])
    plt.axis('off')