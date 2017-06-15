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

from posenet import PoseNet, AE, PoolAE, DeepAE

dtype = torch.cuda.FloatTensor

def pretrain_layers(layers, epochs, data):
    learning_graphs = []

    for index, layer in enumerate(layers):
        print("Train layer %i"%(index+1))
        
        layer.cuda()
        layer.train()
        
        optimizer = torch.optim.RMSprop(layer.parameters(), lr=0.0001)
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
        
    return learning_graphs

def finetune(deep_net, epochs, data):
    training_loss = []
    
    optimizer = torch.optim.RMSprop(deep_net.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    print("Finetune deep AE")
    for epoch in range(epochs):
        
        total_loss = 0
        
        for datum in data:
            X = torch.from_numpy(datum.transpose(2,0,1)).unsqueeze(0).type(dtype)
            X = Variable(X, requires_grad=False)
            
            corruption = Variable(torch.randn(X.size()).type(dtype), requires_grad=False)
            h = X + 0.1 * corruption
            y = deep_net(h)
            
            loss = criterion(y, X)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.data[0]
        
        total_loss /= len(data)
        print("Epoch %i | Training loss: %f"%(epoch+1, total_loss))
        
        training_loss.append(total_loss)
    
    return training_loss

def train():
    layers = [PoolAE(3, 40, 7, stride=3, padding=6),
              PoolAE(40, 80, 5),
              PoolAE(80, 160, 3),
              AE(160, 256, 1),
              AE(256, 256, 1)]
    
    voc_data = np.load('data/pretrain_data.npy')
    pose_data = np.load('data/pose_data.npy')
    google_data = np.load('data/google_data.npy')
    
    data = np.concatenate((voc_data, pose_data, google_data)) / 255
    data = np.concatenate((data, np.flip(data, 2)))
    np.random.shuffle(data)
    
    data_cv = data[-10:]
    #data = data[:30000]     # posenet_00
    data = data[10000:-10] # posenet_01
    
    print(data.shape)
    
    epochs = 15
    
    learning_graphs = pretrain_layers(layers, epochs, data)
    
    deep_net = DeepAE(layers)
    deep_net.cuda()
    deep_net.train()
    
    training_loss = finetune(deep_net, 30, data)
    
    learning_graphs.append(np.array(training_loss))
    
    plt.figure(0)
    for index, graph in enumerate(learning_graphs):
        plt.subplot(2, 4, index+1)
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
    for i in range(40):
        plt.subplot(8,5,i+1)
        plt.imshow(params[i])
        plt.axis('off')
        
    posenet = PoseNet(deep_net)
    #print(posenet)
    
    torch.save(posenet, 'models/posenet_01.model')

train()
