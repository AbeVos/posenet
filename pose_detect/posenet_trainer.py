#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 18:53:32 2017

@author: abe
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt

import preprocess
import posenet
import posenet_pretrainer

dtype = torch.cuda.FloatTensor

class_labels = ('A','B','C','D','E','F','G',
                'H','I','J','K','L','M','N',
                'O','P','Q','R','S','T','U',
                'V','W','X','Y','Z','0')

def accuracy(data, label, net):
    X = data.transpose(0,3,1,2)
    X = Variable(torch.from_numpy(X).type(dtype), requires_grad=False)
    y = net(X)
    predictions = np.argmax(np.exp(y.data.cpu().numpy()), 1)
    
    print(np.sum(np.equal(predictions, label)) / len(label))
    
    return predictions

def accuracy_double(data, label, net1, net2):
    X = data.transpose(0,3,1,2)
    X = Variable(torch.from_numpy(X).type(dtype), requires_grad=False)
    y = (net1(X) + net2(X)) / 2
    y = np.exp(y.data.cpu().numpy())
    predictions = np.argmax(y, 1)
    
    #print(np.argmax(y, 1))
    
    for index, l in enumerate(class_labels):
        a = np.equal(label, index)  # items containing current label
        b = np.equal(predictions, label) # items containing correct predictions
        c = np.logical_and(a, b) # items containing correct predictions of current label
        
        if np.sum(a) > 0:
            print("%s: %.2f%% | %s/%s"%(l, np.sum(c) / np.sum(a), np.sum(c), np.sum(a)))
    
    print("Combined model accuracy: %s"%(np.sum(np.equal(predictions, label)) / len(label)))
    
    return predictions

def train(net, data, label, data_cv, label_cv, lr=3e-5, epochs=30):
    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.00003, momentum=0.3)
    
    criterion = nn.NLLLoss()
    
    training_loss = []
    cv_loss = []
    
    accuracy(data_cv, label_cv, net)
    
    for epoch in range(epochs):
        total_loss = 0
        
        net.train()
        
        for i in range(len(data)):
            X = np.expand_dims(data[i], 0).transpose(0,3,1,2)
            X = Variable(torch.from_numpy(X).type(dtype), requires_grad=False)
            
            t = Variable(torch.from_numpy(np.expand_dims(label[i], 0)).cuda(), requires_grad=False)
            
            y = net(X)
            
            loss = criterion(y, t)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.data[0]
        
        loss = total_loss / len(data)
        training_loss.append(loss)
        
        net.eval()
        
        X = data_cv.transpose(0,3,1,2)
        X = Variable(torch.from_numpy(X).type(dtype), requires_grad=False)
        
        t = Variable(torch.from_numpy(label_cv).cuda(), requires_grad=False)
            
        y = net(X)
        
        loss = criterion(y, t)
        
        cv_loss.append(loss.data[0])
        
        print("Epoch %i | Training loss: %f | Validation loss: %f"%(epoch+1, training_loss[-1], cv_loss[-1]))
    
    accuracy(data_cv, label_cv, net)
    
    return training_loss, cv_loss

dataset = preprocess.load()
dataset = preprocess.concatenate(dataset, preprocess.mirror(dataset))
dataset = preprocess.shuffle(dataset)

data, label = dataset
data /= 255
m = 3000
print("%i samples in total."%len(data))
data_cv = data[m:]
label_cv = label[m:]
data = data[:m]
label = label[:m]

print("Training samples: %i, validation samples: %i"%(len(data), len(data_cv)))

net1 = torch.load('models/posenet_01.model')
net2 = posenet_pretrainer.untrained_net()

predictions = accuracy_double(data_cv, label_cv, net1, net2)

training_loss, cv_loss = train(net1, data, label, data_cv, label_cv, epochs=20)
train(net2, data, label, data_cv, label_cv, epochs=30)

print("Accuracy of model mixture")
predictions = accuracy_double(data_cv, label_cv, net1, net2)

plt.figure(0)
for index, datum in enumerate(data_cv[:32]):
    plt.subplot(4,8,index+1)
    plt.imshow(datum)
    plt.axis('off')
    ground_truth = class_labels[label_cv[index]]
    prediction = class_labels[predictions[index]]
    plt.title("%s/%s"%(prediction, ground_truth))
    
plt.figure(1)
plt.plot(training_loss, 'b', cv_loss, 'r')

'''
params = list(net1.parameters())[0].data.cpu().numpy().transpose(0,2,3,1)
params = (params - params.min()) / (params.max() - params.min())

plt.figure(2)
for i in range(40):
    plt.subplot(8,5,i+1)
    plt.imshow(params[i])
    plt.axis('off')
'''