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

dtype = torch.cuda.FloatTensor

class_labels = ('A','B','C','D','E','F','G',
                'H','I','J','K','L','M','N',
                'O','P','Q','R','S','T','U',
                'V','W','X','Y','Z','0')

plot = 1

def accuracy(data, label, net):
    X = data.transpose(0,3,1,2)
    X = Variable(torch.from_numpy(X).type(dtype), requires_grad=False)
    y = net(X)
    y = np.exp(y.data.cpu().numpy())
    predictions = np.argmax(y, 1)
    
    for index, l in enumerate(class_labels):
        a = np.equal(label, index)  # items containing current label
        b = np.equal(predictions, label) # items containing correct predictions
        c = np.logical_and(a, b) # items containing correct predictions of current label
        
        if np.sum(a) > 0:
            print("%s: %.2f%% | %s/%s"%(l, np.sum(c) / np.sum(a), np.sum(c), np.sum(a)))
    
    print("Combined model accuracy: %s"%(np.sum(np.equal(predictions, label)) / len(label)))
    
    return predictions

def accuracy_double(data, label, net1, net2):
    X = data.transpose(0,3,1,2)
    X = Variable(torch.from_numpy(X).type(dtype), requires_grad=False)
    y1, _ = net1(X)
    y2, _ = net2(X)
    y = (y1 + y2) / 2
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
    global plot 
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=0.001)
    
    criterion = nn.NLLLoss()
    
    training_loss = []
    cv_loss = []
    
    batch_size = 50
    
    for epoch in range(epochs):
        total_loss = 0
        
        net.train()
        
        for i in range(0,len(data), batch_size):
            X = data[i:i+batch_size].transpose(0,3,1,2)
            X = Variable(torch.from_numpy(X).type(dtype), requires_grad=False)
            
            t = Variable(torch.from_numpy(label[i:i+batch_size]).cuda(), requires_grad=False)
            
            y = net(X)
            
            loss = criterion(y, t)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.data[0]
        
        loss = total_loss / len(list(range(0,len(data), batch_size)))
        training_loss.append(loss)
        
        net.eval()
        
        X = data_cv.transpose(0,3,1,2)
        X = Variable(torch.from_numpy(X).type(dtype), requires_grad=False)
        
        t = Variable(torch.from_numpy(label_cv).cuda(), requires_grad=False)
            
        y = net(X)
        
        loss = criterion(y, t)
        
        cv_loss.append(loss.data[0])
        
        print("Epoch %i | Training loss: %f | Validation loss: %f"%(epoch+1, training_loss[-1], cv_loss[-1]))
    
    plt.figure(plot)
    plt.plot(training_loss, 'b', cv_loss, 'r')
    
    plot += 1

dataset = preprocess.load()
dataset = preprocess.concatenate(dataset, preprocess.mirror(dataset))
dataset = preprocess.shuffle(dataset)

data, label = dataset

#label[(label >= 5) & (label < 26)] = 5
#label[(label == 26)] = 5

data /= 255
m = 8322
print("%i samples in total."%len(data))
data_cv = data[m:]
label_cv = label[m:]
data = data[:m]
label = label[:m]

print("Training samples: %i, validation samples: %i"%(len(data), len(data_cv)))

net1 = torch.load('models/posenet_00.model')
net2 = torch.load('models/posenet_01.model')
net = posenet.PoseNetMixture(net1, net2)
net.cuda()

predictions = accuracy(data_cv, label_cv, net1)

train(net, data, label, data_cv, label_cv, epochs=80, lr=1e-3)
train(net, data, label, data_cv, label_cv, epochs=20, lr=1e-4)

#train(net1, data, label, data_cv, label_cv, epochs=20)
#train(net1, data, label, data_cv, label_cv, epochs=10, lr=1e-6)
#train(net2, data, label, data_cv, label_cv, epochs=30)
#train(net2, data, label, data_cv, label_cv, epochs=20, lr=1e-5)

#print("Accuracy of model mixture")
#predictions = accuracy_double(data_cv, label_cv, net1, net2)
predictions = accuracy(data_cv, label_cv, net)

plt.figure(0)
for index, datum in enumerate(data_cv[:32]):
    plt.subplot(4,8,index+1)
    plt.imshow(datum)
    plt.axis('off')
    ground_truth = class_labels[label_cv[index]]
    prediction = class_labels[predictions[index]]
    plt.title("%s/%s"%(prediction, ground_truth))

'''
params = list(net1.parameters())[0].data.cpu().numpy().transpose(0,2,3,1)
params = (params - params.min()) / (params.max() - params.min())

plt.figure(2)
for i in range(40):
    plt.subplot(8,5,i+1)
    plt.imshow(params[i])
    plt.axis('off')
'''

#posenet.save(net)