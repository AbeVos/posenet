#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 20:50:20 2017

@author: abe
"""

from random import randint

letters = ('a','b','c')
level = [randint(1,3) for i in range(10)]

idx = 0
cursor = letters[:2]
score = 0

def display_cursor():
    s = " "
    
    for i in range(idx):
        s += "  ,"

    for i in range(len(cursor)):
        s += cursor[i] + ", "
   
    print(s)
    
def find_group():
    global score
    count = 0
    
    for i in range(len(level)-1):
        if level[i] is level[i+1] and not level[i] is 0:
            count += 1
        else:
            if count >= 2:
                print("Group found!", count+1)
                for j in range(count+1):
                    level[i-j] = 0
                
                score += count+1

            count = 0

while idx < len(level) - 1:
    print("Score: " + str(score))
    display_cursor()
    print(level)

    value = input()
    
    if value is 'a':
        level[idx] += 1
        
        if level[idx] > 3:
            level[idx] = 1
            
    elif value is 'b':
        level[idx+1] += 1
        
        if level[idx+1] > 3:
            level[idx+1] = 1
    
    find_group()
    print()
    
    if idx is 5:
        level.append(randint(1,3))
        level.pop(0)
    else:
        idx += 1