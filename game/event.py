#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:41:04 2017

@author: abe
"""

class Event():
    def __init__(self):
        self.listeners = []
    
    def __call__(self, *args):
        for listener in self.listeners:
            listener(*args)
    
    def subscribe(self, callback):
        if not callback in self.listeners:
            print("Add new listener %s"%callback)
            self.listeners.append(callback)
    
    def unsubscribe(self, callback):
        if callback in self.listeners:
            print("Delete listener")
            self.listeners.remove(callback)