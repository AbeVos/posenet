#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:15:40 2017

@author: abe
"""

import pygame as pg
from actor import Actor

import game_manager as game

class Line(Actor):
    def __init__(self, position, text, font):
        super(Line, self).__init__(position)
        self.antialias = True
        self.color = pg.Color(255, 255, 255)
        
        self.font = font
        self.set_text(text)
        self.set_position(position)
        
    def draw(self, surface):
        super(Line, self).draw(surface)
        #print(self.position)
    
    def set_color(self, color):
        self.color = color
        self.set_text(self.text, self.font)
    
    def set_text(self, text):
        self.text = text
        
        self.set_image(self.font.render(text, self.antialias, self.color))
