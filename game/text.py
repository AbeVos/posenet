#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:15:40 2017

@author: abe
"""

import pygame as pg
from actor import Actor

class Line(Actor):
    def __init__(self, position, text, font):
        super(Line, self).__init__(position)
        self.antialias = True
        self.color = pg.Color(255, 255, 255)
        
        self.set_text(text, font)
        
        self.set_position(position)
        
    def draw(self, surface):
        super(Line, self).draw(surface)
        #print(self.position)
    
    def set_color(self, color):
        self.color = color
        self.set_text(self.text, self.font)
    
    def set_text(self, text, font):
        self.text = text
        self.font = font
        
        self.set_image(font.render(text, self.antialias, self.color))
        #self.rect = self.surface.get_rect()
        
        self.image_rect.center = (self.surface.get_width() / 2, self.surface.get_height() / 2)