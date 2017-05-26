#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:20:26 2017

@author: abe
"""

import pygame as pg
import numpy as np
import game_manager as game
import util

class Actor(object):
    def __init__(self, position):
        self.image = pg.Surface((64,64))
        self.image_rect = self.image.get_rect()
        
        self.surface = pg.Surface((1,1))
        self.rect = self.surface.get_rect()
        
        self.is_active = True
        
        self.children = []
        
        self.set_position(np.array([0,0]))
        self.target_position = self.position
        
        self.at_target = True
        
        self.time = 0
        
        game.game_state_changed.subscribe(self.game_state_changed)
        
    def __del__(self):
        game.game_state_changed.unsubscribe(self.game_state_changed)
        
    def update(self, delta):
        if not self.is_active: return
        
        if util.distance(self.position, self.target_position) < 1:
            self.reach_target()
        
        self.rect.center = self.position
    
        for child in self.children:
            child.update(delta)
    
    def draw(self, surface):
        if not self.is_active: return
        self.surface.fill(pg.Color(0,0,0,0))
        
        self.surface.blit(self.image, self.image_rect)
        
        for child in self.children:
            child.draw(self.surface)
        
        surface.blit(self.surface, self.rect)
    
    def game_state_changed(self, prev_state, new_state):
        self.time = 0
    
    def set_image(self, image):
        self.image = image
        self.image_rect = self.image.get_rect()
        
        if self.image.get_width() > self.surface.get_width():
            self.set_surface(self.image.get_width(), self.surface.get_height())
            
        if self.image.get_height() > self.surface.get_height():
            self.set_surface(self.surface.get_width(), self.image.get_height())
    
    def set_surface(self, width, height):
        self.surface = pg.Surface((width, height), flags=pg.SRCALPHA)
        self.rect = self.surface.get_rect()
    
    def set_position(self, position):
        self.position = np.array(position)
        self.target_position = self.position
        
    def get_position(self):
        return self.position
    
    def set_active(self, active):
        self.is_active = active
        
        for child in self.children:
            child.set_active(active)
    
    def set_target(self, position):
        self.target_position = np.array(position)
        self.at_target = False
        
    def reach_target(self):
        self.set_position(self.target_position)
        self.at_target = True