#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:20:26 2017

@author: abe
"""

import pygame as pg
import numpy as np

import game_manager as game

class Actor(object):
    '''Base class for all game objects.'''
    def __init__(self, position):
        self.is_active = True
        
        self.set_surface(1,1)
        
        self.image = pg.Surface((1,1))
        self.image_rect = self.image.get_rect()
        
        self.set_position(position)
        
        self.children = []
    
    def update(self, delta):
        if not self.is_active: return
        
        for child in self.children:
            child.update(delta)
    
    def draw(self, surface):
        if not self.is_active: return
        
        self.rect.center = self.position
        
        self.surface.fill(pg.Color(0,0,0,0))
        self.surface.blit(self.image, self.image_rect)
        
        for child in self.children:
            child.draw(surface)
        
        surface.blit(self.surface, self.rect)
        
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
        self.position = np.array(position, dtype=float)
        
class AnimatedActor(Actor):
    def __init__(self, position, mode='loop', delay=0):
        super(AnimatedActor, self).__init__(position)
        
        self.mode = mode
        self.delay = delay
        
    def update(self, delta):
        super(AnimatedActor, self).update(delta)
        
        if self.mode is 'loop':
            if self.frame_time <= 1:
                self.frame_time += delta / self.animation_length
                self.current_frame = int(self.frame_time * float(self.n_frames - 1))
                
                if self.current_frame >= self.n_frames:
                    self.current_frame = self.n_frames - 1
                
                self.set_image(self.frames[self.current_frame])
            else:
                self.delay_time += delta
                
                if self.delay_time >= self.delay:
                    self.delay_time = 0
                    self.frame_time = 0
        elif self.mode is 'pingpong':
            if self.frame_time <= 1:
                self.frame_time += delta / self.animation_length
                self.current_frame = int(self.frame_time * float(self.n_frames - 1))
                
                if self.current_frame >= self.n_frames:
                    self.current_frame = self.n_frames - 1
                
                if self.pingpong_to:
                    self.set_image(self.frames[self.current_frame])
                else:
                    self.set_image(self.frames[self.n_frames - self.current_frame - 1])
            else:
                self.delay_time += delta
                
                if self.delay_time >= self.delay:
                    self.delay_time = 0
                    self.frame_time = 0
                
                    self.pingpong_to = not self.pingpong_to
        elif self.mode is 'one_shot':
            if self.frame_time <= 1:
                self.frame_time += delta / self.animation_length
                self.current_frame = int(self.frame_time * float(self.n_frames - 1))
                
                if self.current_frame >= self.n_frames:
                    self.current_frame = self.n_frames - 1
                
                self.set_image(self.frames[self.current_frame])
        
    def set_animation(self, frames, frame_size, n_frames, animation_length=1):
        self.frames = frames
        self.frame_size = frame_size
        self.n_frames = n_frames
        self.animation_length = animation_length
        
        self.current_frame = 0
        self.frame_time = 0
        self.delay_time = 0
        self.pingpong_to = True
    
    def start_animation(self):
        self.current_frame = 0
        self.frame_time = 0
        self.delay_time = 0
        self.pingpong_to = True
        self.set_image(self.frames[0])