#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 13:45:54 2017

@author: abe
"""

import pygame as pg
import numpy as np

import game_manager as game
from actor import Actor
from text import Line
import util

class BlockManager(Actor):
    def __init__(self, position):
        super(BlockManager, self).__init__(position)
        
        self.level = [0,1,2,0,1,2,0,1,2,0,1,2]
        
        self.set_surface(5 * 128, 320)
        self.blocks = [Block((index * 128 + 64, 256), mode=mode) for index, mode in enumerate(self.level)]
        
        self.cursor = Cursor((self.surface.get_width() / 2, 128))
        
        self.state = 'input'
        self.state_time = 0
        
        game.key_down.subscribe(self.key_down)
    
    def update(self, delta):
        super(BlockManager, self).update(delta)
        
        if self.state == 'move':
            if self.state_time >= 1:
                self.set_state('input')
        
        for block in self.blocks:
            block.update(delta)
            
        self.cursor.update(delta)
            
        self.state_time += delta
    
    def draw(self, surface):
        if not self.is_active: return
        
        self.rect.center = self.position
        
        self.surface.fill(pg.Color(0,0,0,0))
        
        self.surface.blit(self.image, self.image_rect)
        
        for block in self.blocks:
            block.draw(self.surface)
        
        self.cursor.draw(self.surface)
        
        surface.blit(self.surface, self.rect)
    
    def key_down(self, key):
        key = key - 49
        
        self.delete_block(key)
        self.set_state('move')
    
    def set_block_targets(self):
        for index, block in enumerate(self.blocks):
            block.set_target((index * 128 + 64, 64))
    
    def delete_block(self, index):
        if index < len(self.blocks):
            self.blocks.pop(index)
            
    def set_state(self, new_state):
        previous_state = self.state
        self.state = new_state
        
        if new_state == 'move':
            self.set_block_targets()
        
        for block in self.blocks:
            block.set_state(new_state)
        
        self.state_time = 0

class Cursor(Actor):
    def __init__(self, position):
        super(Cursor, self).__init__(position)
        
        self.set_image(game.get_image('cursor_block'))

class Block(Actor):
    def __init__(self, position, mode=1):
        super(Block, self).__init__(position)
        
        self.mode = mode
        
        self.set_image(game.get_image('block_red'))
        
        self.icon = Actor((64,64))
        self.icon.set_image(game.get_image('icon_%s'%self.mode))
        
        self.state = 'input'
        self.state_time = 0
        self.target = np.array((0,0))
        self.start_position = np.array((0,0))
    
    def update(self, delta):
        super(Block, self).update(delta)
        
        self.icon.update(delta)
        
        if self.state == 'move':
            self.position = util.lerp(self.start_position, self.target, self.state_time)
            
            self.state_time += 2 * delta
            
            if self.state_time >= 1:
                self.set_state('wait')
        
    def draw(self, surface):
        if not self.is_active: return
        
        self.rect.center = self.position
        
        self.surface.fill(pg.Color(0,0,0,0))
        
        self.surface.blit(self.image, self.image_rect)
        
        self.icon.draw(self.surface)
        
        surface.blit(self.surface, self.rect)
    
    def increase_mode(self, n):
        self.mode = (self.mode + n) % 3
        self.icon.set_image(game.get_image('icon_%s'%self.mode))
        
    def set_target(self, target):
        self.target = np.array(target)
        
    def set_state(self, new_state):
        previous_state = self.state
        self.state = new_state
        
        if new_state == 'move':
            self.start_position = self.position
        
        if previous_state == 'move':
            self.position = self.target
        
        self.state_time = 0
