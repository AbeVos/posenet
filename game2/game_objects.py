#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 13:45:54 2017

@author: abe
"""

import pygame as pg
import numpy as np
import random

import game_manager as game
from actor import Actor
from text import Line
import util

class BlockManager(Actor):
    def __init__(self, position):
        super(BlockManager, self).__init__(position)
        
        self.level_width = 8
        self.set_surface(self.level_width * 128, 320)
        
        self.load_level()
        
        #self.cursor = Cursor((self.surface.get_width() / 2, 128))
        self.cursors = [Cursor((self.level_width / 2 * 128 - 64 + 128 * i, 128), self.activate_cursor, i) for i in range(2)]
        self.set_cursor_keys()
        
        self.state = 'input'
        self.state_time = 0
        
        self.is_finished = False
        
        game.key_down.subscribe(self.key_down)
    
    def update(self, delta):
        super(BlockManager, self).update(delta)
        
        if self.state == 'move':
            if self.state_time >= 1:
                self.set_state('input')
        
        for block in self.blocks:
            block.update(delta)
        
        for cursor in self.cursors:
            cursor.update(delta)
            
        self.state_time += delta
    
    def draw(self, surface):
        if not self.is_active: return
        
        self.rect.center = self.position
        
        self.surface.fill(pg.Color(0,0,0,0))
        
        self.surface.blit(self.image, self.image_rect)
        
        for block in self.blocks:
            block.draw(self.surface)
        
        for cursor in self.cursors:
            cursor.draw(self.surface)
        
        surface.blit(self.surface, self.rect)
    
    def key_down(self, key):
        key = key - 48
        
        self.delete_block(key)
        self.set_state('move')
    
    def activate_cursor(self, index):
        if not self.state == 'input': return
        
        self.blocks[index + 4].increase_mode(1)
        
        while True:
            group = self.find_group()
            
            if group:
                for index in group:
                    self.delete_block(index)
            else:
                break
            
        self.loop()
        self.set_state('move')
    
    def find_group(self):
        print("find_group")
        indices = []
        
        for index, block in enumerate(self.blocks[1:9]):
            
            print(len(self.blocks),index+2)
            if block.mode is self.blocks[index+2].mode:
                indices.append(index+1)
            elif len(indices) >= 3:
                print(len(indices))
                return indices
            else:
                print(len(indices))
                indices = []
        
        return None
    
    def loop(self):
        print("loop")
        block = self.blocks[0]
        block.set_position((8 * 128 - 64, 256))
        self.blocks.pop(0)
        self.blocks.append(block)
        print(len(self.blocks))
    
    def set_cursor_keys(self):
        keys = random.sample(game.get_property('available_letters'), len(self.cursors))
        
        for index, cursor in enumerate(self.cursors):
            cursor.set_key(keys[index])
    
    def set_block_targets(self):
        for index, block in enumerate(self.blocks):
            block.set_target((index * 128 - 64, 256))
    
    def delete_block(self, index):
        if index < len(self.blocks):
            self.blocks.pop(index)
            
    def set_state(self, new_state):
        previous_state = self.state
        self.state = new_state
        
        if new_state == 'move':
            self.set_block_targets()
        elif new_state == 'input':
            self.set_cursor_keys()
        
        for block in self.blocks:
            block.set_state(new_state)
        
        self.state_time = 0
        
    def reset(self):
        self.is_finished = False
        self.load_level()
        self.set_state('input')
        
    def load_level(self):
        self.level = [0,1,2,0,1,2,0,1,2,0,1,2]
        self.blocks = [Block((index * 128 - 64, 256), mode=mode) for index, mode in enumerate(self.level)]

class Cursor(Actor):
    """Block that takes input from the detector and passes this to the BlockManager"""
    def __init__(self, position, listener, index=0):
        super(Cursor, self).__init__(position)
        
        self.listener = listener
        self.index = index
        
        self.key = None
        self.set_image(game.get_image('cursor_block'))
        
        self.text = Line((64, 64), self.key, game.get_font('screen_large'))
        self.text.set_color(pg.Color(62, 255, 4))
        
        self.state = 'input'
        
        game.key_down.subscribe(self.key_down)
        
    def update(self, delta):
        super(Cursor, self).update(delta)
            
        self.text.update(delta)
        
        if self.state == 'remove':
            pass
    
    def draw(self, surface):
        if not self.is_active: return
        
        self.rect.center = self.position
        
        self.surface.fill(pg.Color(0,0,0,0))
        
        self.surface.blit(self.image, self.image_rect)
        
        self.text.draw(self.surface)
        
        surface.blit(self.surface, self.rect)
    
    # TODO: Create detector activator
    def key_down(self, key):
        if self.state == 'input':
            if key == ord(self.key):
                self.listener(self.index)
                
    def set_key(self, key):
        self.key = key
        self.text.set_text(str(key))

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
