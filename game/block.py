#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:29:09 2017

@author: abe
"""

import random
from random import randint

import pygame as pg
import numpy as np

import game_manager as game
from actor import Actor, Player
from text import Line
import util

block_size = 128

class Block(Actor):
    def __init__(self, type=-1):
        super(Block, self).__init__((0,0))
        
        self.set_image(game.images['block_blue'])
        
        self.text = Line((0,0), str(type), game.fonts['screen_large'])
        self.text.set_color(pg.Color(50,240,0))
        self.text.set_position((block_size / 2, block_size / 2))
        self.children.append(self.text)
        
        self.type = type
        
        self.target_position = self.get_position()
    
    def type_increase(self, n):
        self.type += n
        
        if self.type > 3:
            self.type = 1
        elif self.type < 1:
            self.type = 3
        
        self.text.set_text(str(self.type), game.fonts['screen_large'])

class CursorBlock(Block):
    def __init__(self, value):
        super(Block, self).__init__((0,0))
        
        self.set_image(game.images['cursor_block'])
        
        self.letter_text = Line((0,0), '', game.fonts['screen_large'])
        self.letter_text.set_color(pg.Color(50,240,0))
        self.set_letter()
        
        if value < 0:
            action_text = '-%i'%value
        else:
            action_text = '+%i'%value
        
        self.action_text = Line((0,0), action_text, game.fonts['screen_small'])
        self.action_text.set_color(pg.Color(50,240,0))
        self.action_text.set_position((block_size / 2, 1.5 * block_size))
        
        self.children.append(self.letter_text)
        self.children.append(self.action_text)

    def set_letter(self):
        self.letter = random.choice(game.letters)
        
        self.letter_text.set_text(self.letter, game.fonts['screen_large'])
        self.letter_text.set_position((0.5 * block_size, 0.5 * block_size))

class Cursor(Actor):
    def __init__(self, position, offset=5):
        super(Cursor, self).__init__(position)
        
        self.offset = offset
        self.blocks = [CursorBlock(1), CursorBlock(1)]
        
        self.set_surface(block_size * len(self.blocks), 2 * block_size)
        self.set_position(position)
        
        self.assign_block_positions()
    
    def update(self, delta):
        super(Cursor, self).update(self)
        
        for block in self.blocks:
            if block is None: continue
            
            block.update(delta)
    
    def draw(self, surface):
        self.surface.fill(pg.Color(0,0,0,0))
        
        for block in self.blocks:
            block.draw(self.surface)
        
        surface.blit(self.surface, self.rect)
        
    def key_down(self, key):
        for index, block in enumerate(self.blocks):
            if key is ord(block.letter):
                ## Reset cursor letters
                return index
        return -1
    
    def assign_block_positions(self):
        for index, block in enumerate(self.blocks):
            if block is None: continue
            
            block.set_position((block_size / 2 + block_size * index, block_size))

class BlockLine(Actor):
    def __init__(self, position):
        super(BlockLine, self).__init__(position)
        
        self.offset = 5
        
        self.cursor = Cursor((self.offset * block_size, block_size), self.offset)
        self.blocks = [Block(type=randint(1,3)) for i in range(10)]
        
        self.set_surface(block_size * len(self.blocks), 3 * block_size)
        self.set_position(position)
        
        self.assign_block_positions()
    
    def update(self, delta):
        super(BlockLine, self).update(self)
        
        for block in self.blocks:
            if block is None: continue
            
            block.update(delta)
        
        self.cursor.update(delta)
    
    def draw(self, surface):
        self.surface.fill(pg.Color(0,0,0,0))
        
        for block in self.blocks:
            if block is None: continue
        
            block.draw(self.surface)
        
        self.cursor.draw(self.surface)
        
        surface.blit(self.surface, self.rect)
    
    def assign_block_positions(self):
        for index, block in enumerate(self.blocks):
            if block is None: continue
            
            block.set_position((block_size / 2 + block_size * index, 2 * block_size))
    
    def key_down(self, key):
        index = self.cursor.key_down(key)
        print(index)
        
        if index >= 0:
            self.blocks[index + self.offset - 1].type_increase(1)
            self.remove_block(0)

    def find_groups(self):
        return 0

    def remove_block(self, index):
        self.blocks[index] = None
    
    def update_position(self):
        for index, block in enumerate(self.blocks):
            if block is None:
                pass

'''
class BlockField(Actor):
    def __init__(self, position):
        super(BlockField, self).__init__(position)
        
        self.player= Player()
        self.player_position = [2, 0]
        
        self.blocks = [[Block(gravity=False), Block(), Block(gravity=False), Block(gravity=False), None],
                        [None, Block('A'), Block(), Block(), None],
                        [None, Block('B'), Block(), Block(), Block(gravity=False)],
                        [Block(gravity=False), Block(gravity=False), Block('B', gravity=False), Block('A'), None],
                        [None, None, Block(gravity=False), Block(gravity=False), None]]
        
        self.blocks[self.player_position[0]][self.player_position[1]] = self.player
        
        self.set_surface(pg.Surface((64 * len(self.blocks[0]),
                                         64 * len(self.blocks)), flags=pg.SRCALPHA))
        
        self.set_position(position)
        
        self.assign_block_positions()
        self.update_fall()
        
        self.level_finished = False
        
    def update(self, delta):
        super(BlockField, self).update(self)
        
        for row in self.blocks:
            for block in row:
                if not block is None:
                    block.update(delta)
        
        if self.level_finished and self.player.at_target:
            game.set_state('tutorial')
                    
    def draw(self, surface):
        self.surface.fill(pg.Color(0,0,0,0))
        
        for row in self.blocks:
            for block in row:
                if not block is None:
                    block.draw(self.surface)
        
        surface.blit(self.surface, self.rect)
    
    def key_down(self, key):
        if self.level_finished: return
        
        for blocks in enumerate(self.blocks):
            block = blocks[1][self.player_position[1]+1]
            
            if not block is None and block.type.lower() == pg.key.name(key):
                self.blocks[blocks[0]][self.player_position[1]+1] = None
                self.update_fall()
                self.walk_forward()
                
                break
                               
    def update_fall(self):
        for row in reversed(list(enumerate(self.blocks[:-1]))):
            for block in enumerate(self.blocks[row[0]]):
                x,y = block[0], row[0]
                
                if (not block[1] is None and
                    block[1].gravity == True and
                    self.blocks[y+1][x] == None):
                        
                    self.blocks[y+1][x] = block[1]
                    self.blocks[y][x] = None
                               
        self.assign_block_targets()
    
    def walk_forward(self):
        """Check whether player character is able to move forward and, if so, move it 1 tile to the right."""
        r, c = self.player_position
        
        has_walked = False
        
        for i in range(-1,2):
            block = self.blocks[r+i][c+1]
            if block is None and not self.blocks[r+i+1][c+1] is None:
                
                player = self.blocks[r][c]
                self.blocks[r][c] = None
                
                r += i
                c += 1
                
                self.blocks[r][c] = player
                self.player_position = [r, c]
                
                self.assign_block_targets()
                
                has_walked = True
                break
            
        if has_walked:
            if c == len(self.blocks[0]) - 1:
                self.level_finished = True
            else:
                self.walk_forward()
    
    def assign_block_positions(self):
        for row in enumerate(self.blocks):
            for block in enumerate(self.blocks[row[0]]):
                if block[1] is None: continue
                x,y = block[0], row[0]
                
                block[1].set_position((32 + 64 * x, 32 + 64 * y))
    
    def assign_block_targets(self):
        for row in enumerate(self.blocks):
            for block in enumerate(self.blocks[row[0]]):
                if block[1] is None: continue
                x,y = block[0], row[0]
                
                block[1].set_target((32 + 64 * x, 32 + 64 * y))
'''