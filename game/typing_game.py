#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:05:01 2017

@author: abe
"""

import pygame as pg
import pygame.locals as local

import game_manager as game
from game_manager import State
from block import BlockLine
from text import Score

class Tutorial(State):
    def __init__(self):
        super(Tutorial, self).__init__()
        
        self.background = pg.Color(20, 80, 20)
        self.title.set_text("Doe iets geks met je hand en misschien gebeuren er dingen", game.fonts['title'])
        self.title.set_position((screen.get_width() / 2, 200))

    def update(self, delta):
        super(Tutorial, self).update(delta)
        
        self.title.update(delta)
    
    def draw(self):
        super(Tutorial, self).draw()
     
    def key_down(self, key):
        game.set_state('level')
 
class Level(State):
    def __init__(self):
        super(Level, self).__init__()
        
        self.background = pg.Color(10, 20, 5)
        
        self.block_line = BlockLine((screen.get_width() / 2, screen.get_height() / 2))
        
        self.score = Score((500, screen.get_width() / 8))
    
    def update(self, delta):
        self.block_line.update(delta)
        self.score.update(delta)
    
    def draw(self):
        self.block_line.draw(screen)
        self.score.draw(screen)
    
    def key_down(self, key):
        self.block_line.key_down(key)

screen_size = (1920, 1080)
screen = game.create_screen(screen_size, "Gebarentaal")

game_states = {
        'tutorial': Tutorial,
        'level': Level
        }

game.all_states = game_states
game.set_state('tutorial')

while game.running: 
    delta = game.update()
    
    screen.fill(game.state.background)
    
    game.state.update(delta)
    game.state.draw()
    
    for event in pg.event.get():
        if event.type == local.QUIT:
            game.stop()
            break
            
        elif event.type == local.KEYDOWN:            
            if event.key == local.K_ESCAPE:
                pg.event.post(pg.event.Event(local.QUIT))
            else:
                game.state.key_down(event.key)