#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:23:11 2017

@author: abe
"""

import pygame as pg
from text import Line

pg.init()

screen = None

score = 0

fonts = {
        'title': pg.font.Font('resources/fonts/FreeSans.ttf', 64),
        'screen_large': pg.font.Font('resources/fonts/Renegade Master.ttf', 64),
        'screen_small': pg.font.Font('resources/fonts/Renegade Master.ttf', 16)
        }

images = {
        'player' : pg.image.load('resources/images/player.png'),
        'coin': pg.image.load('resources/images/coin.png'),
        'block_white': pg.image.load('resources/images/block_white.png'),
        'block_red': pg.image.load('resources/images/block_red.png'),
        'block_blue': pg.image.load('resources/images/block_blue.png'),
        'block_green': pg.image.load('resources/images/block_green.png'),
        'block_grey': pg.image.load('resources/images/block_grey.png'),
        'cursor_block': pg.image.load('resources/images/cursor_block.png')
        }

letters = ['a', 'b', 'c']

class State():
    def __init__(self):
        self.background = pg.Color(20, 20, 20)
        self.title = Line((screen.get_width() / 2, 100), "This is a state", fonts['title'])
    
    def update(self, delta):
        pass
    
    def draw(self):
        self.title.draw(screen)
    
    def key_down(self, key):
        pass
    
    def reset(self):
        self.__init__()

running = False

all_states = None
state = None
     
def create_screen(size, caption=""):
    """
    Create a new screen with given size and caption.
    Also initialize a clock for updating and starts game.
    """
    global screen, screen_size, clock, running
    
    screen = pg.display.set_mode(size)
    pg.display.set_caption("Gebarentaal")
    
    clock = pg.time.Clock()
    
    running = True
    
    return screen

def update():
    pg.display.update()
    clock.tick(60)
    
    return clock.get_rawtime() / 1000

def stop():
    global running
    print("Quit application")
    pg.quit()
    running = False
    
def set_state(new_state):
    global state
    
    state = all_states[new_state]()
    state.reset()