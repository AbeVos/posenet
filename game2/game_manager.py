#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:22:53 2017

@author: abe
"""

import pygame as pg
import pygame.locals as local

from event import Event

class State():
    def __init__(self):
        self.background_color = pg.Color(127, 255, 63)
    
    def update(self, delta):
        pass
    
    def draw(self, surface):
        pass

def init():
    global running, screen_size, global_states, global_state, screen, clock, update, draw, key_down, global_state_changed, game_quit
    
    running = True
    screen_size = (640, 480)
    
    global_states = {}
    global_state = 'No state selected'
    
    pg.init()
    
    screen = pg.display.set_mode(screen_size)
    clock = pg.time.Clock()
    pg.display.set_caption("Gebarentaal")
    
    load_resources()
    
    ## Global events
    update = Event()
    draw = Event()
    key_down = Event()
    global_state_changed = Event()
    game_quit = Event()

def set_global_states(all_global_states, current_state):
    global global_states, global_state
    global_states = all_global_states
    global_state = current_state

def start():
    global running, global_states, global_state, screen, clock, update, draw, key_down
    
    while running:
        update(clock.get_rawtime() / 1000)
        
        screen.fill(global_states[global_state].background_color)
        draw(screen)
        
        pg.display.update()
        clock.tick(60)
        
        for event in pg.event.get():
            if event.type == local.QUIT:
                stop()
                break
                
            elif event.type == local.KEYDOWN:            
                if event.key == local.K_ESCAPE:
                    pg.event.post(pg.event.Event(local.QUIT))
                else:
                    key_down(event.key)
                
                if event.key == ord('q'):
                    set_global_state('tutorial')
    
    print("Quit application")

def load_resources():
    global fonts, images
    
    fonts = {
        'title': pg.font.Font('resources/fonts/FreeSans.ttf', 64),
        'screen_large': pg.font.Font('resources/fonts/Renegade Master.ttf', 64),
        'screen_small': pg.font.Font('resources/fonts/Renegade Master.ttf', 16)
        }

    images = {
        'block_blue': pg.image.load('resources/images/block_blue.png'),
        'cursor_block': pg.image.load('resources/images/cursor_block.png')
        }

def get_font(font):
    global fonts
    return fonts[font]

def get_image(image):
    global images
    return images[image]

def get_screen_size():
    global screen_size
    return screen_size
 
def set_global_state(new_state):
    global global_states, global_state, global_state_changed
    
    previous_state = global_state
    global_state = new_state
    
    print("Change global state from %s to %s."%(previous_state, new_state))
    
    global_state_changed(previous_state, new_state)
      
def get_global_state():
    global global_states, global_state
    return global_states[global_state]
             
def stop():
    global running, game_quit
    game_quit()
    pg.quit()
    running = False