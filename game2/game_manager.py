#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:22:53 2017

@author: abe
"""

import numpy as np
import pygame as pg
import pygame.locals as local

from event import Event

class State():
    def __init__(self):
        self.background = get_image('bg_yellow')
        
        global_state_changed.subscribe(self.global_state_changed)
    
    def update(self, delta):
        pass
    
    def draw(self, surface):
        pass
    
    def global_state_changed(self, previous_state, new_state):
        pass

def init(width, height):
    global running, screen_size, global_states, global_state, screen, clock, update, draw, key_down, cursor_down, cursor_up, global_state_changed, game_quit, cursor_position
    
    running = True
    screen_size = np.array((width, height))
    
    global_states = {}
    global_state = 'No state selected'
    
    pg.init()
    
    screen = pg.display.set_mode(screen_size)
    clock = pg.time.Clock()
    pg.display.set_caption("Gebarentaal")
    
    load_resources()
    
    cursor_position = np.array([0,0], dtype=float)
    
    ## Global events
    update = Event()
    draw = Event()
    key_down = Event()
    cursor_down = Event()
    cursor_up = Event()
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
        
        #screen.fill(global_states[global_state].background_color)
        screen.blit(global_states[global_state].background, screen.get_rect())
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
                
                #if event.key == ord('q'):
                #    set_global_state('tutorial')
    
    print("Quit application")

def load_resources():
    global fonts, images, animations
    
    fonts = {
        'title': pg.font.Font('resources/fonts/FreeSans.ttf', 64),
        'screen_large': pg.font.Font('resources/fonts/Renegade Master.ttf', 64),
        'screen_small': pg.font.Font('resources/fonts/Renegade Master.ttf', 48)
        }

    images = {
        'block_blue': pg.image.load('resources/images/block_blue.png'),
        'cursor_block': pg.image.load('resources/images/cursor_block.png'),
        'button_down': pg.image.load('resources/images/button_down.png'),
        'button_up': pg.image.load('resources/images/button_up.png'),
        'cancel_down': pg.image.load('resources/images/cancel_down.png'),
        'cancel_up': pg.image.load('resources/images/cancel_up.png'),
        'cursor_standard': pg.image.load('resources/images/cursor_standard.png'),
        'cursor_press': pg.image.load('resources/images/cursor_press.png'),
        'bg_yellow': pg.image.load('resources/images/bg_yellow.png'),
        'screen_active': pg.image.load('resources/images/screen_active.png'),
        'screen_empty': pg.image.load('resources/images/screen_empty.png'),
        'empty': pg.image.load('resources/images/empty.png')
        }
    
    animations = {
        'pose_a': [pg.image.load('resources/images/animation/%.4i.png'%(i+1)) for i in range(15)],
        'pose_b': [pg.image.load('resources/images/animation/%.4i.png'%(i+1)) for i in range(15,30)],
        'pose_c': [pg.image.load('resources/images/animation/%.4i.png'%(i+1)) for i in range(30,45)],
        'pose_d': [pg.image.load('resources/images/animation/%.4i.png'%(i+1)) for i in range(45,60)],
        'pose_e': [pg.image.load('resources/images/animation/%.4i.png'%(i+1)) for i in range(60,75)],
        'load': [pg.image.load('resources/images/animation/load%.4i.png'%(i+1)) for i in range(20)]
        }

def get_font(font):
    global fonts
    return fonts[font]

def get_image(image):
    global images
    return images[image]

def get_animation(animation):
    global animations
    return animations[animation]

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
    
def set_cursor_position(position):
    global cursor_position
    cursor_position = np.array(position, dtype=float)

def get_cursor_position():
    global cursor_position
    return cursor_position