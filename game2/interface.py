#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 12:57:52 2017

@author: abe
"""

import pygame as pg
import numpy as np

from actor import Actor
import game_manager as game
import detector
import util
from event import Event

class Cursor(Actor):
    def __init__(self):
        super(Cursor, self).__init__((100,100))
        
        self.set_image(game.get_image('cursor_standard'))
        
        self.velocity =  np.array((0,0))
        
        game.cursor_down.subscribe(self.cursor_down)
        game.cursor_up.subscribe(self.cursor_up)
        
    def __del__(self):
        game.cursor_down.unsubscribe(self.cursor_down)
        game.cursor_up.unsubscribe(self.cursor_up)
        
    def update(self, delta):
        super(Cursor, self).update(delta)
        
        #self.set_position(detector.current_position())
        
        target = detector.current_position()
        force = 10 * util.normalize(target - self.position)
        
        distance_modifier = np.min([100, util.distance(self.position, target)]) / 100
        self.velocity = distance_modifier * (force + 0.9 * self.velocity)
        
        #if np.linalg.norm(self.velocity) > 10:
        #    self.velocity = 10 * util.normalize(self.velocity)
        
        self.position += self.velocity
        
        if self.position[0] < 0:
            self.position[0] = 0
        elif self.position[0] > game.get_screen_size()[0]:
            self.position[0] = game.get_screen_size()[0]
            
        if self.position[1] < 0:
            self.position[1] = 0
        elif self.position[1] > game.get_screen_size()[1]:
            self.position[1] = game.get_screen_size()[1]
        
        game.set_cursor_position(self.position)
        
    def cursor_down(self, button_type=None):
        if button_type is None:
            self.set_image(game.get_image('cursor_press'))
        elif button_type is HandScreen:
            self.set_image(game.get_image('empty'))
        
    def cursor_up(self):
        self.set_image(game.get_image('cursor_standard'))

class Button(Actor):
    def __init__(self, position, listener):
        super(Button, self).__init__(position)
        
        self.button_up_image = game.get_image('button_up')
        self.button_down_image = game.get_image('button_down')
        self.set_image(self.button_up_image)
        
        self.is_pressed = False
        self.pressed_time = 0.0
        self.wait_time = 1.0
        
        self.button_pressed = Event()
        self.button_pressed.subscribe(listener)
    
    def update(self, delta):
        super(Button, self).update(delta)
        
        if self.cursor_over():
            if not self.is_pressed:
                self.press()
        
        elif self.is_pressed:
            self.unpress()
        
        if self.is_pressed:
            self.pressed_time += delta
        
            if self.pressed_time >= self.wait_time:
                self.button_pressed()
                self.pressed_time = 0
    
    def cursor_over(self):
        return util.distance(game.get_cursor_position(), self.position) < self.image.get_width() / 2
    
    def press(self):
        self.is_pressed = True
        self.set_image(self.button_down_image)
        game.cursor_down()
        
    def unpress(self):
        self.is_pressed = False
        self.pressed_time = 0.0
        self.set_image(self.button_up_image)
        game.cursor_up()

class CancelButton(Button):
    def __init__(self, position, listener):
        super(CancelButton, self).__init__(position, listener)
        
        self.button_up_image = game.get_image('cancel_up')
        self.button_down_image = game.get_image('cancel_down')
        self.set_image(self.button_up_image)
        self.set_surface(self.image.get_width(),self.image.get_height())
        
    def cursor_over(self):
        cursor_position = game.get_cursor_position()
        return (cursor_position[0] >= self.position[0] - self.image.get_width() / 2 and
                cursor_position[0] <= self.position[0] + self.image.get_width() / 2 and
                cursor_position[1] >= self.position[1] - self.image.get_height() / 2 and
                cursor_position[1] <= self.position[1] + self.image.get_height() / 2)

class HandScreen(Button):
    def __init__(self, position):
        super(Button, self).__init__(position)
        
        self.button_up_image = game.get_image('screen_empty')
        self.button_down_image = game.get_image('screen_active')
        self.set_image(self.button_up_image)
        
        self.is_pressed = False
        self.wait_time = 0
        
        self.hand_surface = pg.Surface((128,128))
        self.hand_rect = self.hand_surface.get_rect()
        
    def update(self, delta):
        super(Button, self).update(delta)
        
        if self.cursor_over():
            if not self.is_pressed:
                self.press()
        
        elif self.is_pressed:
            self.unpress()
    
    def draw(self, surface):
        if not self.is_active: return
        
        self.rect.center = self.position
        
        self.surface.fill(pg.Color(0,0,0,0))
        
        if self.is_pressed:
            self.hand_rect.center = (0,0)
            pg.surfarray.blit_array(self.hand_surface, detector.get_hand_frame((128, 128)))
            self.surface.blit(self.hand_surface, self.hand_surface.get_rect())
        
        self.surface.blit(self.image, self.image_rect)
        
        for child in self.children:
            child.draw(surface)
        
        surface.blit(self.surface, self.rect)
    
    def cursor_over(self):
        cursor_position = game.get_cursor_position()
        return (cursor_position[0] >= self.position[0] - self.image.get_width() / 2 and
                cursor_position[0] <= self.position[0] + self.image.get_width() / 2 and
                cursor_position[1] >= self.position[1] - self.image.get_height() / 2 and
                cursor_position[1] <= self.position[1] + self.image.get_height() / 2)