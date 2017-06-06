#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:43:01 2017

@author: abe
"""

import pygame as pg

import game_manager as game
import detector
from text import Line
from interface import Cursor, Button, CancelButton, HandScreen

class MainMenu(game.State):
    def __init__(self):
        super(MainMenu, self).__init__()
        
        self.title = Line((game.screen_size[0] / 2,
                           game.screen_size[1] / 4),
                    "Druk op de knop om te starten", game.get_font('title'))
        
        self.cursor = Cursor()
        self.start_button = Button((game.screen_size[0] / 2, game.screen_size[1] / 2), self.start_button_pressed)
        
        game.global_state_changed.subscribe(self.global_state_changed)
    
    def update(self, delta):
        self.title.update(delta)
        self.cursor.update(delta)
        self.start_button.update(delta)
    
    def draw(self, surface):
        self.title.draw(surface)
        self.start_button.draw(surface)
        self.cursor.draw(surface)
    
    def start_button_pressed(self):
        game.set_global_state('tutorial')
        
    def global_state_changed(self, previous_state, new_state):
        if new_state is 'main_menu':
            self.cursor.set_position((0,0))

class Tutorial(game.State):
    def __init__(self):
        super(Tutorial, self).__init__()
        self.title = Line((game.screen_size[0] / 2, 150), "Tutorial", game.get_font('title'))
        
        self.cursor = Cursor()
        self.return_button = CancelButton(
                (100, game.screen_size[1] - 100),
                 self.return_button_pressed)
        
        self.hand_screen = HandScreen((3 * game.screen_size[0] / 4, game.screen_size[1] / 2))
        
        game.global_state_changed.subscribe(self.global_state_changed)
    
    def update(self, delta):
        self.title.update(delta)
        self.cursor.update(delta)
        self.return_button.update(delta)
        self.hand_screen.update(delta)
    
    def draw(self, surface):
        self.title.draw(surface)
        self.return_button.draw(surface)
        self.hand_screen.draw(surface)
        self.cursor.draw(surface)
        
    def return_button_pressed(self):
        game.set_global_state('main_menu')
        
    def global_state_changed(self, previous_state, new_state):
        if new_state is 'main_menu':
            self.cursor.set_position((0,0))

def main():
    game.init(1280, 960)
    
    global_states = {
            'main_menu': MainMenu(),
            'tutorial': Tutorial()
            }
    
    game.set_global_states(global_states, 'main_menu')
    
    game.update.subscribe(update)
    game.draw.subscribe(draw)
    
    detector.init()
    
    game.start()
    
def update(delta):
    game.get_global_state().update(delta)

def draw(surface):
    game.get_global_state().draw(surface)
    
if __name__ == "__main__":
    main()