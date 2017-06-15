#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:43:01 2017

@author: abe
"""

import game_manager as game
import detect
from text import Line
from interface import Cursor, Button, CancelButton, HandScreen, PoseTutorial
from game_objects import BlockManager

import random

class MainMenu(game.State):
    def __init__(self):
        super(MainMenu, self).__init__()
        
        self.title = Line((game.screen_size[0] / 2,
                           game.screen_size[1] / 4),
                    "Houd de knop ingedrukt", game.get_font('title'))
        
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
            self.cursor.cursor_up()

class Tutorial(game.State):
    def __init__(self):
        super(Tutorial, self).__init__()
        self.title = Line((game.screen_size[0] / 2, 150), "Tutorial", game.get_font('title'))
        
        self.cursor = Cursor()
        self.hand_screen = HandScreen((3 * game.screen_size[0] / 4, game.screen_size[1] / 2))
        
        self.set_letter()
        
        game.global_state_changed.subscribe(self.global_state_changed)
    
    def update(self, delta):
        self.title.update(delta)
        self.cursor.update(delta)
        self.hand_screen.update(delta)
        self.pose_tutorial.update(delta)
        
        if not self.pose_tutorial.is_active:
            game.set_global_state('game')
    
    def draw(self, surface):
        self.title.draw(surface)
        self.hand_screen.draw(surface)
        self.pose_tutorial.draw(surface)
        self.cursor.draw(surface)
    
    def global_state_changed(self, previous_state, new_state):
        if new_state is 'tutorial':
            self.cursor.set_position((0,0))
            self.cursor.cursor_up()
            
            self.set_letter()
    
    def set_letter(self):
        self.letter = random.sample(game.get_property('available_letters'), 1)[0]
        
        self.title.set_text('De letter %s'%self.letter.upper())
        
        self.pose_tutorial = PoseTutorial((game.screen_size[0] / 4, game.screen_size[1] / 2),
                                          self.hand_screen.get_pose, self.letter)

class Game(game.State):
    def __init__(self):
        super(Game, self).__init__()
        
        self.cursor = Cursor()
        
        self.cursor = Cursor()
        self.return_button = CancelButton(
                (128, game.screen_size[1] - 128),
                 self.return_button_pressed)
        
        self.hand_screen = HandScreen((game.screen_size[0] - 256,
                                       256))
        
        self.block_manager = BlockManager((game.screen_size[0] / 2, 4 *  game.screen_size[1] / 7), self.hand_screen)
        
        game.global_state_changed.subscribe(self.global_state_changed)

    def update(self, delta):
        self.cursor.update(delta)
        self.return_button.update(delta)
        self.hand_screen.update(delta)
        
        self.block_manager.update(delta)
        
        if self.block_manager.is_finished:
            game.set_global_state('tutorial')
        
    def draw(self, surface):
        self.return_button.draw(surface)
        self.block_manager.draw(surface)
        self.hand_screen.draw(surface)
        self.cursor.draw(surface)
    
    def return_button_pressed(self):
        game.set_global_state('tutorial')
    
    def global_state_changed(self, previous_state, new_state):
        if new_state is 'game':
            self.block_manager.reset()

def main():
    game.init(1280, 960)
    
    available_letters = ['a','b','c','d','e']
    game.set_property('available_letters', available_letters)
    
    global_states = {
            'main_menu': MainMenu(),
            'tutorial': Tutorial(),
            'game': Game()
            }
    
    ## Set initial state
    game.set_global_states(global_states, 'main_menu')
    
    game.update.subscribe(update)
    game.draw.subscribe(draw)
     
    detect.init()
    
    game.start()
    
def update(delta):
    game.get_global_state().update(delta)

def draw(surface):
    game.get_global_state().draw(surface)
    
if __name__ == "__main__":
    main()