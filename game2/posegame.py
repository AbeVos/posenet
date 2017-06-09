#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:43:01 2017

@author: abe
"""

import game_manager as game
import detect
from text import Line
from actor import AnimatedActor
from interface import Cursor, Button, CancelButton, HandScreen

class MainMenu(game.State):
    def __init__(self):
        super(MainMenu, self).__init__()
        
        self.title = Line((game.screen_size[0] / 2,
                           game.screen_size[1] / 4),
                    "Houd de knop ingedrukt", game.get_font('title'))
        
        self.cursor = Cursor()
        self.start_button = Button((game.screen_size[0] / 2, game.screen_size[1] / 2), self.start_button_pressed)
        
        self.pose1 = AnimatedActor((300, 600))
        self.pose1.set_animation(game.get_animation('pose_a'), (256, 256), 15)
        
        self.pose2 = AnimatedActor((500, 600), delay=0.5)
        self.pose2.set_animation(game.get_animation('pose_b'), (256, 256), 15)
        
        self.pose3 = AnimatedActor((700, 600), mode='pingpong')
        self.pose3.set_animation(game.get_animation('pose_c'), (256, 256), 15)
        
        self.pose4 = AnimatedActor((900, 600), mode='pingpong', delay = 0.5)
        self.pose4.set_animation(game.get_animation('pose_d'), (256, 256), 15)
        
        self.pose5 = AnimatedActor((1100, 600), mode='pingpong')
        self.pose5.set_animation(game.get_animation('pose_e'), (256, 256), 15)
        
        game.global_state_changed.subscribe(self.global_state_changed)
    
    def update(self, delta):
        self.title.update(delta)
        self.cursor.update(delta)
        self.start_button.update(delta)
        
        self.pose1.update(delta)
        self.pose2.update(delta)
        self.pose3.update(delta)
        self.pose4.update(delta)
        self.pose5.update(delta)
    
    def draw(self, surface):
        self.title.draw(surface)
        self.start_button.draw(surface)
        self.cursor.draw(surface)
        
        self.pose1.draw(surface)
        self.pose2.draw(surface)
        self.pose3.draw(surface)
        self.pose4.draw(surface)
        self.pose5.draw(surface)
    
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
        if new_state is 'tutorial':
            self.cursor.set_position((0,0))
            self.cursor.cursor_up()

def main():
    game.init(1280, 960)
    
    global_states = {
            'main_menu': MainMenu(),
            'tutorial': Tutorial()
            }
    
    game.set_global_states(global_states, 'tutorial')
    
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