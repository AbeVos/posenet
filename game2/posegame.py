#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:43:01 2017

@author: abe
"""

import game_manager as game
import detector
from text import Line

class MainMenu(game.State):
    def __init__(self):
        super(MainMenu, self).__init__()
        
        self.title = Line((200, 200), "Titel", game.get_font('title'))
    
    def update(self, delta):
        self.title.set_position(detector.current_position())
        self.title.update(delta)
    
    def draw(self, surface):
        self.title.draw(surface)

def main():
    game.init()
    
    global_states = {
            'main_menu': MainMenu()}
    
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