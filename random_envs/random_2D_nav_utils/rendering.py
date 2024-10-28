import pygame
import threading
import numpy as np
import time, os
from .geometry import Box
from .dynamics import AbstractRandom2DNavigationDynamics

class Random2DNavRenderer():
    def game_loop(self):
        pygame.init()
        pygame.display.init()
        pygame.font.init()
        self.complete_init.set()
        while not self.quit.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass # TODO
            time.sleep(0.1)
        pygame.display.quit()
        pygame.quit()
    
    def __init__(self):
        self.quit = threading.Event()
        self.complete_init = threading.Event()
        thread = threading.Thread(target=self.game_loop)
        thread.daemon = True  # Set the thread to daemon mode so it exits when the main thread exits
        self.pygame_thread = thread
        thread.start()
        self.complete_init.wait()
        self.game_display = None
        self.game_font = None
        self.clock = None
    
    def __del__(self):
        self.close()
    
    def close(self):
        self.quit.set()
        self.pygame_thread.join()
    
    def render(self, bounding_box: Box, dynamics: AbstractRandom2DNavigationDynamics):
        """Render the scene"""
        L = 800
        P = 100  # padding
        W = (255, 255, 255)
        G = (0, 255, 0)
        B = (0, 0, 0)
        BL = (0, 0, 255)
        SCALE = (L - P) / 1.2

        box_pos = dynamics.box_pos
        goal = dynamics.goal
        wind = dynamics.wind

        def t(x, y):
            return L / 2 + x * SCALE, (L - P) - y * SCALE + P / 2

        if self.game_display is None:
            self.game_display = pygame.display.set_mode((L, L))
        if self.game_font is None:
            self.game_font = pygame.font.SysFont('Arial', 10)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.game_display.fill(W)
        for r in bounding_box.rectangles:
            left, top = t(r.values[0], r.values[3])
            left, top = int(left), int(top)
            width = int((r.values[1] - r.values[0]) * SCALE)
            height = int((r.values[3] - r.values[2]) * SCALE)
            pygame.draw.rect(self.game_display, B, (left, top, width, height))
        left, top = t(*box_pos)
        left, top = int(left), int(top)
        pygame.draw.circle(self.game_display, BL, (left, top), 20)
        left, top = t(*goal)
        left, top = int(left), int(top)
        pygame.draw.circle(self.game_display, G, (left, top), 20)
        text_surface = self.game_font.render("wind: " + str(wind), False, (0, 0, 0))
        self.game_display.blit(text_surface, (0,0))
        pygame.display.update()
        self.clock.tick(24)