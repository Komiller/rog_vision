import time
import math as m
import numpy as np
import pydirectinput as pd
from line_profiler import profile

class position():
    timer=0
    speed=77 #кол-во пикселей для поворота на 15 градусов

    def __init__(self):
        """
        при инициаилизации текущие положение персонажа и камеры принимается за 0
        """
        self.x=0
        self.y=0
        self.alpha=0
        self.betta=0
        self.walk=False
        self.x2=0
        self.x2_time = 0
        self.last_call=0

    def checkpoint(self):
        self.x = 0
        self.y = 0
        self.alpha = 0
        self.betta = 0
        self.timer = 0

    #@profile
    def rotate(self,x,y):
        self.alpha+=x
        if self.betta>77*7 and y>0 : y=y*2
        self.betta+=y
        pd.move(x, y, relative=True)
        """
        while abs(y) > 80:
            pd.move(0, np.sign(y) * 80, relative=True)
            y = y - np.sign(y) * 80
        else:
            pd.move(0, y, relative=True)


        while abs(x) > 80:
            pd.move(np.sign(x) * 80, 0, relative=True)
            x = x - np.sign(x) * 80
        else:
            pd.move(x, 0, relative=True)"""

    def start_walk(self):
        self.last_call=time.time()
        pd.keyDown('w')
        self.timer = time.time()
        self.walk=True

    def continue_walk(self):
        if self.walk:pass
        else:
            pd.keyDown('w')
            self.timer = time.time()
            self.walk = True

    def end_walk(self):
        if self.walk:
            self.timer = time.time() - self.timer
            pd.keyUp('w')
            self.x += m.cos(15 * self.alpha / self.speed * m.pi / 180) * self.timer
            self.y += m.sin(15 * self.alpha / self.speed * m.pi / 180) * self.timer
            self.walk=False
        else: pass


    def return_back(self, x_return=True):
        if self.y != 0:
            if self.y > 0: self.rotate(-self.alpha - self.speed * 6, 0)
            if self.y < 0: self.rotate(-self.alpha + self.speed * 6, 0)
            pd.keyDown('w')
            time.sleep(abs(self.y))
            pd.keyUp('w')
            self.y=0
        if self.x != 0 and x_return:
            if self.x > 0: self.rotate(-self.alpha - self.speed * 12, 0)
            if self.x < 0: self.rotate(-self.alpha, 0)
            pd.keyDown('w')
            time.sleep(abs(self.x))
            pd.keyUp('w')
            self.x=0
        self.rotate(-self.alpha, 0)
        pd.move(0, 1080, relative=True)
        pd.move(0, 77 * 5, relative=True)

    def betta_to_zero(self):
        pd.move(0,1080, relative=True)
        pd.move(0, 77 * 5, relative=True)
        self.betta=77*5

    def x2_timer_start(self):
        self.x2_time=time.time()

    def x2_timer_end(self):
        self.x2+=time.time()-self.x2_time
        self.x2_time=0








