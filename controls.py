import pyautogui as pg
import pydirectinput as pd
import time
from mss import mss
import numpy as np
import cv2


def swing_it(cd):
    """
    spining attack with two-handed weapon
    """
    t=time.time()
    pd.click(button='right')
    time.sleep(0.2)
    for _ in range(6):
        pd.move(-77 * 8, 0, relative=True)
    for _ in range(6):
        pd.move(77 * 8, 0, relative=True)

    if time.time()-t<cd:
        time.sleep(cd-(time.time()-t))
    time.sleep(0.1)

def find_color(screenshot,w,h,h_left,h_right,v_left):
    """main driver for finding color"""
    screenshot_hsv = cv2.cvtColor(cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2HSV).reshape(
        (h, w, 3))
    color = np.logical_and(np.logical_and(np.logical_and(screenshot_hsv[:, :, 0] < h_right, screenshot_hsv[:, :, 0] > h_left),
                         screenshot_hsv[:, :, 2] > v_left / 100 * 255),screenshot_hsv[:, :, 1] > 60 / 100 * 255)
    return color


class interface:
    """
    class for controlling and character in RoG
    """
    holding=0
    def __init__(self,equipped):
        """
        :param equipped: {<keynumber>:{'cd':<cooldown>,'self':True or False},...}
        """
        self.equipped=equipped

    def use(self,number,swing=False):
        """
        main method for using smthinq
        :param number: item cell number
        :param swing: optional for using aoe spin attack on two-handed
        :return:
        """
        if number not in self.equipped:
            raise Exception('Key is not equipped')
        if number!=self.holding:
            pg.keyDown(str(number))
            time.sleep(0.05)
            pg.keyUp(str(number))
            self.holding=number

        if swing:
            swing_it(self.equipped[number]['cd'])
            return

        if self.equipped[number]['self']:
            pd.keyDown('q')
            time.sleep(0.05)
        pg.click()
        time.sleep(self.equipped[number]['cd'])
        if self.equipped[number]['self']:
            pd.keyUp('q')

    def healt_status(self):
        w = 451
        h = 3
        with mss() as sct:
            screenshot = sct.grab({'mon': 1, 'top': 15, 'left': 0, 'width': w, 'height': h})
            screenshot = np.array(screenshot)
            red = find_color(screenshot,w, h, -3, 3, 40)
            right = np.array(np.where(red == True))[1].max()
            return round(right / w, 2) * 100
    def mana_status(self):
        w = 451
        h = 3
        with mss() as sct:
            screenshot = sct.grab({'mon': 1, 'top': 31, 'left': 0, 'width': w, 'height': h})
            screenshot = np.array(screenshot)
            blue = find_color(screenshot, w, h, 97, 106, 25)
            right = np.array(np.where(blue == True))[1].max()
            return round(right / w, 2) * 100

    def energy_status(self):
        w = 424
        h = 3
        with mss() as sct:
            screenshot = sct.grab({'mon': 1, 'top': 46, 'left': 7, 'width': w, 'height': h})
            screenshot = np.array(screenshot)
            green = find_color(screenshot, w, h, 15, 22, 70)
            coords = np.where(green == True)
            if len(coords[1]) != 0:
                right = max(np.array(coords)[1].max(), 0)
            else:
                right = 0
            return round(right / w, 2) * 100

    def hunger_status(self):
        w = 423
        h = 3
        with mss() as sct:
            screenshot = sct.grab({'mon': 1, 'top': 57, 'left': 7, 'width': w, 'height': h})
            screenshot = np.array(screenshot)
            green = find_color(screenshot, w, h, 55, 66, 30)
            coords = np.where(green == True)
            if len(coords[1]) !=0:
                right=max(np.array(coords)[1].max(),0)
            else:
                right=0
            return round(right / w, 2) * 100



    def combat_status(self):
        """

        :return:
        True for combat
        False for non-combat
        """
        with mss() as sct:
            w = 56
            h = 143
            screenshot = sct.grab({'mon': 1, 'top': 937, 'left': 1780, 'width': w, 'height': h})
            screenshot = np.array(screenshot)
            cv2.imwrite('test_status.jpg',screenshot)
            violet=find_color(screenshot, w, h, 134, 150, 40)
            red = find_color(screenshot, w, h, -5, 5, 45)
            np.set_printoptions(threshold=np.inf)
            if True in red and True in violet:
                if np.array(np.where(red == True))[0].max()>np.array(np.where(violet == True))[0].max():
                    return True
                else:
                    return False
            if True in red: return True
            if True in violet: return False
            return False












