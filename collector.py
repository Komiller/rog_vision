import keyboard
import time
import colorsys
from mss import mss
import numpy as np
import pyautogui as pg
import pydirectinput as pd
from mss import mss
import cv2
import math as m
import test
from mover import position
from profilehooks import profile
from icecream import ic

#@profile(stdout=False, filename='baseline.prof')
def alternative(arr, i=0):
    l = len(arr)
    if i == l: return arr
    arr1 = arr.copy()
    count = 0
    rect1 = arr[i]
    for j in range(1, len(arr) - i):
        rect2 = arr[i + j]
        if abs((rect1[0] + rect1[2] // 2) - (rect2[0] + rect2[2] // 2)) < (rect2[2] + rect1[2])*0.4 and abs(
                (rect1[1] + rect1[3] // 2) - (rect2[1] + rect2[3] // 2)) < (rect2[3] + rect1[3])*0.4:
            new_counter = np.array(
                [[rect2[0], rect2[1]], [rect2[0], rect2[1] + rect2[3]], [rect2[0] + rect2[2], rect2[1]],
                 [rect2[0] + rect2[2], rect2[1] + rect2[3]],
                 [rect1[0], rect1[1]], [rect1[0], rect1[1] + rect1[3]], [rect1[0] + rect1[2], rect1[1]],
                 [rect1[0] + rect1[2], rect1[1] + rect1[3]]], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(new_counter)
            rect1 = [x, y, w, h]
    for j in range(1, len(arr) - i):
        rect2 = arr[i + j]
        if abs((rect1[0] + rect1[2] // 2) - (rect2[0] + rect2[2] // 2)) < (rect2[2] + rect1[2]) * 0.6 and abs(
                (rect1[1] + rect1[3] // 2) - (rect2[1] + rect2[3] // 2)) < (rect2[3] + rect1[3]) * 0.6:
            arr1.pop(i + j - count)
            count += 1
    arr1[i] = rect1
    return alternative(arr1, i=i + 1)


def find_target(first=True):
    with mss() as sct:
        #делаем скриншот и фильтруем пискели подсветки лута
        screenshot = sct.grab({'mon': 1, 'top': 0, 'left': 0, 'width': 1920, 'height': 1080})
        hsv_min = np.array((85, 0 / 100 * 255, 95 / 100 * 255), np.uint8)
        hsv_max = np.array((98, 17 / 100 * 255, 100 / 100 * 255), np.uint8)
        screenshot = np.array(screenshot)
        hsv = cv2.cvtColor(cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, hsv_min, hsv_max)
        #находим области в которых есть пиксели обводки лута
        contours0, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        count=0
        for contour in contours0:
            if len(contour) < 4:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w*h<10*10:continue

            flag = False
            if len(rectangles) == 0:
                rectangles.append([x, y, w, h])

            for i, re in enumerate(rectangles):
                if abs((re[0]+re[2]//2)-(x+w//2))<(w+re[2])*0.7 and abs((re[1]+re[3]//2)-(y+h//2))<(h+re[3])*0.7:
                    new_counter=np.array([[x,y],[x+w,y],[x,y+h],[x+h,y+w],[re[0],re[1]],[re[0]+re[2],re[1]],[re[0],re[1]+re[3]],[re[0]+re[2],re[1]+re[3]]],dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(new_counter)
                    rectangles[i] = [x, y, w, h]

                    flag = True
            if not flag:
                rectangles.append([x, y, w, h])







        rectangles = alternative(rectangles)
        rectangles = np.array(rectangles)

        def choose(rectangles):
            """
            :param rectangles: прямоугольники
            :return: прямоугольник, удоволетворяющий условиям (нижний из близких к центру, или самый близкий к курсору
            """
            x_close=abs(rectangles[:,0]+rectangles[:,2]//2-pg.size()[0]//2) < 77*2 # прямоугольники с центром в +-30 градусов от центра экрана
            if True in x_close:
                rect= rectangles[x_close]
                arg=(rectangles[x_close,1]+rectangles[x_close,3]).argmax()
                return rect[arg]
            l=(np.power(rectangles[:,0]+rectangles[:,2]//2-pg.size()[0],2)+np.power(rectangles[:,0]+rectangles[:,2]//2-pg.size()[0],2)).argmin()
            return rectangles[l]

        def filter(rect):
            """
            удаляет пиксели далёкие от прямоугольника
            """
            thresh[0:max(rect[1] - rect[3], 0), :] = 0
            thresh[-1:max(rect[1] + 2 * rect[3], 1080):-1, :] = 0
            thresh[:, 0:max(rect[0] - rect[2], 0)] = 0
            thresh[:, -1:max(rect[0] + 2 * rect[2], 1920):-1] = 0


        if len(rectangles) == 0:
            return False
        """
        for rect in rectangles:
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
            cv2.rectangle(screenshot, [x, y], [x + w, y + h], [255, 0, 0, 255], 1)"""

        if len(rectangles) == 1:
            filter(rectangles[0])
            coords = np.array(np.where(thresh.astype(bool) == True))
            x = np.average(coords[1]) - pg.size()[0] // 2
            y = np.average(coords[0]) - pg.size()[1] // 2
            #cv2.imwrite('logs/screenshot.png', screenshot)
            #cv2.imwrite('logs/thresh.png', thresh)



            if True in (coords[0]>750):
                delta_h = 5
            else:
                delta_h = rectangles[0][3]

            while (np.isnan(x) or np.isnan(y)) and first:
                x, y, delta_h = find_target(False)
                if count >= 5: return False

            x = int(x)
            y = int(y)

            return x, y, delta_h



        rect=choose(rectangles)

        for rectangle in rectangles:
            if np.array_equal(rect,rectangle): continue
            else:thresh[rectangle[1]:rectangle[1]+rectangle[3],rectangle[0]:rectangle[0]+rectangle[2]]=0

        filter(rect)

        coords = np.array(np.where(thresh.astype(bool) == True))
        x = np.average(coords[1]) - pg.size()[0] // 2
        y= np.average(coords[0]) - pg.size()[1] // 2
        #cv2.imwrite('logs/screenshot.png', screenshot)
        #cv2.imwrite('logs/thresh.png', thresh)
        count=0
        if True in (coords[0]>750):
            delta_h = 5
        else:
            delta_h = rect[3]

        while (np.isnan(x) or np.isnan(y)) and first:
            x,y,delta_h=find_target(False)
            if count>=5: return False


        x = int(x)
        y = int(y)


        return x, y , delta_h

def ready():
    with mss() as sct:
        screenshot = sct.grab({'mon': 1, 'top': 519, 'left': 939, 'width': 42, 'height': 42})
        screenshot = np.array(screenshot)
        cv2.imwrite('screenshot1.png', screenshot)
        for x in range(42):
            for y in range(42):
                if ((x-21)**2+(y-21)**2)**0.5<=18 or ((x-21)**2+(y-21)**2)**0.5>=20: screenshot[x,y]=[0,0,0,255]
        hsv = cv2.cvtColor(cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2HSV)
        hsv_min = np.array((0, 0 / 100 * 255, 80 / 100 * 255), np.uint8)
        hsv_max = np.array((255, 1 / 100 * 255, 100 / 100 * 255), np.uint8)
        thresh = cv2.inRange(hsv, hsv_min, hsv_max).astype(bool)


        coords=np.where(thresh==True)

        if len(coords[1])>35 and np.average(coords[1]-21)+np.average(coords[0]-21)<2:return True
        return False

def move_to_target(cam,x,y):
    length = 10
    speed = 77  # pixel move for 15 degree turn
    x = int(m.atan(x / 800) * 180 / m.pi * speed / 15)
    y = int(m.atan(y / 800) * 180 / m.pi * speed / 15)
    cam.rotate(x, y)
    a = find_target()
    if not a: return True
    x=a[0]
    y=a[1]
    while abs(x) > 5 or abs(y) > 5:
        speed = 77  # pixel move for 15 degree turn
        x = int(m.atan(x / 800) * 180 / m.pi * speed / 15)
        y = int(m.atan(y / 800) * 180 / m.pi * speed / 15)
        cam.rotate(x, y)
        a = find_target()
        if not a: return True
        x = a[0]
        y = a[1]
        if abs(x) < length or abs(y) < length:
            length = length - 1
        if length == 2: break
        with mss() as sct:
            screenshot = sct.grab({'mon': 1, 'top': 519, 'left': 939, 'width': 42, 'height': 42})
        if test.classify_image(model, screenshot):
            pd.keyDown('e')
            time.sleep(0.05)
            pd.keyUp('e')
            continue
    return False











"""
if __name__ == '__main__':
    key = 'f'
    fin='g'
    regens=0
    while True:
        if keyboard.is_pressed(key):
            find_target()

"""
if __name__ == '__main__':
    key = 'f'
    fin='q'
    regens=0
    count=0
    weights_path = 'best_model.keras'
    model = test.load_model(weights_path)
    x=5
    y=20
    delta_y=5
    direction=1
    nturns=y/delta_y
    turns=0
    cam = position()
    while True:
        if keyboard.is_pressed(key):
            while True:

                cam.start_walk()
                cam.x2_timer_start()

                while True:
                    cam.x2_timer_end()
                    if cam.x2>50:
                        print('123243143')
                        cam.end_walk()

                        cam.rotate(direction*77*6,0)
                        cam.start_walk()
                        time.sleep(delta_y)
                        cam.end_walk()

                        cam.rotate(direction * 77 * 6,0)
                        cam.start_walk()
                        time.sleep(cam.x2-50)
                        cam.end_walk()

                        direction=direction*-1
                        cam.checkpoint()
                        cam.start_walk()
                        cam.x2=0
                        cam.x2_timer_start()

                        turns+=1
                    else: cam.x2_timer_start()

                    if turns>nturns:
                        direction = direction * -1


                    match find_target():
                        case False:
                            pass
                        case x, y,d:
                            break
                        case _:
                            pass

                cam.end_walk()
                cam.x2_timer_end()
                cam.checkpoint()
                while True:
                    if keyboard.is_pressed(key):break
                    a = find_target()
                    if not a: break
                    if abs(a[0])>5 or abs(a[1])>a[2]:
                        cam.end_walk()
                        if move_to_target(cam,a[0],a[1]):
                            continue
                        with mss() as sct:
                            screenshot = sct.grab({'mon': 1, 'top': 519, 'left': 939, 'width': 42, 'height': 42})
                        if test.classify_image(model, screenshot):
                            pd.keyDown('e')
                            time.sleep(0.05)
                            pd.keyUp('e')
                            cam.betta_to_zero()
                            continue
                        cam.start_walk()

                    with mss() as sct:
                        screenshot = sct.grab({'mon': 1, 'top': 519, 'left': 939, 'width': 42, 'height': 42})
                    if test.classify_image(model, screenshot):
                        pd.keyDown('e')
                        time.sleep(0.05)
                        pd.keyUp('e')
                        cam.end_walk()
                        cam.betta_to_zero()
                        continue

                    cam.continue_walk()
                    #wprint(time.time()-t)


                if cam.x<10:
                    cam.return_back(x_return=False)
                    print(cam.x2)
                    cam.x2+=cam.x
                    print(cam.x2)
                else: cam.return_back()







