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
from icecream import ic
from controls import interface
from rout_routine import rout
from drawer import draw_rout
count=0

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
            num=True not in (coords[0]>pg.size()[1]+200)
            #cv2.imwrite(f'logs/screenshot{num}.png', screenshot)
            #cv2.imwrite(f'logs/thresh{num}.png', thresh)


            if rectangles[0][2]<60 and rectangles[0][3]<60:
                y+=120
                delta_y=120+rectangles[0][3]//2
                delta_x=rectangles[0][2]//2
            elif (True in (coords[0]<pg.size()[1]-50)) and (True not in (coords[0]>pg.size()[1]//2+200)):
                delta_y = rectangles[0][3]//4
                delta_x = rectangles[0][2]//4
            else:
               delta_y=5
               delta_x=5

            while (np.isnan(x) or np.isnan(y)) and first:
                x, y, delta_y = find_target(False)
                if count >= 5: return False

            x = int(x)
            #if x< 10: x=0
            y = int(y)

            return x, y,delta_x, delta_y



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

        if rect[2] < 60 and rect[3] < 60:
            y += 120
            delta_y = 120+rect[3]//2
            delta_x = rectangles[0][2] // 2
        elif (True in (coords[0] < pg.size()[1] - 50)) and (True not in (coords[0] > pg.size()[1] // 2 + 200)):
            delta_y = rect[3] // 4
            delta_x = rect[2] // 4
        else:
            delta_y = 5
            delta_x = 5


        x = int(x)
        #if x < 10: x = 0
        y = int(y)


        return x, y ,delta_x, delta_y

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


def move_to_target(cam,x,y,delta_x,delta_y):
    ic(delta_x,delta_y,x,y)
    if abs(x) > delta_x or abs(y) > delta_y:
        cam.end_walk()
    speed = 77  # pixel move for 15 degree turn
    x = int(m.atan(x / 800) * 180 / m.pi * speed / 15)
    y = int(m.atan(y / 800) * 180 / m.pi * speed / 15)
    cam.rotate(x, y)

    a = find_target()
    if not a: return True
    x=a[0]
    y=a[1]
    global count1
    count1=0
    while abs(x) > 5 or abs(y) > 5:
        ic(a[2], a[3], x, y)
        if count1>30:
            count1=0
            pd.keyDown('s')
            time.sleep(1)
            pd.keyUp('s')
        if abs(x) > a[2] or abs(y) > a[3]:
            cam.end_walk()
        speed = 77  # pixel move for 15 degree turn
        x = int(m.atan(x / 800) * 180 / m.pi * speed / 15)
        y = int(m.atan(y / 800) * 180 / m.pi * speed / 15)
        cam.rotate(x, y)
        a = find_target()
        if not a: return True
        x = a[0]
        y = a[1]

        with mss() as sct:
            screenshot = sct.grab({'mon': 1, 'top': 519, 'left': 939, 'width': 42, 'height': 42})
        if test.classify_image(model, screenshot):
            pd.keyDown('e')
            time.sleep(0.05)
            pd.keyUp('e')
            if test.classify_image(model, screenshot):
                pd.keyDown('e')
                time.sleep(0.05)
                pd.keyUp('e')

            count1=0
            look_around(cam)
            continue
        count1+=1
    return False


def look_around(cam):
    cam.betta_to_zero()
    for _ in range(4):
        if find_target():
            return True
        else:
            cam.rotate(77*6,0)
    return False






"""


if __name__ == '__main__':
    key = 'f'
    fin='g'
    regens=0
    cam = position()
    weights_path = 'best_model.keras'
    model = test.load_model(weights_path)
    while True:

        if keyboard.is_pressed(key):
            cam.start_walk()
            while True:
                a = find_target()
                if not a: continue
                move_to_target(cam, 0, a[1], a[2], a[3])
                cam.continue_walk()

"""


if __name__ == '__main__':
    key = 'f'
    fin='q'
    regens=0
    count1=0
    weights_path = 'best_model.keras'
    model = test.load_model(weights_path)
    x_walk=20
    y_walk=34
    delta_y=4
    direction=1
    nturns=y_walk/delta_y
    turns=0
    cam = position()
    inter=interface({2:{'cd':0,'self':True}})
    while True:
        if keyboard.is_pressed(key):
            points=draw_rout()
            routt=rout(points)
            routt.at_position()
            routt.to_current_point()

            turn = int((routt.angle_t - routt.angle) * 180 / np.pi * 77 // 15)
            pd.move(turn, 0, relative=True)


            while True:

                cam.start_walk()
                pd.keyDown('shift')
                cam.x2_timer_start()

                while True:
                    cam.x2_timer_end()
                    if cam.x2>routt.time_estimate:
                        if routt.at_position():
                            routt.go_to_next()
                            routt.at_position()
                            routt.to_current_point()
                        else:
                            routt.to_current_point()

                        pd.keyUp('shift')
                        cam.end_walk()

                        turn = int((routt.angle_t - routt.angle) * 180 / np.pi * 77 // 15)
                        pd.move(turn, 0, relative=True)

                        cam.checkpoint()

                        cam.start_walk()
                        pd.keyDown('shift')
                        cam.x2=0
                        cam.x2_timer_start()

                    else: cam.x2_timer_start()

                    if turns>nturns:
                        direction = direction * -1
                    if inter.hunger_status()<25:
                        pd.keyDown('i')
                        time.sleep(0.05)
                        pd.keyUp('i')
                        pd.moveTo(1841,549)
                        time.sleep(0.05)
                        pd.click()
                        time.sleep(0.1)
                        pd.click()
                        time.sleep(0.2)
                        pd.keyDown('i')
                        time.sleep(0.05)
                        pd.keyUp('i')

                    if inter.energy_status()<15:
                        cam.end_walk()
                        cam.x2_timer_end()
                        time.sleep(16)
                        cam.x2_timer_start()
                        cam.start_walk()



                    match find_target():
                        case False:
                            pass
                        case x, y,delta_x,delta_y:
                            break
                        case _:
                            pass

                cam.end_walk()
                pd.keyUp('shift')
                cam.x2_timer_end()
                cam.checkpoint()
                while True:
                    a = find_target()
                    if not a: break
                    if abs(a[0])<5:
                        move_to_target(cam, 0, a[1],a[2],a[3])
                        cam.continue_walk()
                    else:
                        cam.end_walk()
                        move_to_target(cam, a[0], a[1],a[2],a[3])
                        cam.continue_walk()

                    cam.continue_walk()

                routt.at_position()
                routt.to_current_point()
                cam.x2=0
                turn = int((routt.angle_t - routt.angle) * 180 / np.pi * 77 // 15)
                pd.move(turn, 0, relative=True)



"""

if __name__ == '__main__':
    key = 'f'
    fin='q'
    count=0
    while True:
        if keyboard.is_pressed(key):
            with mss() as sct:
                screenshot = sct.grab({'mon': 1, 'top': 519, 'left': 939, 'width': 42, 'height': 42})
                screenshot = np.array(screenshot)
                cv2.imwrite(f'dataset/curs/curs{0+count}.png',screenshot)
                time.sleep(1)
                count+=1
"""







