import pydirectinput as pd
import time
from mss import mss
import numpy as np
import cv2



def find_intersection(x, contour_points):
    intersections = []
    for i in range(len(contour_points)):
        x0, y0 = contour_points[i]
        x1, y1 = contour_points[i - 1]

        if (x0 <= x <= x1) or (x1 <= x <= x0):
            # Линейная интерполяция для нахождения y
            y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            intersections.append([int(x), int(y)])
    if len(intersections) != 2: print('GG')
    return intersections


class rout:
    def __init__(self,points,seed=True):
        self.points=np.array(points)
        self.current=0
        self.find_speed()
        if seed: self.seed()

    def take_screenshot(self):
        pd.keyDown('m')
        time.sleep(0.05)
        pd.keyUp('m')
        time.sleep(0.1)
        with mss() as sct:
            screenshot = sct.grab({'mon': 1, 'top': 0, 'left': 0, 'width': 1920, 'height': 1080})
            self.map = np.array(screenshot)

            pd.keyDown('m')
            time.sleep(0.05)
            pd.keyUp('m')

    def find_angle(self):
        """
        Функция для нахождения положения персонажа и угла его поворта
        :return: угол поворота (0 соответствует севреу), координаты на экране карты
        """
        self.take_screenshot()
        hsv = cv2.cvtColor(self.map.copy(), cv2.COLOR_BGR2HSV)

        # Определите диапазоны для hue
        lower_hue1 = np.array([0, 98, 62], dtype=np.uint8)  # от 0 до 1
        upper_hue1 = np.array([1, 241, 223], dtype=np.uint8)  # до 1

        lower_hue2 = np.array([179, 98, 62], dtype=np.uint8)  # от 179 до 180
        upper_hue2 = np.array([181, 241, 223], dtype=np.uint8)  # до 180
        # Создайте маски для hue
        mask1 = cv2.inRange(hsv, lower_hue1, upper_hue1).astype(bool)
        mask2 = cv2.inRange(hsv, lower_hue2, upper_hue2).astype(bool)
        # Объедините обе маски
        # combined_mask = cv2.bitwise_or(mask1, mask2)
        thresh = np.zeros((1080, 1920), dtype=np.uint8)
        thresh[np.logical_or(mask1, mask2)] = 255
        contours0, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area = 0

        for cnt in contours0:
            rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
            box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
            box = np.int0(box)  # округление координат
            cop = self.map.copy()
            if cv2.contourArea(box) > area:
                bbox = box
                cont = cnt
                area = cv2.contourArea(box)

        cont_chng = cv2.approxPolyDP(cont, 6, True)

        #cv2.drawContours(self.map, cont_chng, -1, (0, 255, 0, 255), 1)

        side_sum = np.zeros(len(cont_chng))
        for i in range(len(cont_chng)):
            side = cont_chng[i, 0] - cont_chng[i - 1, 0]
            l = side[0] * side[0] + side[1] * side[1]
            side_sum[i] += l
            side_sum[i - 1] += l

        moments = cv2.moments(thresh, 1)
        dM01 = moments['m01']
        dM10 = moments['m10']
        dArea = moments['m00']
        # будем реагировать только на те моменты,
        # которые содержать больше 100 пикселей
        if dArea > 50:
            x = int(dM10 / dArea)
            y = int(dM01 / dArea)

        tip = cont_chng[side_sum.argmax(), 0]
        base = [x, y]
        arrow = tip - base
        angle = np.arctan2(arrow[0], arrow[1])
        if arrow[0] >= 0:
            if angle >= 0:
                angle = np.pi / 2 - angle

            else:
                angle = np.pi / 2 + angle

        if arrow[0] < 0:
            if angle >= 0:
                angle = 3 * np.pi / 2 + angle

            else:
                angle = np.pi / 2 - angle
        self.x=x
        self.y=y
        self.angle=angle + np.pi / 2

    def at_position(self):
        self.find_angle()
        if ((self.x-self.points[self.current][0])**2+(self.y-self.points[self.current][1])**2)**0.5 < 15: return True
        else: return False

    def go_to_next(self):
        if self.current+1< len(self.points):self.current+=1
        else: self.current=0

    def find_speed(self):
        """нахождение скорости персонажа
           сравнивает положение метки на карте до и после бега в течении walk_time
        """
        walk_time=5
        self.find_angle()
        x_old=self.x
        y_old=self.y

        pd.keyDown('w')
        pd.keyDown('shift')
        time.sleep(walk_time)
        pd.keyUp('w')
        pd.keyUp('shift')

        self.find_angle()

        self.speed = (((self.x - x_old) ** 2 + (self.y - y_old) ** 2) ** 0.5) / 5

    def seed(self):
        """ разбивает область маршрута на вертикальные отрезки"""
        cont = np.array(self.points)
        width=self.speed*2
        left_arg = cont[:, 0].argmin()
        x_left = cont[left_arg, 0]
        x_right = cont[:, 0].max()
        tmp = int((x_right - x_left) // width)
        count = 0
        all_intersections = []
        for i in range(1, tmp):
            intersections = find_intersection(x_left + int(i * width), cont)
            all_intersections.extend(intersections)
            count += 1
        all_intersections = np.array(all_intersections)
        all_intersections = all_intersections[np.argsort(all_intersections[:, 0])]
        final_cont = [cont[left_arg]]
        for i in range(len(all_intersections) // 2):
            pt1 = all_intersections[2 * i]
            pt2 = all_intersections[2 * i + 1]

            if abs(pt1[1] - final_cont[-1][1]) <= abs(pt2[1] - final_cont[-1][1]):
                final_cont.append(pt1)
                y = pt2[1]
            else:
                final_cont.append(pt2)
                y = pt1[1]

            if i < len(all_intersections) // 2 - 1:
                next = np.array(all_intersections[2 * i + 2:2 * i + 4, 1])
                y = next[abs(next - final_cont[-1][1]).argmax()]

            final_cont.append(np.array([pt1[0], y]))

        cv2.drawContours(self.map, [self.points.astype(np.int32)], -1, (255, 0, 0, 255), 1)
        self.points=np.array(final_cont)
        cv2.drawContours(self.map, [self.points.astype(np.int32)], -1, (0, 0, 255, 255), 1)
        cv2.imshow('Route checl',self.map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        time.sleep(5)

    def to_current_point(self):
        """функция для нахождения направления движения на текущую точку и рассчётного времени движения до неё"""
        base = [self.x, self.y]
        tip = self.points[self.current]
        arrow = tip - base
        self.angle_t = np.arctan2(arrow[0], arrow[1])
        if arrow[0] >= 0:
            if self.angle_t >= 0:
                self.angle_t = np.pi / 2 - self.angle_t
            else:
                self.angle_t = np.pi / 2 + self.angle_t
        if arrow[0] < 0:
            if self.angle_t >= 0:
                self.angle_t = 3 * np.pi / 2 + self.angle_t
            else:
                self.angle_t = np.pi / 2 - self.angle_t
        self.angle_t += np.pi / 2
        self.time_estimate = ((self.points[self.current][0]-self.x)**2+(self.points[self.current][1]-self.y)**2)**0.5/self.speed


