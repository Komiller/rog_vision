import tkinter as tk

import PIL.Image
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
import math as m
from mss import mss
import keyboard
from icecream import ic
import pydirectinput as pd
import time
from rout_routine import rout
class ImageZoomApp:
    def __init__(self, root,image_arr):
        self.root = root
        self.root.title("Image Zoom")

        # Create a Canvas widget
        self.canvas = tk.Canvas(self.root, width=1920, height=1080, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize image and image reference
        self.image = None
        self.tk_image = None

        # Load an initial image (you can use your own image file)
        self.load_image(image_arr)
        self.draw_obj = ImageDraw.Draw(self.image)

        # with Windows OS
        self.canvas.bind("<MouseWheel>", self.zoomer)
        # with Linux OS
        root.bind("<Button-4>", self.zoomer)
        root.bind("<Button-5>", self.zoomer)

        self.canvas.bind("<ButtonPress-2>", self.start_drag)  # Для начала перетаскивания
        self.canvas.bind("<B2-Motion>", self.drag)  # Для перетаскивания
        self.canvas.bind("<ButtonRelease-2>", self.end_drag)  # Для окончания перетаскивания

        self.canvas.bind("<ButtonPress-1>", self.start_draw)  # Начало рисования
        self.canvas.bind("<Motion>", self.draw)  # Рисование линии
        self.canvas.bind("<Double-Button-1>", self.end_draw)  # Окончание рисования
        self.canvas.bind("<ButtonPress-3>", self.put_down)

        self.scale = 1  # scale factor
        self.offset_x = 0
        self.offset_y = 0
        self.offset_x_real=0
        self.offset_y_real = 0

        self.drawing = False
        self.start_point = (0,0)
        self.line=False
        self.points=[]  # Список для хранения координат угловых точек
        self.canvas.create_oval([(1920//2-2,1080//2-2),(1920//2+2,1080//2+2)],fill='red',tags='first')

    def load_image(self, filename):
        # Load the image using PIL (Python Imaging Library)
        self.image = Image.fromarray(filename)
        self.tk_image = ImageTk.PhotoImage(self.image)

        # Display the image on the Canvas
        self.img_obj=self.canvas.create_image((0, 0), anchor=tk.NW,image=self.tk_image)



    def start_draw(self, event):
        if not self.drawing:
            self.drawing = True
            self.start_point = (event.x, event.y)  # Сохраняем координаты в списке
            start = (
            (self.start_point[0] - self.offset_x_real - (1920 - self.image.width * self.scale) // 2) // self.scale,
            (self.start_point[1] - self.offset_y_real - (1080 - self.image.height * self.scale) // 2) // self.scale)
            self.points.append(start)


    def draw(self, event):
        if self.drawing:
            if self.line:
                self.canvas.coords(self.line, [self.start_point, (event.x, event.y)])
            else:
                self.line = self.canvas.create_line(self.start_point, event.x, event.y, fill='red', width=2)


    def end_draw(self,event):
        start=((self.start_point[0]-self.offset_x_real-(1920-self.image.width * self.scale)//2)//self.scale,
               (self.start_point[1]-self.offset_y_real-(1080-self.image.height * self.scale)//2)//self.scale)
        end=((self.points[0][0]-self.offset_x_real-(1920-self.image.width * self.scale)//2)//self.scale,
             (self.points[0][1]-self.offset_y_real-(1080-self.image.height * self.scale)//2)//self.scale)
        self.draw_obj.line([end,start], fill='red')
        self.canvas.delete(self.img_obj)
        self.tk_image = ImageTk.PhotoImage(
            self.image.resize((int(self.image.width * self.scale), int(self.image.height * self.scale)),
                              Image.LANCZOS))
        self.img_obj=self.canvas.create_image(1920//2+self.offset_x_real,1080//2+self.offset_y_real,image=self.tk_image)
        self.canvas.create_oval([(1920 // 2 - 2, 1080 // 2 - 2), (1920 // 2 + 2, 1080 // 2 + 2)], fill='red',
                                tags='first')
        self.canvas.delete(self.line)
        self.line=False
        self.drawing=False

    def put_down(self,event):
        if self.drawing:
            start=((self.start_point[0]-self.offset_x_real-(1920-self.image.width * self.scale)//2)//self.scale,
                   (self.start_point[1]-self.offset_y_real-(1080-self.image.height * self.scale)//2)//self.scale)
            end=((event.x-self.offset_x_real-(1920-self.image.width * self.scale)//2)//self.scale,
                 (event.y-self.offset_y_real-(1080-self.image.height * self.scale)//2)//self.scale)
            self.draw_obj.line([start,end], fill='red')
            self.canvas.delete(self.img_obj)
            self.tk_image = ImageTk.PhotoImage(
                self.image.resize((int(self.image.width * self.scale), int(self.image.height * self.scale)),
                                  Image.LANCZOS))
            self.img_obj=self.canvas.create_image(1920//2+self.offset_x_real,1080//2+self.offset_y_real,image=self.tk_image)
            self.canvas.create_oval([(1920 // 2 - 2, 1080 // 2 - 2), (1920 // 2 + 2, 1080 // 2 + 2)], fill='red',
                                    tags='first')
            self.canvas.delete(self.line)
            self.points.append(end)
            self.line=False
            self.start_point = (event.x, event.y)


    def start_drag(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def drag(self, event):
        self.update_image(event.x - self.start_x, event.y - self.start_y)
        self.start_x = event.x
        self.start_y = event.y


    def end_drag(self, event):
        pass

    def update_image(self,x,y):
        self.offset_x_real = self.offset_x * self.scale+x
        self.offset_y_real = self.offset_y * self.scale+y
        self.offset_x+=x/ self.scale
        self.offset_y += y/ self.scale
        self.canvas.coords(self.img_obj,[1920//2+self.offset_x_real,1080//2+self.offset_y_real])



    def zoomer(self, event=None):
        if not (event is None):
            if event.num == 4 or event.delta == 120:
                self.scale = self.scale * 1.2
                self.canvas.delete(self.img_obj)
                self.tk_image = ImageTk.PhotoImage(
                    self.image.resize((int(self.image.width * self.scale), int(self.image.height * self.scale)),
                                      Image.LANCZOS))
                self.offset_x_real = self.offset_x * self.scale
                self.offset_y_real = self.offset_y * self.scale
                self.img_obj=self.canvas.create_image(1920//2+self.offset_x_real,1080//2+self.offset_y_real,image=self.tk_image)
                self.canvas.create_oval([(1920 // 2 - 2, 1080 // 2 - 2), (1920 // 2 + 2, 1080 // 2 + 2)], fill='red',
                                        tags='first')
            if event.num == 5 or event.delta == -120:
                self.scale = self.scale * 0.8
                self.canvas.delete(self.img_obj)
                self.tk_image = ImageTk.PhotoImage(
                    self.image.resize((int(self.image.width * self.scale), int(self.image.height * self.scale)),
                                      Image.LANCZOS))
                self.offset_x_real=self.offset_x*self.scale
                self.offset_y_real = self.offset_y * self.scale
                self.img_obj=self.canvas.create_image(1920//2+self.offset_x_real,1080//2+self.offset_y_real,image=self.tk_image)
                self.canvas.create_oval([(1920 // 2 - 2, 1080 // 2 - 2), (1920 // 2 + 2, 1080 // 2 + 2)], fill='red',
                                        tags='first')





def draw_rout():
    pd.keyDown('m')
    time.sleep(0.05)
    pd.keyUp('m')
    time.sleep(0.1)
    with mss() as sct:
        screenshot = sct.grab({'mon': 1, 'top': 0, 'left': 0, 'width': 1920, 'height': 1080})
        img = np.array(screenshot)
        root = tk.Tk()
        app = ImageZoomApp(root, img)
        root.mainloop()
        time.sleep(5)
        pd.keyDown('m')
        time.sleep(0.05)
        pd.keyUp('m')
    return app.points














