import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import time
import keyboard
import pyautogui as pg
from mss import mss
import cv2
# Загрузка модели

size=42
def load_model(weights_path):
    # Предполагается, что вы знаете архитектуру вашей модели
    model = Sequential([
        layers.Conv2D(4, (3, 3), activation='selu', input_shape=(size, size, 3)),
        layers.MaxPool2D(1, 1),
        layers.Conv2D(16, (3, 3), activation='selu'),
        layers.MaxPool2D(1, 1),
        layers.Dropout(0.05),

        layers.Conv2D(32, (3, 3), activation='selu'),
        layers.MaxPool2D(1, 1),
        layers.Dropout(0.1),
        layers.Conv2D(64, (2, 2), activation='selu'),
        layers.MaxPool2D(1, 1),
        layers.Conv2D(128, (2, 2), activation='selu'),
        layers.MaxPool2D(1, 1),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(150, activation='selu'),

        layers.Dense(1, activation='sigmoid')
    ])

    # Загружаем веса
    model.load_weights(weights_path)
    return model


# Функция для подготовки изображения
def preprocess_image(screenshot, target_size):
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность для батча
    img_array /= 255.0  # Нормализация, если требуется
    return img_array


# Функция для классификации изображения
def classify_image(model, screenshot):
    # Подготовим изображение
    img_array = preprocess_image(screenshot,target_size=(42, 42))


    # Сделаем предсказание
    predictions = model.predict(img_array)
    print(predictions)
    if predictions[0]<0.01:
        return True
    else:
        return False
"""
# Параметры
image_height = 42  # Задайте высоту изображения
image_width = 42  # Задайте ширину изображения
num_channels = 3  # Например, 3 для RGB
num_classes = 2  # Укажите количество классов

# Пример использования
weights_path = 'best_model.keras'
model = load_model(weights_path)
img_path = 'no_curs.png'  # Укажите путь к вашему изображению
img_path1 = 'curs.png'

predicted_class = classify_image(model, img_path)

print(f'Предсказанный класс: {predicted_class}')
predicted_class = classify_image(model, img_path1)
print(f'Предсказанный класс: {predicted_class}')



if __name__ == '__main__':
    key = 'f'
    fin='g'
    regens=0
    count=0
    weights_path = 'best_model.keras'
    model = load_model(weights_path)
    while True:
        if keyboard.is_pressed(key):
            with mss() as sct:
                screenshot = sct.grab({'mon': 1, 'top': 519, 'left': 939, 'width': 42, 'height': 42})
            classify_image(model, screenshot)"""
