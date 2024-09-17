import tensorflow as tf
from tensorflow import keras

# импортируем библиотеку pathlib, а также функцию Path для работы с директориями
import pathlib
from pathlib import Path
import os
# для упорядочивания файлов в директории
import natsort

# библиотеки для работы с изображениями
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# -- Импорт для подготовки данных: --
# модуль для предварительной обработки изображений
from tensorflow.keras.preprocessing import image
# Класс ImageDataGenerator - для генерации новых изображений на основе имеющихся
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -- Импорт для построения модели: --
# импорт слоев нейросети
from tensorflow.keras import layers
#from tensorflow.keras.layers.experimental import preprocessing
# импорт модели
from tensorflow.keras.models import Sequential
# импорт оптимайзера
from tensorflow.keras.optimizers import RMSprop
import splitfolders

woman_path='no_curs'
men_path='curs'
splitfolders.ratio('dataset', 'faces_splited', ratio=(0.8, 0.15, 0.05), seed=18, group_prefix=None )

# определим параметры нормализации данных
train = ImageDataGenerator(rescale=1 / 255)
val = ImageDataGenerator(rescale=1 / 255)

size=42
# сгенерируем нормализованные данные
train_data = train.flow_from_directory('faces_splited/train', target_size=(size, size),
                                       class_mode='binary', batch_size=3, shuffle=True)
val_data = val.flow_from_directory('faces_splited/val', target_size=(size, size),
                                   class_mode='binary', batch_size=3, shuffle=True)

# Определяем параметры аугментации
data_augmentation = keras.Sequential(
  [
    # Отражение по горизонтали
    layers.RandomFlip("horizontal", input_shape=(size, size,3)),
    # Вращение на рандомное значение до 0.05
    layers.RandomRotation(0.05),
    # Меняем контрастность изображений
    layers.RandomContrast(0.23),
    # Изменяем размер
    layers.RandomZoom(0.2)
  ]
)

model = Sequential([
    # добавим аугментацию данных
    data_augmentation,
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

# Файл для сохранения модели с лучшими параметрами
checkpoint_filepath = 'best_model.keras'
# Компиляция модели
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.00024),
              # optimizer=tf.keras.optimizers.Adam(learning_rate=0.000244),
              metrics=['binary_accuracy'])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_binary_accuracy',
    mode='max',
    save_best_only=True)

model.load_weights('best_model.keras')
# Тренировка модели
history = model.fit(train_data, batch_size=500, verbose=1, epochs=35,
                    validation_data=val_data,
                    callbacks=[model_checkpoint_callback])
