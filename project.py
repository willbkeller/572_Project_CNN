
import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPool2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator

path = pathlib.Path('./Data/Training')
class_names = np.array(os.listdir(path))
class_names

train_dir = './Data/Training/'
test_dir = './Data/Testing/'
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)
train_data = train_datagen.flow_from_directory(train_dir, target_size=(244,244), class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(244,244), class_mode='categorical')

cnn_model = Sequential([
    Conv2D(filters=10, kernel_size=3, input_shape=train_data.image_shape, activation='relu'),
    MaxPool2D((2,2)),
    Conv2D(filters=10, kernel_size=3, activation='relu'),
    MaxPool2D((2,2)),
    Flatten(),
    Dense(4, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn_model.fit(train_data, epochs=3, shuffle=True, steps_per_epoch=len(train_data))