
import os
import sys
import pathlib
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPool2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator

path = pathlib.Path('./Data/Training')
class_names = np.array(os.listdir(path))


train_dir = './Data/Training/'
test_dir = './Data/Testing/'
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)
train_data = train_datagen.flow_from_directory(train_dir, target_size=(224,224), class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(224,224), class_mode='categorical')

if sys.argv[1] == 'train':
    top_checkpt = tf.keras.callbacks.ModelCheckpoint(filepath='./weights/top_acc.ckpt', save_weights_only = False, 
                                                 monitor = 'val_accuracy', save_best_only = True,
                                                 save_freq='epoch')

    cnn_model = Sequential([
        Conv2D(filters=10, kernel_size=3, input_shape=train_data.image_shape, activation='relu'),
        MaxPool2D((2,2)),
        Conv2D(filters=10, kernel_size=3, activation='relu'),
        MaxPool2D((2,2)),
        Flatten(),
        Dense(4, activation='softmax')
    ])
    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    cnn_model.fit(train_data, epochs=20, shuffle=True, steps_per_epoch=len(train_data),
                validation_data = test_data, validation_steps = len(test_data),
                callbacks = [top_checkpt])

elif sys.argv[1] == 'test':
    cnn_model = tf.keras.models.load_model('./models/top_model')
    cnn_model.evaluate(test_data)

else:
    print("Invalid Call:\n\tpython3 project.py [train/test]")