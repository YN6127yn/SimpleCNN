# -*- coding : utf-8 -*-

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import numpy as np

def simple_cnn(input_shape, num_classes):
    """
    Simple CNN

    #Arguments
        input_shape: tuple(channels, rows, cols) or
                     tuple(rows, cols, channels),  Input shape.
        num_classes: Integer, number of classes.
    """

    # Define model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=5,
                     strides=(1, 1), padding='valid', input_shape=input_shape,
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=5,
                     strides=(1, 1), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=128, kernel_size=3,
                     strides=(1, 1), padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3,
                     strides=(1, 1), padding='valid',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
