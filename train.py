# -*- coding : utf-8 -*-

import keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras import backend as K
import matplotlib
matplotlib.use('AGG')
import matplotlib.pylab as plt
import numpy as np
import os
import pathlib
from simple_cnn import simple_cnn

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# image size
img_rows, img_cols = 28, 28

# number of classes
num_classes = 10 #0-9

# Reshape data for each data format
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Convert pixel data(0-255) to one-hot(0 or 1)
x_train /= 255
x_test /= 255

y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

# Convert number(0-9) to one-hot(0 or 1) array
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Use simple CNN
model = simple_cnn(input_shape, num_classes)

batch_size = 100
epochs = 20
history = model.fit(x_train, y_train,
                   batch_size=batch_size, epochs=epochs, verbose=1,
                   validation_data=(x_test, y_test))

# Make Result directory
abs_pardir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(abs_pardir, 'result')
pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

# Save model and weight
model_json = model.to_json()
open(os.path.join(result_dir, 'model.json'), 'w').write(model_json)
model.save_weights(os.path.join(result_dir, 'weight.h5'))

# Make history graph
fig = plt.figure(figsize=(8,5))
# Accuracy
acc = fig.add_subplot(211)
acc.plot(history.history['acc'])
acc.plot(history.history['val_acc'])
acc.set_ylabel('accuracy')
acc.legend(['train', 'test'], loc='lower right')
# Loss
loss = fig.add_subplot(212)
loss.plot(history.history['loss'])
loss.plot(history.history['val_loss'])
loss.set_ylabel('loss')
loss.set_xlabel('epoch')
loss.legend(['train', 'test'], loc='lower right')
fig.savefig(os.path.join(result_dir, 'history.png'), dpi=100)
