
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
import tensorflow as tf

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=(64,64,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

batch_size=32

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
tf.keras.utils.plot_model(
    model,
    to_file='model.png',
    show_shapes=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False,
    dpi=96
)

'''
from glob import glob
import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def read_images_to_array():

    data = glob('dataset/training_set/processed/*', recursive=False)
    images = []
    for i in range(2):
        img = misc.imread(data[i])
        img = misc.imresize(img, (64,64))
        images.append(img)

    images = read_images_to_array()
    images_arr = np.asarray(images)
    images_arr = images_arr.astype('float32')
    images_arr = images_arr.reshape(-1, 64,64, 1)
    images_arr = images_arr / np.max(images_arr)
    return images_arr


def read_correct_images_to_array():

    data = glob('dataset/training_set/cats/*', recursive=False)
    images = []
    for i in range(2):
        img = misc.imread(data[i])
        img = misc.imresize(img, (64,64))
        images.append(img)

    images = read_images_to_array()
    images_arr = np.asarray(images)
    images_arr = images_arr.astype('float32')
    images_arr = images_arr.reshape(-1, 64,64, 1)
    images_arr = images_arr / np.max(images_arr)
    return images_arr


train_X = read_images_to_array()
train_Y = read_correct_images_to_array()









model.fit(train_X,train_Y, epochs=25, steps_per_epoch=4000/32, use_multiprocessing=True, workers=20)

model.save("model_pixel.h5")'''