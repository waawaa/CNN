# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:05:09 2019

@author: chacel
"""


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
    return images
images = read_images_to_array()
images_arr = np.asarray(images)
images_arr = images_arr.astype('float32')
images_arr = images_arr.reshape(-1, 64,64, 1)
images_arr = images_arr / np.max(images_arr)
print(images_arr)