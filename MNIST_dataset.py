from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
'''Downloading data'''
if True:
    fasion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fasion_mnist.load_data()

'''Exploration'''
if True:
    print(
        train_images.shape,
        len(train_images),
        train_labels,
        test_images.shape,
        len(test_labels),
    )

'''Preprocesing'''
if True:
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.show()