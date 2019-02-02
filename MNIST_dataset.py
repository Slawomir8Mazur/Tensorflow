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

'''Model building'''
if True:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

'''Training'''
if True:
    model.fit(train_images, train_labels, epochs=3)

'''Evaluate'''
if True:
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy', test_acc)

'''Predictions'''
if True:
    predictions = model.predict(test_images)
    print(predictions[0],
          np.argmax(predictions[0]),
          test_labels[0])

    def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], \
                                             true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'


    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    i = 0
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions, test_labels)
    plt.show()

if True:
    img = test_images[0]
    print(img.shape)

    img = (np.expand_dims(img,0))
    print(img.shape)

    predictions_single = model.predict(img)
    print(predictions_single)
    plot_value_array(0, predictions_single, test_labels)
    np.argmax(predictions_single[0])
    plt.show()