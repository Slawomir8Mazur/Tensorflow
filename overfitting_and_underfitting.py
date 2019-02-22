#from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    result = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        result[i,word_indices] = 1.0
    return result

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)
#plt.plot(train_data[0])
#plt.show()

''' Defining baseline model'''
if True:
    baseline_model = keras.Sequential([
        keras.layers.Dense(16,activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    baseline_model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy', 'binary_crossentropy'])

    baseline_model.summary()

    baseline_history = baseline_model.fit(
        train_data,
        train_labels,
        epochs=10,
        batch_size=512,
        validation_data=(test_data, test_labels),
        verbose=2
    )

''' Defining small model'''
if False:
    small_model = keras.Sequential([
        keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(4, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    small_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy', 'binary_crossentropy'])

    small_model.summary()

    small_history = small_model.fit(
        train_data,
        train_labels,
        epochs=10,
        batch_size=512,
        validation_data=(test_data, test_labels),
        verbose=2
    )

''' Defining big model'''
if False:
    big_model = keras.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    big_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy', 'binary_crossentropy'])

    big_model.summary()

    big_history = big_model.fit(
        train_data,
        train_labels,
        epochs=10,
        batch_size=512,
        validation_data=(test_data, test_labels),
        verbose=2
    )


''' Ploting training and validation loss'''
if True:
    def plot_history(histories, key='binary_crossentropy'):
        plt.figure(figsize=(16,10))

        for name, history in histories:
            val = plt.plot(history.epoch, history.history['val_'+key],
                           '--', label=name.title()+'val')
            plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                     label=name.title()+'Trian')

        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_', ' ').title())
        plt.legend()

        plt.xlim([0, max(history.epoch)])

if False:
    plot_history([('baseline', baseline_history),
                  ('small', small_history),
                  ('big', big_history)])
    plt.show()

''' Regularization'''
if True:
    l2_model = keras.Sequential([
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    l2_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'binary_crossentropy'])

    l2_history = l2_model.fit(train_data,
                              train_labels,
                              epochs=10,
                              batch_size=512,
                              validation_data=(test_data, test_labels),
                              verbose=2)

    plot_history([('baseline', baseline_history),
                  ('l2', l2_history)])
    plt.show()

''' Dropout'''