import os

import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1,28*28) / 255.0
test_images = test_images[:1000]. reshape(-1, 28*28) / 255.0

def create_model():
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(28*28,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model

model = create_model()
model.summary()

''' Saving checkpoints during training'''
if False:
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model = create_model()

    model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels),
              callbacks=[cp_callback])              #Saving

''' Restore model'''
if False:
    new_model = create_model()

    loss, acc = new_model.evaluate(test_images, test_labels)
    print("Untrained model, accuracy: {:5.2g}%".format(100*acc))

    new_model.load_weights(checkpoint_path)   # Restoring
    loss, acc = new_model.evaluate(test_images, test_labels)
    print("Untrained model, accuracy: {:5.2g}%".format(100*acc))

''' Checkpoint callback options'''
if True:
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True, period=5    #Save weights every 5-epochs
    )

    model = create_model()
    model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(train_images, train_labels,
              epochs=50, callbacks=[cp_callback],
              validation_data=(test_images, test_labels),
              verbose=0)

    ''' Bringing back'''
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    new_model =create_model()
    model.load_weights(latest)
    loss, acc = new_model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}".format(100*acc))