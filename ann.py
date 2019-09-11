# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 23:02:50 2019

@author: tony
"""

!pip install tensorflow-gpu==2.00.alpha0

import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

tf.__version__


# Load the fashion mnist dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

#Reshape the dataset
X_train = X_train.reshape(-1, 28*28)

X_tain.shape

X_test = X_test.reshape(-1, 28*28)

#Define the model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(748, )))

model.add(tf.keras.layers.Dense(units=64,activation='relu'))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

#compiling the model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.summary()


model.fit(X_train, y_train, epochs=10)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print('Test accuracy: {}'.format(teat_accuracy))

model_json = model.to_json()
with open('fashion_model.json', 'w') as json_file:
    json_file.write(model_json)
    
model.save_weights('fashion_model.h5')