# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:06:52 2019

@author: tony
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb

tf.__version__

number_of_words = 20000
max_len = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)

X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

vocab_size = number_of_words
vocab_size

embed_size = 164

model = tf.keras.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size, output_dim=164, input_shape=(X_train.shape[1],)))

model.add(tf.keras.layers.LSTM(units=164, return_sequences=True, activation='tanh'))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(units=128, return_sequences=True, activation='tanh'))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.LSTM(units=64,activation='sigmoid'))

model.add(tf.keras.layers.Dropout(0.2))

#adding the dense layer
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#compiling the model

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

#training the model
model.fit(X_train,y_train,epochs=3, batch_size=164)

#Evaluating the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print('Test accuracy: {}'.format(test_accuracy))


