# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:47:07 2019

@author: tony
"""

!pip install tensorflow-gpu==2.00.alpha0

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

%matplotlib inline
tf.__version__


#Setting class names for the dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Loading the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#Image normalization
X_train = X_train / 255.0
X_train.shape

X_test = X_test / 255.0
plt.imshow(X_test[10])


model = tf.keras.models.Sequential()

model.add(tf.keras.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[32,32,3]))

#Adding 2nd CNN layer and max pool layer

model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2,padding='valid'))

#Adding the third cnn Layer

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

model.add(tf.keras.layers.Flatten())

#Adding the first dense layer
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

#Adding the second dense layer
model.add(tf.keras.Dense(units=10, activation='softmax'))

model.summary()

#compiling the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['sparse_categorical_accuracy'])

#Training the model
model.fit(X_train, y_train, epochs=12)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print('Test accuracy: {}'.format(test_accuracy))


