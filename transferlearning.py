# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:29:27 2019

@author: tony
"""

!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O ./cats_and_dogs_filtered.zip
    
import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator

%matplotlib inline
tf.__version__

dataset_path = './cats_and_dogs_filtered.zip'

zip_object = zipfile.ZipFile(file=dataset_path, mode='r')

zip_object.extractall('./')

zip_object.close()

dataset_path_new = './cats_adn_dogs_filtered/'

train_dir = os.path.join(dataset_path_new, 'validation')

# Building the model
IMG_SHAPE =(128,128,3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

base_model.summary()

base_model.tainable = False

base_model.output

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

global_average_layer

prediction_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(global_average_layer)

#Defining the model
model = tf.keras.layers.Dense(units=1, activation='sigmoid')(global_average_layer)
model.summary()

#Compiling the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

#creating data generators

data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)

train_generator = data_gen_train.flow_from_directory(train_dir, target_size=(128,128), batch_size=128, class_mode='binary')

valid_generator = data_gen_valid.flow_from_directory(validation_dir, target_size=(128,128), batch_size=128, class_mode='binary')

# training the model
model.fit_generator(train_generator, epochs=7, validation_data=valid_generator)

valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)

print('Accuracy after transfer learning: {}'.format(valid_accuracy))

#Fine Tuning
#unfreeze a few top layers from the model
base_model.trainable = True
print('Number of layers in the base model: {}'.format(len(base_model.layers)))

fine_tune_at = 130

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    
#Compiling the model for fine-tuning
    
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metics=['accuracy'])

#Fine tuning
model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)

#evaluating the fine tuned model
valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)

print('Validation accuracy after fine tuning: {}'.format(valid_accuracy))