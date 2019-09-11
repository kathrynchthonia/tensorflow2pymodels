# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 03:50:40 2019

@author: tony
"""

import os
import json
import random
import requests
import subprocess
import numpy as np
import tensorflowas as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

%matplotlib inline
tf.__version__

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train.shape

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[32,32,3]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=3, padding='valid'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10,activation='softmax'))

model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.fit(X_train, y_train, batch_size=128,epochs)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print('Test accuracy is {}'.format(test_accuracy))

MODEL_DIR = 'model/'
version = 1

export_path = os.path.join(MODEL_DIR, str(version))
export_path

if os.path.isdir(export_path):
    !rm -r {export_path}
    
tf.saved_model.simple_save(tf.keras.backend.get_session(), export_dir=export_path, inputs={'input_image':model.input}, outputs={t.name:t for t in model.outputs})

os.environ['MODEL_DIR'] =os.path.abspath(MODEL_DIR)

%%bash --bg
nohup tensorflow_model_server --rest_api_port8501 --model_name=cifar10 --model_base_path='${MODEL_DIR}' >server.log 2>&1

!tail server.log

random_image = np.random.randint(0, len(X_test))
random_image

data = json.dumps({'signature_name':'serving_default', 'instances':[X_test[random_image].tolist()]})
data

headers = {'content-type':'application/json}

json_response = requests.post(url="http://localhost:8501/v1/models/cifar10:predict", data=data, headers=headers)

json_response

predictions = json.loads(json_response.text)['predictions']

predictions

plt.imshow(X_test[random_image])

class_names[np.argmax(predictions[0])]

specific_json_response = requests.post(url="http://localhost:8501/v1/models/cifar10/versions/1:predict", data=data, headers=headers)

specific_json_response