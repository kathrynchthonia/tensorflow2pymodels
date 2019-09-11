# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:25:34 2019

@author: tony
"""

!apt-get install python-dev python-snappy

!pip install -q tensorflow_data_validation

import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdv
from __future__ import print_function

dataset = pd.read_csv('pollution_small.csv')

dataset.shape

training_data = dataset[:1600]

training_data.describe()

test_set = dataset[1600:]

test_set.describe()

#Data analysis and validation with tfdv
train_stats = tfdv.generate_statistics_from_dataframe(dataframe=dataset)

#Inferring the schema
schema =tfdv.infer_schema(statistics=train_stats)

tfdv.display_schema(schema)

#calculate test set statistics
test_stats = tfdv.generate_statistics_from_dataframe(dataframe=test_set)

# Stage4: compare test stats with the schema
anomalies = tfdv.validate_statistics(statistics=test_stats, schema=schema)

#display all detected anomalies in new data
anomalies = tfdv.validate_statistics(statistics=test_stats, schema=schema)

#Displayign all detected anomalies
tfdv.display_anomalies(anomalies)

#New data with anomalies
test_set_copy = test_set.copy()

test_set_copy.drop('soot', axis=1, inplace=True)

#stats based on data with anomalies
test_set_copy_stats = tfdv.generate_statistics_from_dataframe(dataframe=test_set_copy)

anomalies_new = tfdv.validate_statistics(statistics=test_set_copy_stats, schema=schema)

tfdv.display_anomalies(anomalies_new)

#Stage 5: prepare the schema for serving
schema.default_environment.append('TRAINING')
schema.default_environment.append('SERVING')

#Removing a target column from the Serving Schema
tfdv.get_feature(schema, 'soot').not_in_environment.append('SERVING')

#Checking for anomalies between serving environment and new test set
serving_env_anomalies = tfdv.validate_statistics(test_set_copy_stats, schema, environment='SERVING')

tfdv.display_anomalies(serving_env_anomalies)