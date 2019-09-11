# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 23:22:57 2019

@author: tony
"""

!pip install tensorflow-transform

import tempfile
import pandas as pd
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam

from __future__ import print_function

from tensorflow_transform.tf_metadata import dataset_metadata, dataset_schema

#dataset preprocessing

#load the pollution dataset
dataset = pd_readcsv('pollution_small.csv')

dataset.head()

#dropping the data column
features = dataset.drop('Date', axis=1)
features.head()

# converting the dataset from dataframe to list of Python dictionaries

dict_features = list(features.to_dict('index').values())
dict_features[:2]

# defining the dataset metadata

data_metadate = dataset_metadata.DatasetMetadata(
        dataset_schema.from_feature_spec({
                'no2':tf.FixedLenFeature([], tf.float32),
                'so2':tf.FixedLenFeature([], tf.float32)),
                'pm10':tf.FixedLenFeature([], tf.float32),
                'soot':tf.FixedLenFeature([], tf.float32)
                }))
data_metadata

#preprocessing function
def preprocessing_fn(inputs):
    no2 = inputs['no2']
    pm10 = inputs['pm10']
    so2 = inputs['so2']
    soot = inputs['soot']
    
    no2_normalized = no2 - tftmean(no2)
    so2_normalized = so2 - tft.mean(so2)
    
    pm10_normalized = tft.scale_to_0_1(pm10)
    soot_normalized = tft.scale_by_min_max(soot)
    
    return {
            'no2_normalized':no2_normalized,
            'so2_normalized':so2_normalized,
            'pm10_normalized':pm10__normalized,
            'soot_normalized':soot_normalized
            }
    
#putting everything together
def data_transform():
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
        transformed_dataset, transform_fn = ((dict_features, data_metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
        
        transformed_data, transformed_metadata = transformed_dataset
        
        for i in range(len(transformed_data)):
            print('Raw: ', dict_features[i])
            print('Transformed: ', transformed_data[i])
            
data_transform()