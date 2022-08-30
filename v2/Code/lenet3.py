import time, math
from tensorflow import keras
import numpy as np
from tensorflow .keras import regularizers
import matplotlib.pyplot as plt
import np_utils
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import flags as f
import pydot
import graphviz
import flags as f

counter = 3

def lenet3 (x):
  
  x = layers.Conv2D (32, (5, 5), strides = 1, padding='same', name='first_conv_layer')(x)
  x = layers.MaxPooling2D((2, 2), name='first_pool_layer')(x)
  x = layers.Conv2D (64, (3, 3), strides = 1, padding='same', name='second_conv_layer')(x)
  x = layers.MaxPooling2D((2, 2), name='second_pool_layer')(x)
  x = layers.Flatten()(x)
  x = layers.Dense(4096, activation='relu')(x)
  x = keras.layers.Dropout(f.FLAGS.dropout)(x)
  x = layers.Dense(1600, activation='relu')(x)
  x = keras.layers.Dropout(f.FLAGS.dropout)(x)
  x = layers.Dense(f.FLAGS.classes, activation='softmax')(x)
  
  return x

