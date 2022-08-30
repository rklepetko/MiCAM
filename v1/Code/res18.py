import time
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

counter = 3

class_types = range(f.FLAGS.classes)

def resnet18 (x):
  def add_common_layers (y, name):
    y = layers.BatchNormalization (name = name + '_bn')(y)
    y = layers.Activation ('relu', name = name + '_relu')(y)
    return y
  
  def add (y, name):
    y = layers.add (y, name = name + '_add')
    return y
  
  def conv (y, channel_size, kernel_size, stride, name):
    global counter
    if counter % 2 == 0:
      y = add_common_layers (y, name)
    y = layers.Conv2D (channel_size, (kernel_size, kernel_size), strides = stride, padding = 'same', name = name + '_conv_' + str (counter), kernel_regularizer=regularizers.l2(0.0005))(y)
    counter += 1
    return (y)
  
  def residual (y, channel_size, kernel_size, name, do_max_pool = False):
    global counter
    if do_max_pool == False:
      residue = conv (y, channel_size, kernel_size, 1, name + '_residue_1')
      #y = layers.Conv2D (channel_size, (1, 1), name = name + '_original')(y)
    else:
      residue = conv (y, channel_size, kernel_size, 2, name + '_residue_1')
      y = layers.Conv2D (channel_size, (1, 1), strides = 2, name = name + '_original_reshaped')(y)

    residue = conv (residue, channel_size, kernel_size, 1, name + '_residue_2')
    y = add ([y, residue], name + '_1')
    y = add_common_layers (y, name + '_1')
    
    residue = conv (y, channel_size, kernel_size, 1, name + '_residue_3')
    residue = conv (residue, channel_size, kernel_size, 1, name + '_residue_4')
    y = add ([y, residue], name + '_2')
    if counter != 19:
      y = add_common_layers (y, name + '_2')
    
    return y
  
  x = layers.Conv2D (64, (3, 3), strides = 1, padding='same', name='first_layer_112_1')(x)
  #x = layers.Conv2D (64, (3, 3), strides = 2, padding='same', name='second_layer_56_2')(x)
  x = add_common_layers (x, 'preparation')

  x = residual (x, 64, 3, 'B1')
  x = residual (x, 128, 3, 'B2', True)
  x = residual (x, 256, 3, 'B3', True)
  x = residual (x, 512, 3, 'B4', True)
  
  maxpool = layers.MaxPooling2D((4, 4))(x)
  avgPool = layers.AveragePooling2D ((4, 4))(x)
  
  x = layers.concatenate ([maxpool, avgPool])
  x = layers.Conv2D (10, 1, name = 'linear')(x)
  x = layers.Flatten ()(x)

  x = layers.Dense(len(class_types), activation='softmax', kernel_initializer='he_normal')(x)
  
  return x

