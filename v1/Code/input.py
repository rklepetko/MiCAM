from __future__ import print_function
import os, sys, glob, json, numpy as np, random, time, gc
import tensorflow as tf
import flags as f
import pdb

"""
FILE: input.py
The input module creates and returns 3 datasets of TFRecords. It divides TFRecords to 3 datasets:
  training dataset (60% of files)
  validation dataset (20% of files)
  testing dataset (20% of files)

I tried to follow the best practices of using tesnrflow dataset for performance optimization.
For further details see: https://www.tensorflow.org/guide/performance/datasets
"""

SAMPLE_HEIGHT = f.FLAGS.height
SAMPLE_WIDTH = f.FLAGS.width
SAMPLE_DEPTH = f.FLAGS.window_size

#Parse the tfrecords and reshaping the input to 3d image format (3d matrix)
def _parse_fn(batch_record):
    global IMG_COUNT
    global IMG_VARIATION
#    global IMG_ENTROPY
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'sample': tf.io.FixedLenFeature([SAMPLE_HEIGHT * SAMPLE_WIDTH * SAMPLE_DEPTH], tf.float32)
    }
    parsed_features = tf.io.parse_example(serialized=batch_record, features=features)
    data = parsed_features["sample"]
    #print(data.shape)
    data = tf.reshape(data, [f.FLAGS.batch_size, SAMPLE_HEIGHT, SAMPLE_WIDTH, SAMPLE_DEPTH])
    labels = parsed_features["label"]
    #print(data.shape)
    #print("Labels")
    #print(labels.shape)
    #print(labels)
    if (f.FLAGS.classes == 2):
        lab = tf.one_hot(labels, f.FLAGS.classes)
        #labels = tf.one_hot(labels, f.FLAGS.classes)
    else:
        #labels = tf.cast(labels, tf.int32)
        #labels = tf.keras.utils.to_categorical(labels, f.FLAGS.classes)
        #lab = tf.zeros([labels.shape[0],f.FLAGS.classes])
        #lab[labels] = 1
        #label = list(labels)
        #print(label)
        lab = tf.one_hot(labels, f.FLAGS.classes)
        
        #labels= tf.reshape(labels,[labels.shape[0],1,labels.shape[1]])
    #print(labels.shape)
    #print(labels)
    return data, lab

def _get_dataset(files, batch_size, shuffle_data):
    dataset = tf.data.TFRecordDataset(files)
    if shuffle_data:
        dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(_parse_fn, num_parallel_calls=8)
    #dataset = tf.data.Dataset.from_tensor_slices(dataset).batch(batch_size, drop_remainder=True)
    return dataset

#Divide the files and returns 3 datasets (train, validate, and test)
def input(shuffle_files=False):

    IMG_COUNT = 1
    IMG_VARIATION = 0
    IMG_ENTROPY = 0
    
    files = glob.glob(f.FLAGS.data_dir + "/*.tfrecord")
    if len(files) == 0:
        exit("TFRecords directory is empty")
    if len(files) == 1:
        data = _get_dataset(files=files, batch_size=f.FLAGS.batch_size, shuffle_data=False)
        cnt = tf.data.experimental.cardinality(data).numpy()
        cnt = 0
        for x,y in data:
          cnt = cnt + 1
        #print(cnt)
        train_data = data.take(int(cnt*0.6))
        test_data = data.skip(int(cnt*0.6))
        validate_data = test_data.take(int(cnt*0.2))
        test_data = test_data.skip(int(cnt*0.2))
        print ("Training images:",int(cnt*.6))
        print ("Validate images",int(cnt*.2))
        print ("Testing images",int(cnt*.2))
    else:
        if shuffle_files:
            random.shuffle(files)

        #Dividing data
        train_files = files[:int(0.6*len(files))]
        validate_files = files[int(0.6*len(files)):int(0.80*len(files))]
        test_files = files[int(0.80*len(files)):]
    
        print ("Number of training files: ", len(train_files))
        print ("Number of validation files: ", len(validate_files))
        print ("Number of testing files: ", len(test_files))
    
        train_data = _get_dataset(files=train_files, batch_size=f.FLAGS.batch_size, shuffle_data=True)
        validate_data = _get_dataset(files=validate_files, batch_size=f.FLAGS.batch_size, shuffle_data=False)
        test_data = _get_dataset(files=test_files, batch_size=f.FLAGS.batch_size, shuffle_data=False)

    return train_data, validate_data, test_data
