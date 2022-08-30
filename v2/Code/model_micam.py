#!/usr/bin/env python
##                            Research
## Randy Klepetko                                 08/16/2022
#
## CNN MiCam Plotter V3
#
# FILE: model_micam.py
# This code was written for research, and isn't considered production quality.
# 
# Version 3 changes:
# - Included four degrees of parameters for CAM visualization taking advantage of the RGBA pallete.
# - Updated graphics engine to pass pixel parameter to the graphics engine as a grid instead of by pixel incresing performance.
# - Modularized code. 
#
# This code is the lead training module that processes CNN models and exporting a MiCAM plot.
# 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Python Module Imports
import math, time, os, sys
from scipy.special import logsumexp
import numpy as np
from numpy.fft  import fft2, ifft2
import tensorflow as tf
from tensorboard import summary as summ_lib
from tensorboardX import SummaryWriter
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from keras import activations as kact

# Local Module Imports
import micam as mc
import flags as f
import input, evaluate
# Local CNN Models
from lenet5 import lenet5
from lenet3 import lenet3
from res18 import resnet18
# Modified Keras Code: Local Module Imports

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.compat.v1.set_random_seed(1)
np.random.seed(10)

# Initialize Plot Coordinate Storage
summaries = {'train': [], 'validate': [], 'test': []}
class_types = range(f.FLAGS.classes)
offsets = 0
offdist = 20

def smp_str(label, pred, idx):
    if f.FLAGS.classes == 2:
        lab = list(label)[idx][1]       
        prd = str(np.argmax(pred,axis=1)[idx])+"("+str(pred[idx,np.argmax(pred,axis=1)[idx]])+")"
    else:
        lab = np.argmax(label,axis=1)[idx]  
        prd = str(np.argmax(pred,axis=1)[idx])+"("+str(pred[idx,np.argmax(pred,axis=1)[idx]])+")"
    return("#"+str(idx)+" label="+str(lab)+" pred="+str(prd))


# Main code that compiles te CNN model, trains, and passes the trained model and test samples to the MiCAM generator. 
def main(unused_argv):

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f.FLAGS.log_dir)

    # get datasets
    train_data, valid_data, test_data = input.input(shuffle_files=False)
    #Text information
    info = tf.constant(
        ["Batch size = %s" % f.FLAGS.batch_size,
         "Epochs = %s" % f.FLAGS.num_epochs,
         "Learning rate = %s" % f.FLAGS.learning_rate,
         "Batch normalization = No",
         "Window size = %s" % f.FLAGS.window_size,
         "Shuffle Files = No",
         "CNN model = %s" % f.FLAGS.cnn_model,
         "Shuffle Samples = YES"]
    )
    tf.summary.trace_on(graph=True)
  
    # load defaults from flags and parameters 
    img_width = f.FLAGS.width
    img_height = f.FLAGS.height 
    img_channels = f.FLAGS.window_size
    cnn_name = f.FLAGS.cnn_model
    dropout = f.FLAGS.dropout
    learn_r8 = f.FLAGS.learning_rate
    #print((img_height, img_width, img_channels))
    
    # Load CNN model
    image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    if cnn_name in ['resnet18','lenet5','lenet3']:
        cnn = eval(cnn_name)
        network_output = cnn(image_tensor)

        model = models.Model(inputs=[image_tensor], outputs=[network_output])
    else:
        cnn_model = eval("tf.keras.applications."+cnn_name)
    
        CNN_Model = cnn_model(input_tensor = image_tensor, \
            include_top = False, \
            weights = None, \
            classes = len(class_types))
    
        flatten_layer = tf.keras.layers.Flatten()(CNN_Model.output)
        dropout_layer = tf.keras.layers.Dropout(dropout)(flatten_layer)
        prediction_layer = tf.keras.layers.Dense(len(class_types),activation='softmax')(dropout_layer)
    
        model = models.Model(inputs=[image_tensor], outputs=prediction_layer)

    if (len(class_types)>2):
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learn_r8), metrics=['acc'])
    else:
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learn_r8), metrics=['acc'])

    #model.summary()

    batch_size=f.FLAGS.batch_size

    model_train = model.fit(train_data, 
                        epochs=f.FLAGS.num_epochs, 
                        validation_data=valid_data, 
                        #test_data=test_data, 
                        callbacks=[tensorboard])
                           
    # Evaluate the model on the test data using `evaluate`
    #print("Evaluate on test data")
    results = model.evaluate(test_data, batch_size=batch_size,callbacks=[tensorboard])
    #print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    pred = model.predict(test_data,callbacks=[tensorboard])
    #print("predictions shape:", pred.shape)
   
    ### Plot train and validation curves
    loss = model_train.history['loss']
    v_loss = model_train.history['val_loss']

    acc = model_train.history['acc']
    v_acc = model_train.history['val_acc']

    epochs = range(len(loss))

    # output pr curve to tensorboard
    pr_labls = []
    pr_preds = []
    label = np.concatenate([y for x, y in test_data], axis=0)
    features = np.concatenate([x for x, y in test_data], axis=0)
    
    # binary of multi class
    if f.FLAGS.classes == 2:
        pr_labl = tf.cast(label,tf.bool).numpy()
        pr_labl = np.array(np.reshape(label,len(pr_labl)*f.FLAGS.classes),dtype=bool)
    else:
        pr_labl = tf.argmax(label,axis=1)
        pr_labl = np.array(np.reshape(label,len(pr_labl)*f.FLAGS.classes))
    
    # export mAP
    pr_pred = tf.nn.softmax(pred).numpy()
    pr_pred = np.reshape(pr_pred,len(pred)*f.FLAGS.classes)
    pr_labls.extend(pr_labl)
    pr_preds.extend(pr_pred)
    map_writer = SummaryWriter(f.FLAGS.log_dir + "/mAP")
    map_writer.add_pr_curve('PR Curve', np.array(pr_labls), np.array(pr_preds), f.FLAGS.num_epochs, 512)
    
    # find the first item with a label of 1 (not zero)
    idx = 0
    while label[idx][0] == 1:
        idx += 1
    if idx < offsets+offdist:
        idx = offsets+offdist
        
    # plot the CAM for samples within offset from first sample labeled "1"
    for offset in range(offsets,offsets+offdist):
        mc.plot_maps(model,features[idx+offset+1],smp_str(label,pred,idx+offset+1),cnn_name)
        mc.plot_maps(model,features[idx-offset],smp_str(label,pred,idx-offset),cnn_name)

if __name__ == "__main__":
    tf.compat.v1.app.run()
