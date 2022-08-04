#!/usr/bin/env python
##                            Research
## Randy Klepetko                                 07/08/2022
#
## CNN Cam Plot
#
# This code is the lead training module that processes CNN models and exporting a MiCAM plot.
# This code was written for research, and isn't considered production quality.
# It could be improved by using a faster graphics engine in generating the CAM plots before model integration.
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Python Module Imports
import math, time, os, sys
import numpy as np
from numpy.fft  import fft2, ifft2
import tensorflow as tf
from tensorboard import summary as summ_lib
from tensorboardX import SummaryWriter
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from PIL import Image
import matplotlib.pyplot as plt
from keras import activations as kact

# Local Module Imports
import flags as f
import input, evaluate
# Local CNN Models
from lenet5 import lenet5
from res18 import resnet18
# Modified Keras Code: Local Module Imports
import vis_utils as vu

"""
FILE: model_vis.py
This is the starting point file where the main tensorflow graph is created.
It does the training, validation and testing, and the plots the CAM.
"""

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# Fix random generators seeds of tensorflow and numpy for debugging (to give similar results every run)
tf.compat.v1.set_random_seed(1)
np.random.seed(10)

# Initialize Plot Coordinate Storage
summaries = {'train': [], 'validate': [], 'test': []}
class_types = range(f.FLAGS.classes)
offsets = 0
offdist = 5

# this routine takes a set of images files and merges them horizontaly consecutively
def merge_horizontal_image(filenames,path):
    height = 0
    width = 0
    for filename in filenames:
        im = Image.open(filename)
        if im.height > height:
            height = im.height
        width += im.width
    imt = Image.new('RGBA',(width,height))
    width = 0
    for filename in filenames:
        im = Image.open(filename)
        imt.paste(im,(width,0))
        width += im.width
    imt.save(path)

# this routine takes a set of images files and merges them vertically consecutively
def merge_vertical_image(filenames,path):
    height = 0
    width = 0
    for filename in filenames:
        im = Image.open(filename)
        if im.width > width:
            width = im.width
        height += im.height
    imt = Image.new('RGBA',(width,height))
    height = 0
    for filename in filenames:
        im = Image.open(filename)
        imt.paste(im,(0,height))
        height += im.height
    imt.save(path)
    
def np_fftconvolve(A, B):
    return np.real(ifft2(fft2(A)*fft2(B, s=A.shape)))

# this routine takes the saved image and truncates it down to remove excess white space
def truncate_image(source,dest):
    im = Image.open(source)
    im = im.crop([590,520,1850,925])
    ns = im.size
    im = im.resize([int(im.width/2),int(im.height/2)])
    im.save(dest)

# This routine generates the text string for the CAM label
def gen_label(layer,act):
  #label = layer.__class__.__name__
  label = layer.name
  #label = '%s\n%s' % (layer.name, label)
  def format_dtype(dtype):
    if dtype is None:
      return '?'
    else:
      return str(dtype)
  #label = '%s|%s' % (label, format_dtype(layer.dtype))
  def format_shape(shape):
    return str(shape).replace(str(None), 'None')

  try:
    outputlabels = format_shape(layer.output_shape)
  except AttributeError:
    outputlabels = '?'
  if hasattr(layer, 'input_shape'):
    inputlabels = format_shape(layer.input_shape)
  elif hasattr(layer, 'input_shapes'):
    inputlabels = ', '.join([format_shape(ishape) for ishape in layer.input_shapes])
  else:
    inputlabels = '?'
  label = '%s\nI:%s\nO:%s' % (label,inputlabels,outputlabels)
  label = '%s\n%s' % (label, act)
  return(label)

# this routine plots the MiCAM diagram by first generating the individual CAM plots, 
# and then pass them in a list to the model layout generator
def plot_maps(model,features,label,pred,idx,cnn_name):

    if f.FLAGS.classes == 2:
        lab = list(label)[idx][1]       
        prd = str(np.argmax(pred,axis=1)[idx])+"("+str(pred[idx,np.argmax(pred,axis=1)[idx]])+")"
    else:
        lab = np.argmax(label,axis=1)[idx]  
        prd = str(np.argmax(pred,axis=1)[idx])+"("+str(pred[idx,np.argmax(pred,axis=1)[idx]])+")"
    lab = str(lab)

    prd = str(prd)
    
    cams = []
    
    input_size = model.layers[0].input_shape[0][1:3]

    if not os.path.exists(f.FLAGS.log_dir + "/MCAM/unit/"):
        os.makedirs(f.FLAGS.log_dir + "/MCAM/unit/")

    ### Generate Activations
    act_index = -1
    layer_idx = []
    layer_names = []
    layer_labels = []
    layer_weights = []
    layer_outputs = []
    conv_idx = []
    pool_idx = []
    
    # gather convolutional layers
    for layer in model.layers:
        act_index += 1;
        # check for convolutional layer, and stack activations with weights
        if 'Conv2D' in str(layer.__class__):
            weights = layer.get_weights()[0][:,:,0,:]

            layer_weights.append(weights)
            layer_idx.append(act_index)
            layer_names.append(layer.name)
            layer_outputs.append(layer.output)
        
            # generate label
            layer_labels.append(gen_label(layer,kact.serialize(layer.activation)))
            conv_idx.append(act_index)
                                                              
        #elif 'Pool' in str(layer.__class__):
        # stack non-convolution layers outputs (weight = 1)
        else:
            weights = 1
            layer_weights.append(weights)
            layer_idx.append(act_index)
            layer_names.append(layer.name)
            layer_outputs.append(layer.output)
        
            # generate label
            layer_labels.append(gen_label(layer,""))
            pool_idx.append(act_index)
        
            #print(layer.name)
            #print(str(layer.__class__))
            #print(layer.input)
            #print(layer.output)

            continue

    # build activation model and predict sample
    activation_model = models.Model(model.input, layer_outputs)
    activations = activation_model.predict(features[idx].reshape(1,features.shape[1],features.shape[2],features.shape[3]))
    
    # Gather global limits from activations
    amax = 0
    amin = 0
    for a_idx in range(len(layer_idx)):
        activation = activations[a_idx]
        amax = max(amax,np.max(activation))
        amin = min(amin,np.min(activation))

    ### Gather activations into CAMs
    cmaxs = 0
    cmins = 0
    for a_idx in range(len(layer_idx)):
        activation = activations[a_idx]
        norm = plt.Normalize(vmin=amin, vmax=amax)
        
        # plot activations (set to True)
        if (False and len(activation.shape)>2): 
            act_cnt = max(64,activation.shape[3])
            row_size = int(math.floor(math.sqrt(act_cnt)))
            col_size = int(math.ceil(act_cnt/row_size))
        
            activation_index=0
            fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*1.5,col_size*1.5))
            for row in range(0,row_size):
                for col in range(0,col_size):
                    #print((row,col,activation_index,activation.shape[3]))
                    if activation_index < act_cnt:
                        ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray', norm = norm)
                    
                    activation_index += 1
        
            fig.suptitle(cnn_name+" "+layer_names[a_idx]+" Activations\n"+"Sample-"+str(idx)+" label="+lab+" pred="+prd)
            filename_act = f.FLAGS.log_dir + "/MCAM/unit/"+cnn_name+"_"+layer_names[a_idx]+"_Activations-"+str(idx)+"_label-"+lab+"_pred-"+prd+".png"
            plt.savefig(filename_act, dpi=300, orientation='portrait')
            plt.close('all')

        #print(layer_names[a_idx])
        #print(layer_labels[a_idx])
        if layer_idx[a_idx] in conv_idx:
            cam = np.zeros(activation[0,:,:,0].shape, dtype = np.float32)
            weights = activation_model.layers[a_idx].get_weights()
            for i, w in enumerate(np.swapaxes(layer_weights[a_idx],0,2)):
                wact = np_fftconvolve(activation[0,:,:,i], 1/w)
                #wact = amin+wact / (amin+amax) # Global normalize
                cam += wact
                
        #elif layer_idx[a_idx] in pool_idx: 
        elif len(activation.shape)>3:
            cam = np.zeros(activation[0,:,:,0].shape, dtype = np.float32)
            #print(activation.shape)
            #print(np.swapaxes(activation,0,3).shape)
            #print(np.squeeze(np.swapaxes(activation,0,3)).shape)
            if (len(np.squeeze(np.swapaxes(activation,0,3)).shape) == 2):
                cam += np.squeeze(np.swapaxes(activation,0,3))
            else:
                for i, w in enumerate(np.squeeze(np.swapaxes(activation,0,3))):
                    #print(w.shape)
                    cam += w
            cam = np.divide(cam,activation.shape[3]) # Normalize
        elif len(activation.shape)>2:
            cam = np.zeros(activation[:,:,0].shape, dtype = np.float32)
            for i, w in enumerate(activation[:,:,-1]):
                cam += w
            cam = np.divide(cam,activation.shape[3]) # Normalize
        else:
            if (activation.shape[1]<input_size[0]):
                cam = activation[:,:]
            else:
                c = math.ceil(activation.shape[1]/input_size[0])
                cam = np.pad(np.squeeze(activation),(0,c*input_size[0]-activation.shape[1])).reshape(input_size[0],-1)
                layer_labels[a_idx] = '%s\n%s' % (layer_labels[a_idx], str(cam.shape))
                
        #print('cam.shape')
        #print(cam.shape)
        cmax = np.max(cam)
        cmin = np.min(cam)
        #print(cmax)
        #print(cmin)
        cmaxs = max(cmaxs,cmax)
        cmins = min(cmins,cmin)
        #print()
        
        # plot resulting cam (set to True)
        if (False): 
            val = np.multiply(np.divide(np.add(cam,-cmin),(cmax-cmin)),255) #  normalize pixels
            plt.title(cnn_name+" "+layer_names[a_idx]+" CAM\n"+"Sample-"+str(idx)+" label="+lab+" pred="+prd+"\n Max="+str(amax)+" Min="+str(amin))
            plt.imshow(val[:,:], cmap='gray', norm = norm) 
            filename_c = f.FLAGS.log_dir + "/MCAM/unit/"+cnn_name+"_"+layer_names[a_idx]+"_CAM-"+str(idx)+"_label-"+lab+"_pred-"+prd+".png"
            plt.savefig(filename_c, dpi=300, orientation='portrait')
            plt.close('all')
            #truncate_image(filename_gc)
                       
        # resize/upscale to match input image
        im = Image.fromarray(cam)
        im = im.resize(input_size,resample=Image.NEAREST)
        cam = np.asarray([np.asarray(im)])
        cams.append(cam)

    cams = np.concatenate(np.asarray([cams]),axis=0)

    # generate isometric 3d cam plots
    #print(cams.shape)
    filenames = []
    for pz in range(int(cams.shape[0])):
        #plt.title(cnn_name+" "+" Joined CAM\n"+"Sample-"+str(idx)+" label="+lab+" pred="+prd+"\n Max="+str(amax)+" Min="+str(amin))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        #lim_sc = max((int(cams.shape[0])-1)/6,1)
        #ax.set_xlim3d(-int(cams.shape[2])*lim_sc/3,int(cams.shape[2])*lim_sc/1.5)

        # Hide axis lines and marks
        ax.w_xaxis.line.set_lw(0)
        ax.set_xticks([])
        ax.w_yaxis.line.set_lw(0)
        ax.set_yticks([])
        ax.w_zaxis.line.set_lw(0)
        ax.set_zticks([])
        ax.w_xaxis.set_pane_color((1.0,1.0,1.0,0.0))
        ax.w_yaxis.set_pane_color((1.0,1.0,1.0,0.0))
        ax.w_zaxis.set_pane_color((1.0,1.0,1.0,0.0))

        ax.elev = 30
        ax.azim = 15
    
        cmin = np.min(cams[pz])
        cmax = max(np.max(cams[pz]),cmin*(1+(0.1*abs(cmin)/cmin)))
        #cmax = np.max(cams[pz])
        #cmin = min(np.min(cams[pz]),cmax*(1+(0.1*abs(cmax)/cmax)))
        for px in range(int(cams.shape[2])):
            for py in range(int(cams.shape[3])):
                val = float((cams[pz, 0, px, py]-cmin)/(cmax-cmin)) #  normalize pixels over this cam
                alv = float((cams[pz, 0, px, py]-cmins)/(cmaxs-cmins)) #  normalize transparency over all cams
                ax.scatter(px, py, 0, c = abs(float(val)*255), cmap='gray', alpha = alv, s=5*val*math.log((pz+1)*10), edgecolors='none', marker="s")
        ax.text(cams.shape[2]*0.8,cams.shape[3]*1.15,0,layer_labels[pz])
        #print(layer_labels[pz])
        filename = f.FLAGS.log_dir + "/MCAM/unit/"+cnn_name+"_CAM-"+str(idx)+"-"+str(pz)+"_label-"+lab+"_pred-"+prd+".png"
        plt.savefig(filename, dpi=300)
        filename_t = f.FLAGS.log_dir + "/MCAM/unit/"+cnn_name+"_CAM-"+str(idx)+"-"+str(pz)+"_label-"+lab+"_pred-"+prd+"trim.png"
        truncate_image(filename,filename_t)
        filenames.append(filename_t)
        plt.close('all')
        print(filename)
    # plot cam plots in vertical line, set to true
    if (False): 
        joined = f.FLAGS.log_dir + "/MCAM/"+cnn_name+"_MCAM_sample-"+str(idx)+"_label-"+lab+"_pred-"+prd+".png"
        merge_vertical_image(filenames,joined)
    
    # merge CAM plots with model diagram
    plot_name = f.FLAGS.log_dir + "/MCAM/"+cnn_name+"_sample-"+str(idx)+"_label-"+lab+"_pred-"+prd+".png"
    vu.plot_model_act(model,plot_name,image_layers=layer_names,layer_images=filenames)
    print(plot_name)

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
    if cnn_name in ['resnet18','lenet5']:
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

    model.summary()

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
        plot_maps(model,features,label,pred,idx+offset+1,cnn_name)
        plot_maps(model,features,label,pred,idx-offset,cnn_name)


if __name__ == "__main__":
    tf.compat.v1.app.run()
