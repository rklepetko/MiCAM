#!/usr/bin/env python
##                            Research
## Randy Klepetko                                 08/16/2022
#
## CNN MiCam Plotter V3
#
# This code was written for research, and isn't considered production quality.
# 
# Version 3 changes:
# - Included four degrees of parameters for CAM visualization taking advantage of the RGBA pallete.
# - Updated graphics engine to pass pixel parameter to the graphics engine as a grid instead of by pixel incresing performance.
# - Ugraded flat layer representations and included all of the weights and biases in the calculations per layer.
# - Modularized code. 
#
# This code is the MiCAM plot module.
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
from tensorflow.keras import layers, models
from PIL import Image
import matplotlib.pyplot as plt
from keras import activations as kact

# Local Module Imports
import picedits as pe
import flags as f
import vis_utils as vu

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# Fix random generators seeds of tensorflow and numpy for debugging (to give similar results every run)
tf.compat.v1.set_random_seed(1)
np.random.seed(10)

# Initialize Plot Coordinate Storage
summaries = {'train': [], 'validate': [], 'test': []}
class_types = range(f.FLAGS.classes)
offsets = 0
offdist = 5
    
def np_fftconvolve(A, B):
    return np.real(ifft2(fft2(A)*fft2(B, s=A.shape)))

def deconvolve(A, B):
    idx_cnt = len(A.shape)
    loc = []
    rol = []
    if (B.shape[0] != B.shape[1]):
      print("act","flt")
      print(A.shape,B.shape)
    for idx in range(idx_cnt):
      Av = A.shape[idx]
      Bv = B.shape[idx]
      b = math.floor((Av - Bv) / 2)
      a = Av - b - Bv
      #loc.append((int(b),int(a)))
      #rol.append(math.floor(Av/2))
      loc.append((0,Av-Bv))
      rol.append(0)
    if (B.shape[0] != B.shape[1]):
      print("loc","rol")
      print(loc,rol)
    Bt = np.roll(np.pad(B,tuple(loc), 'constant'),rol)
    return np.real(ifft2(fft2(A)/fft2(Bt)))

# This routine generates the text string for the CAM label
def gen_label(layer):
  label = layer.__class__.__name__
  label = '%s\n%s' % (layer.name, label)
  def format_dtype(dtype):
    if dtype is None:
      return '?'
    else:
      return str(dtype)
  #label = '%s|%s' % (label, format_dtype(layer.dtype))
  def format_shape(shape):
    rtn = str(shape).replace(str(None), 'None')
    rtn = rtn.replace('), (','),\n  (')
    return rtn

  try:
    outputlabels = format_shape(layer.output_shape)
  except AttributeError:
    outputlabels = '?'
  if hasattr(layer, 'input_shape'):
    inputlabels = format_shape(layer.input_shape)
  elif hasattr(layer, 'input_shapes'):
    inputlabels = ',\n  '.join([format_shape(ishape) for ishape in layer.input_shapes])
  else:
    inputlabels = '?'
  label = '%s\nI:%s\nO:%s' % (label,inputlabels,outputlabels)
  return(label)

# this routine plots the MiCAM diagram by first generating the individual CAM plots, 
# and then pass them in a list to the model layout generator
def plot_maps(model,sample,smp_str,cnn_name, plt_act = False, plt_fft = False, plt_cam = False):
    
    fil_smp = smp_str.replace(" ","_")
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
    layer_bias = []
    layer_outputs = []
    conv_idx = []
    pool_idx = []
    
    # gather convolutional layers
    for layer in model.layers:
        act_index += 1;
        # check for convolutional layer, and stack activations with weights
        if 'Conv2D' in str(layer.__class__):
            weights = layer.get_weights()

            layer_weights.append(weights)
            layer_idx.append(act_index)
            layer_names.append(layer.name)
            layer_outputs.append(layer.output)
        
            # generate label
            llabel = gen_label(layer)
            llabel = '%s\n%s' % (llabel , kact.serialize(layer.activation))
            llabel = '%s\n%s' % (llabel , str(weights[0].shape))
            for w in weights[1:]:
              llabel = llabel + " + " +str(w.shape)
            layer_labels.append(llabel)
            conv_idx.append(act_index)
                                                              
        else:
            layer_weights.append(layer.get_weights())
            layer_idx.append(act_index)
            layer_names.append(layer.name)
            layer_outputs.append(layer.output)
        
            # generate label
            layer_labels.append(gen_label(layer))
            pool_idx.append(act_index)
        
            continue

    # build activation model and predict sample
    activation_model = models.Model(model.input, layer_outputs)
    activations = activation_model.predict(sample.reshape(1,sample.shape[0],sample.shape[1],sample.shape[2]))
    
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
        if (plt_act and len(activation.shape)>2): 
            act_cnt = max(64,activation.shape[3])
            row_size = math.floor(math.sqrt(act_cnt/1.5))
            col_size = math.ceil(act_cnt/row_size)
            activation_index=0
            fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*1.5,col_size*1.5))
            for row in range(0,row_size):
                for col in range(0,col_size):
                    if activation_index < act_cnt and activation_index < activation.shape[3]:
                        actv = activation[0, :, :, activation_index]
                        actv = (actv-np.min(actv))/(np.max(actv)-np.min(actv))
                        ax[row][col].imshow(actv*255, cmap = 'gray', norm = norm)
                    activation_index += 1
        
            fig.suptitle(cnn_name+" "+layer_names[a_idx]+" Activations\n"+smp_str)
            filename_act = f.FLAGS.log_dir + "/MCAM/unit/"+cnn_name+"_"+layer_names[a_idx]+"_Activations-"+fil_smp+".png"
            plt.savefig(filename_act, dpi=300, orientation='portrait')
            plt.close('all')

        if layer_idx[a_idx] in conv_idx:
            
            if plt_fft: 
                act_cnt = max(64,layer_weights[a_idx][0].shape[3]*layer_weights[a_idx][0].shape[2])
                row_size = math.floor(math.sqrt(act_cnt/1.5))
                col_size = math.ceil(act_cnt/row_size)
                fig2, ax2 = plt.subplots(row_size, col_size, figsize=(row_size*1.5,col_size*1.5))
                act_idx = 0

            cam = np.zeros(activation[0,:,:,0].shape, dtype = np.float32)

            #print(layer_labels[a_idx])
            #print("act")
            #print(activation.shape)
            #print("wht")
            #print(layer_weights[a_idx][0].shape)
            for i, v in enumerate(np.swapaxes(layer_weights[a_idx][0],0,3)):
                for j, w in enumerate(np.swapaxes(v,0,1)):
                  if (w.shape[1] == activation.shape[0]) and (w.shape[0] == activation.shape[1]) and (w.shape[0] == 1) and (w.shape[1] != 1):
                    wact = deconvolve(activation[:,0,:,i], w)
                  else:
                    wact = np_fftconvolve(activation[0,:,:,i], w)
                  
                  if len(layer_weights[a_idx]) > 1:
                    #print("Mult")
                    #print(wact.shape,layer_weights[a_idx][1].shape)
                    if len(layer_weights[a_idx][1].shape) == 4:
                      cam += wact*layer_weights[a_idx][1][0,0,i,j]
                    else:
                      cam += wact*layer_weights[a_idx][1][i]
                  else:
                    cam += wact*1/activation.shape[3]
                    
                  if plt_fft:
                    col = math.floor(act_idx / row_size)
                    row = act_idx % row_size
                    actv = (wact-np.min(wact))/(np.max(wact)-np.min(wact))
                    ax2[row][col].imshow(actv*255, cmap = 'gray', norm = norm)
                    act_idx += 1
            if plt_fft: 
              fig2.suptitle(cnn_name+" "+layer_names[a_idx]+" IFFT-"+str(act_cnt)+"\n"+smp_str)
              filename_act = f.FLAGS.log_dir + "/MCAM/unit/"+cnn_name+"_"+layer_names[a_idx]+"-"+str(idx)+"_IFFT-"+fil_smp+".png"
              plt.savefig(filename_act, dpi=300, orientation='portrait')
              plt.close('all')
            
        #elif layer_idx[a_idx] in pool_idx: 
        elif len(activation.shape)>3:
            cam = np.zeros(activation[0,:,:,0].shape, dtype = np.float32)
            if (np.swapaxes(activation,0,3).shape[0] == 1 and np.swapaxes(activation,0,3).shape[3] == 1):
                cam += activation[0,:,:,0]
            else:
                for i, w in enumerate(np.swapaxes(activation,0,3)):
                    cam += w[:,:,0]
                cam = np.divide(cam,activation.shape[3]) # Normalize
        elif len(activation.shape)>2:
            cam = np.zeros(activation[:,:,0].shape, dtype = np.float32)
            for i, w in enumerate(activation[:,:,-1]):
                cam += w*layer_weights[a_idx][1][i] # Normalize included in weights
        else:
            if (activation.shape[1]<input_size[0]):
                cam = activation[:,:]
            else:
                cc = math.ceil(activation.shape[1]/min((math.sqrt(activation.shape[1])*max(input_size)/min(input_size)),max(input_size)))
                rc=math.ceil(activation.shape[1]/cc)
                cam = np.pad(np.squeeze(activation),(0,cc*rc-activation.shape[1])).reshape(cc,rc)
                layer_labels[a_idx] = '%s\n%s' % (layer_labels[a_idx], str(cam.shape))
                
        cmax = np.max(cam)
        cmin = np.min(cam)
        cmaxs = max(cmaxs,cmax)
        cmins = min(cmins,cmin)
        
        # plot resulting cam (set to True)
        if (plt_cam): 
            val = np.multiply(np.divide(np.add(cam,-cmin),(cmax-cmin)),255) #  normalize pixels
            plt.title(cnn_name+" "+layer_names[a_idx]+" CAM\n"+smp_str+"\n Max="+str(amax)+" Min="+str(amin))
            plt.imshow(val[:,:], cmap = 'gray', norm = norm) 
            filename_c = f.FLAGS.log_dir + "/MCAM/unit/"+cnn_name+"_"+layer_names[a_idx]+"_CAM-"+fil_smp+".png"
            plt.savefig(filename_c, dpi=300, orientation='portrait')
            plt.close('all')
                       
        # resize/upscale to match input image
        im = Image.fromarray(cam)
        im = im.resize(input_size,resample=Image.NEAREST)
        cam = np.asarray([np.asarray(im)])
        cams.append(cam)

    cams = np.concatenate(np.asarray([cams]),axis=0)

    # generate isometric 3d cam plots
    filenames = []
    for pz in range(int(cams.shape[0])):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

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
        if cmin == 0.0:
          cmax = np.max(cams[pz])
        elif cams[pz].all == 0:
          cmax = 0.00000001
        else:
          a = np.max(cams[pz])
          b = 0.1*abs(cmin)
          b = cmin*(1+(0.1*abs(cmin)/cmin))
          cmax = max(a,b)

        val = (cams[pz, 0, :, :]-cmin)/(cmax-cmin)  #  normalize pixels over this cam
        alv = np.sqrt((cams[pz, 0, :, :]-cmins)/(cmaxs-cmins))     #  normalize transparency over all cams
        col_b = val
        col_r = np.maximum(cams[pz, 0, :, :],0)/cmax
        if cmin == 0:
          col_g = np.zeros(cams[pz, 0, :, :].shape)
        else:
          col_g = np.maximum(-cams[pz, 0, :, :],0)/-cmin

        col=np.dstack((col_r,col_g))
        col=np.dstack((col,col_b))
        col=np.dstack((col,alv))
        col=col.reshape((val.shape[0]*val.shape[1],4))

        xa = np.repeat(np.array([range(val.shape[1],0,-1)]),val.shape[0],axis=0)
        ya = np.repeat(np.array(range(val.shape[0]))[...,None],val.shape[1],axis=1)
        za = np.zeros(val.shape)
        siz = 5*val*math.log((pz+1)*10)

        plabel = layer_labels[pz]+"\nmin:"+str(cmin)+"\nmax:"+str(cmax)
        ax.text(cams.shape[3]*0.9,cams.shape[2]*1.3,0,plabel,size="smaller",c='black')
        ax.scatter(xa, ya, za,  c = col, s=siz, edgecolors='none', marker="s")

        filename = f.FLAGS.log_dir + "/MCAM/unit/"+cnn_name+"_CAM-"+fil_smp+"-"+str(pz)+".png"
        plt.savefig(filename, dpi=300)
        filename_t = f.FLAGS.log_dir + "/MCAM/unit/"+cnn_name+"_CAM-"+fil_smp+"-"+str(pz)+"trim.png"
        pe.truncate_image(filename,filename_t,[580,530,1850,940])
        filenames.append(filename_t)
        plt.close('all')
        #print(filename)
    # plot cam plots in vertical line, set to true
    if (plt_cam): 
        joined = f.FLAGS.log_dir + "/MCAM/"+cnn_name+"_MCAM_sample-"+fil_smp+".png"
        pe.merge_vertical_image(filenames,joined)
    
    # merge CAM plots with model diagram
    plot_name = f.FLAGS.log_dir + "/MCAM/"+cnn_name+"_sample"+fil_smp+".png"
    vu.plot_model_act(model,plot_name,image_layers=layer_names,layer_images=filenames)
    #print(plot_name)

