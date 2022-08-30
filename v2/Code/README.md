# MiCAM Generator for Convolutional Neural Networks (CNN)

This folder contains version-2 code for the MiCAM visualization engine.

This version differs on 4 major substantial points:
- First, we use 4 degrees of information using the RGBA color pallete to provide a deeper understanding for the variations in the positive and negative extremes between layers' plots.
- Second, we increase proficiency of the code by steamlining the pixel generation by interacting with the graphics engine only after all of the pixles for a layers has been generated.  This is opposed the a single pixel at a time which was inefficient.
- Third, we included the biases and weights in the CAM generation and guaranteed that all of the activations were included in the final CAM plot.  We also established concrete reference point within the deconvolution step for combined assimilation.
- Fourth, modularized code, extracting the MiCAM generator and task specific routines into their own modules.

It contains 13 files:
- README.md
- MNIST_to_tfrec.py
- model_micam.py
- micam.py
- picedits.py
- vis_utils.py
- input.py
- flags.py
- evaluate.py
- lenet3.py
- lenet5.py
- res18.py
- run_mnist.sh

## README.md
This documentation file.

## MNIST_to_tfrec.py
This is a sample code that takes the MNIST data, upsamples and converts them into tensorflow records (TFRecords) for the CNN model.

## model_micam.py
This is the main module that builds and trains the CNN model and and passes the model and a sample to the MiCAM generator.  The CNN model is specified within the input/flag parameters, and can be either defined within the code, such as the ResNet-18 and the LeNet-5 versions included, or a predefined model that is included within the Tensorflow Keras Application module.

## micam.py
This is primary MiCAM engine which generates the individual layer CAM plots and passes them to the model integrator.

## picedits.py
This code has image modification routines that massage the images to the appropriate size and shape for the integration.

## vis_utils.py
This modified python tensorflow module that integrates the CAM images into the model layout diagram generator.

## input.py
The input module is responsible for fetching data to be used by the CNN model.  It uses TFRecord files as the data source and returns 3 datasets:  train (60% of files), validate (20% of files) and test (20% of files) datasets.

## flags.py
This file contains the flags/arguments needed to run the model_micam.py

## evaluate.py
The evaluation module creates evaluate metrics operations.  First it calculates true positive, true negative, false positive and false negative.  Then they are used to calculate precision, recall, accuracy and f1 score.  Note/Heads-up: to avoid division by zero (NaN results), 1e-10 value is added to denominators.

## lenet3.py
This file defines a similar setup of "LeNet-3" model.

## lenet5.py
This file defines a similar setup of "LeNet-5" model.

## res18.py
This file defines a similar setup of "ResNet-18" model.

## run_mnist.sh
This file contains an example of how to run the model_micam.py.  Use absolute paths.

## Usage
Change the required arguments and run: ```./run_mnist.sh```
It is recommended to install tensorflow in a python virtualenv and run it inside a screen session.
The MiCAM generator requires absolute paths for the graphvis engine to properly receive the layers images.

The MiCAM plots are placed within the MCAM folder of the log directory specified within the flag or input parameters.

To use [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) for graph visualizations of the trained model, details on accuracy during training, and PR curves.   Run ```python -m tensorboard.main --logdir=path_to_log_folder``` inside a screen session. By default tensorboard should be listening to port `6006`.  If the training is running over an ssh server, redirect a local port to the `6006` port: ```ssh -i key_path -L local-port:127.0.0.1:6006 user@server-ip ```.