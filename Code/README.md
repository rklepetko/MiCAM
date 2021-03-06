# MiCAM Generator for Convolutional Neural Networks (CNN)

This folder contains research level code for the MiCAM visualization engine.
It contains 10 files:
- README.md
- MNIST_to_tfrec.py
- model_vis.py
- vis_utils.py
- input.py
- flags.py
- evaluate.py
- lenet5.py
- res18.py
- run.sh

## README.md
This documentation file.

## MNIST_to_tfrec.py
This is a sample code that takes the MNIST data, upsamples and converts them into tensorflow records (TFRecords) for the CNN model.

## model_vis.py
This is the main module that builds and trains the CNN model and generates the CAM images.  The CNN model is specified within the input/flag parameters, and can be either defined within the code, such as the ResNet-18 and the NeNet-5 versions included, or a predefined model that is included within the Tensorflow Keras Application module.

## vis_utils.py
This modified python tensorflow module  that integrates the CAM images into the model layout diagram generator.

## input.py
The input module is responsible for fetching data to be used by the CNN model.  It uses TFRecord files as the data source and returns 3 datasets:  train (60% of files), validate (20% of files) and test (20% of files) datasets.

## flags.py
This file contains the flags/arguments needed to run the cnn_model.py

## evaluate.py
The evaluation module creates evaluate metrics operations.  First it calculates true positive, true negative, false positive and false negative.  Then they are used to calculate precision, recall, accuracy and f1 score.  Note/Heads-up: to avoid division by zero (NaN results), 1e-10 value is added to denominators.

## lenet5.py
This file defines a similar setup of "LeNet-5" model.

## res18.py
This file defines a similar setup of "ResNet-18" model.

## run.sh
This file contains an example of how to run the cnn_model.py

## Usage
Change the required arguments and run: ```./run.sh```
It is recommended to install tensorflow in a python virtualenv and run it inside a screen session.

The MiCAM plots are placed within the MCAM folder of the log directory specified within the flag or input parameters.

To use [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) for graph visualizations of the trained model, details on accuracy during training, and PR curves.   Run ```python -m tensorboard.main --logdir=path_to_log_folder``` inside a screen session. By default tensorboard should be listening to port `6006`.  If the training is running over an ssh server, redirect a local port to the `6006` port: ```ssh -i key_path -L local-port:127.0.0.1:6006 user@server-ip ```.