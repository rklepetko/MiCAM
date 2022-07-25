# MiCAM Generator for Convolutional Neural Networks (CNN)

This folder contains research level code for the MiCAM visualization engine.
It contains 6 files:
- MNIST_to_tfrec.py
- model_vis.py
- vis_utils.py
- input.py
- flags.py
- evaluate.py
- lenet5.py
- res18.py
- run.sh

## MNIST_to_tfrec.py
This is a sample code that takes the MNIST dats and converts them into tensorflow records (TFRecords) for the CNN model.

## model_vis.py
This is the main module that builds and trains the CNN model, generates the CAM images, it uses the TFRecords as a source.

## vis_utils.py
This modified python module integrates the CAM images into the model layout diagram generator.

## input.py
The input module is responsible for fetching data to be used by the cnn model.  It uses tensorflow recordsas the source of data files and returns 3 datasets:  train (60% of files), validate (20% of files) and test (20% of files) datasets.

## flags.py
This file contains the flags/arguments needed to run the cnn_model.py

## evaluate.py
The evaluation module creates evaluate metrics operations.  First it calculates true positive, true negative, false positive and false negative.  Then they are used to calculate precision, recall, accuracy and f1 score.  Note/Heads-up: to avoid division by zero (NaN results), 1e-10 value is added to denominators.

## lenet5.py
This file defines a similar setup of "LeNet5" model.

## res18.py
This file defines a similar setup of "ResNet18" model.

## run.sh
This file contains an example of how to run the cnn_model.py

## Usage
Change the required arguments and run: ```./run.sh```
It is recommended to install tensorflow in a python virtualenv and run it inside a screen session.

The MiCAM plots are placed within the MCAM folder of the log file specified within the flag or input parameters.

To use [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) for graph visualization of the trained model, run ```python -m tensorboard.main --logdir=path_to_log_folder``` inside a screen session. By default tensorboard should be listening to port `6006`. If the training is running over an ssh server, redirect a local port to the `6006` port: ```ssh -i key_path -L local-port:127.0.0.1:6006 user@server-ip ```