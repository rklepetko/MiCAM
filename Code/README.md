# 3D Convolutional Neural Networks (CNN)

This folder contains 3d cnn code for online malware detection.
It contains 6 files:
- input.py
- lenet5.py
- cnn_model.py
- evaluate.py
- flags.py
- run.sh

## input.py

The input module is responsible for fetching data to be used by the cnn model.
It uses tensorflow records (TFRecords) as the source of data files.

It returns 3 datasets: train (60% of files), validate (20% of files) and test (20% of files) datasets.

I tried to follow the best practices of using tesnrflow dataset for performanc optimization. For further details see: [tensorflow-datasets-performance](https://www.tensorflow.org/guide/performance/datasets)

## lenet5.py
This file defines a similar setup of "LeNet5" model (7 layers) but using 3d CNN and different (parameters, dropout, batch_normalization and optimizer).

## cnn_model.py
This is the starting point file where the main tensorflow graph is created.
It does the training, validation and testing.
                                                                                
Note: This file doesn't save the model learned. See: [tensorflow-save-models](https://www.tensorflow.org/guide/saved_model) for saving and restoring models.

## evaluate.py
The evaluation module creates evaluate metrics operations.
First it calculates true positive, true negative, false positive and false negative.
Then they are used to calculate precision, recall, accuracy and f1 score.
Note/Heads-up: to avoid division by zero (NaN results), 1e-10 value is added to denominators.

## flags.py
This file contains the flags/arguments needed to run the cnn_model.py

## run.sh
This file contains an example of how to run the cnn_model.py

## Usage
Change the required arguments and run: ```./run.sh```
It is recommended to install tensorflow in a python virtualenv and run it inside a screen session.

To use [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) for graph visualization of the trained model, run ```python -m tensorboard.main --logdir=path_to_log_folder``` inside a screen session. By default tensorboard should be listening to port `6006`. If the training is running over an ssh server, redirect a local port to the `6006` port: ```ssh -i key_path -L local-port:127.0.0.1:6006 user@server-ip ```