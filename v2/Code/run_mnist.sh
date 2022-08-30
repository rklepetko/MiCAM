#!/bin/bash
tee() { mkdir -p ${1%/*} && command tee "$@"; }
time python -u model_micam.py\
 --cnn_model lenet5\
 --classes 10\
 --window_size 1\
 --height 75\
 --width 75\
 --batch_size 64\
 --num_epochs 3\
 --data_dir /MNIST/75x75/tfrecords\
 --learning_rate 1e-5\
 --log_dir /log/MNIST/lenet5/ | tee /log_run/MNIST/lenet5.out

