#!/usr/bin/env python 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import numpy as np
import math
import tensorflow as tf
import tempfile
import flags as f

"""
FILE: evaluate.py
The evaluation module creates evaluate metrics operations
First it calculates true positive, true negative, false positive and false negative.
Then they are used to calculate precision, recall, accuracy and f1 score.

Note/Heads-up: to avoid division by zero (NaN results), we add 1e-10 value to denominators
"""

def pred(logits):
    preds = tf.argmax(input=logits, axis=1)

    ones_preds = tf.ones_like(preds)
    zeros_preds = tf.zeros_like(preds)

    return preds

def tp(logits, labels):
    preds = tf.argmax(input=logits, axis=1)
    actuals = tf.argmax(input=labels, axis=1)

    ones_actuals = tf.ones_like(actuals)
    zeros_actuals = tf.zeros_like(actuals)
    ones_preds = tf.ones_like(preds)
    zeros_preds = tf.zeros_like(preds)
    
    tp_op = tf.reduce_sum(
        input_tensor=tf.cast(
            tf.logical_and(tf.equal(actuals, ones_actuals), tf.equal(preds, ones_preds)),
            "float"
        )
    )
    return tp_op

def tn(logits, labels):
    preds = tf.argmax(input=logits, axis=1)
    actuals = tf.argmax(input=labels, axis=1)

    ones_actuals = tf.ones_like(actuals)
    zeros_actuals = tf.zeros_like(actuals)
    ones_preds = tf.ones_like(preds)
    zeros_preds = tf.zeros_like(preds)

    tn_op = tf.reduce_sum(
        input_tensor=tf.cast(
            tf.logical_and(tf.equal(actuals, zeros_actuals), tf.equal(preds, zeros_preds)),
            "float"
        )
    )
    return tn_op

def fp(logits, labels):
    preds = tf.argmax(input=logits, axis=1)
    actuals = tf.argmax(input=labels, axis=1)

    ones_actuals = tf.ones_like(actuals)
    zeros_actuals = tf.zeros_like(actuals)
    ones_preds = tf.ones_like(preds)
    zeros_preds = tf.zeros_like(preds)

    fp_op = tf.reduce_sum(
        input_tensor=tf.cast(
            tf.logical_and(tf.equal(actuals, zeros_actuals), tf.equal(preds, ones_preds)),
            "float"
        )
    )
    return fp_op

def fn(logits, labels):
    preds = tf.argmax(input=logits, axis=1)
    actuals = tf.argmax(input=labels, axis=1)

    ones_actuals = tf.ones_like(actuals)
    zeros_actuals = tf.zeros_like(actuals)
    ones_preds = tf.ones_like(preds)
    zeros_preds = tf.zeros_like(preds)

    fn_op = tf.reduce_sum(
        input_tensor=tf.cast(
            tf.logical_and(tf.equal(actuals, ones_actuals), tf.equal(preds, zeros_preds)),
            "float"
        )
    )
    return fn_op

def precision(tp, tn, fp, fn):
    tpr_op = tf.divide(tp, tf.add(tf.add(tp, 1e-10), fn))
    fpr_op = tf.divide(fp, tf.add(tf.add(fp, 1e-10), tn))
    precision_op = tf.divide(tf.add(tp, 1e-10), tf.add(tp, fp))
    return precision_op

def accuracy(tp, tn, fp, fn):
    tpr_op = tf.divide(tp, tf.add(tf.add(tp, 1e-10), fn))
    fpr_op = tf.divide(fp, tf.add(tf.add(fp, 1e-10), tn))
    accuracy_op = tf.divide(
        tf.add(tp, tn),
        tf.add(tf.add(tp, 1e-10), tf.add(fp, tf.add(tn, fn)))
    )
    return accuracy_op

def mAP(logits, labels):
    lbl = tf.cast(labels, tf.int64)
    ap = tf.compat.v1.metrics.average_precision_at_k(lbl, logits, 1)
    res = tf.reduce_mean(input_tensor=ap[0])

    return ap, res

def recall(tp, tn, fp, fn):
    tpr_op = tf.divide(tp, tf.add(tf.add(tp, 1e-10), fn))
    recall_op = tpr_op
    return recall_op

def fscore(tp, tn, fp, fn):
    precision_op = precision(tp=tp, fp=fp, fn=fn, tn=tn)
    recall_op = recall(tp=tp, fp=fp, fn=fn, tn=tn)
    fscore_op = tf.divide(
        tf.multiply(2.0, tf.multiply(precision_op, recall_op)),
        tf.add(tf.add(precision_op, 1e-10), recall_op)
    )
    return fscore_op

