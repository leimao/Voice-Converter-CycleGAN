import tensorflow as tf
import os
import random
import numpy as np

def l1_loss(y, y_hat):
    print( tf.math.reduce_mean(tf.math.abs(y - y_hat)))
    return tf.math.reduce_mean(tf.math.abs(y - y_hat))

def l2_loss(y, y_hat):

    return tf.math.reduce_mean(tf.math.square(y - y_hat))

def cross_entropy_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))

