import tensorflow as tf
import os
import random
import numpy as np

def l1_loss(y, y_hat):

    return tf.reduce_mean(tf.abs(y - y_hat))

def l2_loss(y, y_hat):

    return tf.reduce_mean(tf.square(y - y_hat))

def cross_entropy_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))


