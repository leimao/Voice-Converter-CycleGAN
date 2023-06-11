
import os
import random
import numpy as np
import tensorflow as tf

def l1_loss(y, y_hat):
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    loss = tf.reduce_mean(tf.math.abs(y - y_hat),axis=[1, 2])
    return loss


def l2_loss(y, y_hat):
    return tf.losses.mean_squared_error((y, y_hat),axis=[1, 2])

def cross_entropy_loss(logits, labels):
    return tf.losses.sigmoid_cross_entropy((labels, logits),axis=[0, 2])

