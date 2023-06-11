
import os
import random
import numpy as np
import tensorflow as tf

def l1_loss(y, y_hat):
    y = tf.cast(y, dtype=tf.float32)  # Cast y to float32
    y_hat = tf.cast(y_hat, dtype=tf.float32)  # Cast y_hat to float32
    loss = tf.reduce_mean(tf.abs(y - y_hat))
    return loss


def l2_loss(y, y_hat):
    return tf.losses.mean_squared_error((y, y_hat),axis=[1, 2])

def cross_entropy_loss(logits, labels):
    return tf.losses.sigmoid_cross_entropy((labels, logits),axis=[0, 2])

