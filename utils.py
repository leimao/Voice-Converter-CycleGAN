
import os
import random
import numpy as np
import tensorflow as tf

def l1_loss(y, y_hat):
    loss = tf.reduce_mean(tf.abs(y - y_hat))
    return loss


def l2_loss(y, y_hat):
    return tf.losses.mean_squared_error(y, y_hat)

def cross_entropy_loss(logits, labels):
    return tf.losses.sigmoid_cross_entropy(labels, logits)

