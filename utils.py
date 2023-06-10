
import os
import random
import numpy as np
import tensorflow as tf

def l1_loss(y, y_hat):
    if y.shape[-1] != y_hat.shape[-1]:
        y_shape = tf.shape(y)
        y_hat_shape = tf.shape(y_hat)
        max_dim = tf.maximum(y_shape[-1], y_hat_shape[-1])
        y = tf.broadcast_to(y, tf.concat([y_shape[:-1], [max_dim]], axis=0))
        y_hat = tf.broadcast_to(y_hat, tf.concat([y_hat_shape[:-1], [max_dim]], axis=0))
    loss = tf.reduce_mean(tf.abs(y - y_hat))
    return loss



def l2_loss(y, y_hat):
    return tf.losses.mean_squared_error(y, y_hat)

def cross_entropy_loss(logits, labels):
    return tf.losses.sigmoid_cross_entropy(labels, logits)

