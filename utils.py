
import os
import random
import numpy as np
import tensorflow as tf

def l1_loss(y, y_hat):
    y_shape = tf.shape(y)
    y_hat_shape = tf.shape(y_hat)
    
    # Reshape tensors to a common shape
    y_reshaped = tf.reshape(y, [-1, tf.reduce_max(y_shape)])
    y_hat_reshaped = tf.reshape(y_hat, [-1, tf.reduce_max(y_hat_shape)])
    
    print(y_reshaped.shape)
    print(y_hat_reshaped.shape)
    # Calculate L1 loss
    loss = tf.reduce_mean(tf.abs(y_reshaped - y_hat_reshaped))
    return loss




def l2_loss(y, y_hat):
    return tf.losses.mean_squared_error(y, y_hat)

def cross_entropy_loss(logits, labels):
    return tf.losses.sigmoid_cross_entropy(labels, logits)

