
import os
import random
import numpy as np
import tensorflow as tf

def l1_loss(y, y_hat):
    # Checks the shapes of y and y_hat
    y_shape = tf.shape(y)
    y_hat_shape = tf.shape(y_hat)
    
    # Checks if shapes are compatible for broadcasting
    if y_shape != y_hat_shape:
        # Reshape or expand dimensions to match shapes
        if len(y_shape) == len(y_hat_shape):
            # Reshape y to match y_hat_shape
            y = tf.reshape(y, y_hat_shape)
        else:
            # Expand dimensions of y to match y_hat_shape
            y = tf.expand_dims(y, axis=-1)
            y = tf.broadcast_to(y, y_hat_shape)
        
    # Perform subtraction and calculate absolute difference
    loss = tf.reduce_mean(tf.abs(y - y_hat))
    return loss


def l2_loss(y, y_hat):
    return tf.losses.mean_squared_error(y, y_hat)

def cross_entropy_loss(logits, labels):
    return tf.losses.sigmoid_cross_entropy(labels, logits)

