
import os
import random
import numpy as np
import tensorflow as tf

def l1_loss(y, y_hat):
    if y.shape[-1] != y_hat.shape[-1]:
        if y.shape[-1] < y_hat.shape[-1]:
            padding_amount = y_hat.shape[-1] - y.shape[-1]
            pad_width = tf.constant([[0, 0], [0, 0], [0, padding_amount]])
            y = tf.pad(y, pad_width)
        else:
            padding_amount = y.shape[-1] - y_hat.shape[-1]
            pad_width = tf.constant([[0, 0], [0, 0], [0, padding_amount]])
            y_hat = tf.pad(y_hat, pad_width)
    loss = tf.reduce_mean(tf.abs(y - y_hat))
    return loss


def l2_loss(y, y_hat):
    return tf.losses.mean_squared_error(y, y_hat)

def cross_entropy_loss(logits, labels):
    return tf.losses.sigmoid_cross_entropy(labels, logits)

