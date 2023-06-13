import tensorflow as tf

def pad_tensor(tensor, target_dim):
    last_dim = tf.shape(tensor)[-1]
    padding = target_dim - last_dim
    padding_shape = tf.concat([tf.shape(tensor)[:-1], [padding]], axis=0)
    padded_tensor = tf.concat([tensor, tf.zeros(padding_shape)], axis=-1)
    return padded_tensor

def l1_loss(y, y_hat):
    last_dim_y = y.get_shape().as_list()[-1]
    last_dim_y_hat = y_hat.get_shape().as_list()[-1]

    if last_dim_y is None or last_dim_y_hat is None:
        # Handle case where shape is unknown
        padded_y = y
        padded_y_hat = y_hat
    else:
        # Pad tensors to match the larger dimension
        if last_dim_y < last_dim_y_hat:
            padded_y = pad_tensor(y, last_dim_y_hat)
            padded_y_hat = y_hat
        elif last_dim_y_hat < last_dim_y:
            padded_y = y
            padded_y_hat = pad_tensor(y_hat, last_dim_y)
        else:
            padded_y = y
            padded_y_hat = y_hat

    loss = tf.reduce_mean(tf.abs(tf.subtract(padded_y, padded_y_hat)))
    return loss

def l2_loss(y, y_hat):
    last_dim_y = y.get_shape().as_list()[-1]
    last_dim_y_hat = y_hat.get_shape().as_list()[-1]

    if last_dim_y is None or last_dim_y_hat is None:
        # Handle case where shape is unknown
        padded_y = y
        padded_y_hat = y_hat
    else:
        # Pad tensors to match the larger dimension
        if last_dim_y < last_dim_y_hat:
            padded_y = pad_tensor(y, last_dim_y_hat)
            padded_y_hat = y_hat
        elif last_dim_y_hat < last_dim_y:
            padded_y = y
            padded_y_hat = pad_tensor(y_hat, last_dim_y)
        else:
            padded_y = y
            padded_y_hat = y_hat

    loss = tf.reduce_mean(tf.square(tf.subtract(padded_y ,padded_y_hat)))
    return loss