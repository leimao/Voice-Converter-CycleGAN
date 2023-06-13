import tensorflow as tf

def pad_tensor(tensor, target_dim):
    last_dim = tf.shape(tensor)[-1]
    padding = target_dim - last_dim
    padding_shape = tf.concat([tf.shape(tensor)[:-1], [padding]], axis=0)
    padded_tensor = tf.concat([tensor, tf.zeros(padding_shape)], axis=-1)
    return padded_tensor


def l1_loss(y, y_hat):
    shape_y = tf.shape(y)
    shape_y_hat = tf.shape(y_hat)

    last_dim_y = shape_y[-1]
    last_dim_y_hat = shape_y_hat[-1]

    if last_dim_y is None:
        last_dim_y = y.get_shape().as_list()[-1]
    if last_dim_y_hat is None:
        last_dim_y_hat = y_hat.get_shape().as_list()[-1]

    padded_y = tf.cond(last_dim_y < last_dim_y_hat,
                       lambda: pad_tensor(y, last_dim_y_hat),
                       lambda: y)
    padded_y_hat = tf.cond(last_dim_y_hat < last_dim_y,
                           lambda: pad_tensor(y_hat, last_dim_y),
                           lambda: y_hat)

    loss = tf.reduce_mean(tf.abs(padded_y - padded_y_hat))
    return loss

def l2_loss(y, y_hat):
    shape_y = tf.shape(y)
    shape_y_hat = tf.shape(y_hat)

    last_dim_y = shape_y[-1]
    last_dim_y_hat = shape_y_hat[-1]

    if last_dim_y is None:
        last_dim_y = y.get_shape().as_list()[-1]
    if last_dim_y_hat is None:
        last_dim_y_hat = y_hat.get_shape().as_list()[-1]

    padded_y = tf.cond(last_dim_y < last_dim_y_hat,
                       lambda: pad_tensor(y, last_dim_y_hat),
                       lambda: y)
    padded_y_hat = tf.cond(last_dim_y_hat < last_dim_y,
                           lambda: pad_tensor(y_hat, last_dim_y),
                           lambda: y_hat)

    loss = tf.reduce_mean(tf.square(padded_y - padded_y_hat))
    return loss

