
import tensorflow as tf 

def conv2d_layer(
    inputs, 
    filters, 
    kernel_size = [4, 4], 
    strides = [2, 2], 
    padding = 'same', 
    activation = None,
    kernel_initializer = tf.truncated_normal_initializer(stddev = 0.02),
    name = None):

    conv_layer = tf.layers.conv2d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def conv2d_transpose_layer(
    inputs,
    filters,
    kernel_size,
    strides,
    padding = 'same',
    activation = None,
    kernel_initializer = tf.truncated_normal_initializer(stddev = 0.02),
    name = None):

    deconv_layer = tf.layers.conv2d_transpose(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return deconv_layer

    
def instance_norm_layer(
    inputs, 
    epsilon = 1e-06, 
    activation_fn = None, 
    name = None):

    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs = inputs,
        epsilon = epsilon,
        activation_fn = activation_fn)

    return instance_norm_layer


def residual_block(
    inputs, 
    filters, 
    kernel_size = [3, 3], 
    strides = [1, 1],
    name_prefix = 'residule_block_'):

    p1 = (kernel_size[0] - 1) // 2
    p2 = (kernel_size[1] - 1) // 2 

    paddings = [[0, 0], [p1, p1], [p2, p2], [0, 0]]

    h0_pad = tf.pad(tensor = inputs, paddings = paddings, mode = 'REFLECT', name = 'pad0')
    h1 = conv2d_layer(inputs = h0_pad, filters = filters, kernel_size = kernel_size, strides = strides, padding = 'valid', activation = None, name = name_prefix + 'conv1')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = tf.nn.relu, name = name_prefix + 'norm1')
    h1_pad = tf.pad(tensor = h1_norm, paddings = paddings, mode = 'REFLECT', name = 'pad1')
    h2 = conv2d_layer(inputs = h1_pad, filters = filters, kernel_size = kernel_size, strides = strides, padding = 'valid', activation = None, name = name_prefix + 'conv2')
    h2_norm = instance_norm_layer(inputs = h2, activation_fn = None, name = name_prefix + 'norm2')

    return inputs + h2_norm


def discriminator(inputs, num_filters = 64, reuse = False, scope_name = 'discriminator'):

    with tf.variable_scope(scope_name) as scope:

        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h0 = conv2d_layer(inputs = inputs, filters = num_filters, activation = tf.nn.leaky_relu, name = 'h0_conv')
        h1 = conv2d_layer(inputs = h0, filters = num_filters * 2, activation = None, name = 'h1_conv')
        h1_norm = instance_norm_layer(inputs = h1, activation_fn = tf.nn.leaky_relu, name = 'h1_norm')
        h2 = conv2d_layer(inputs = h1_norm, filters = num_filters * 4, activation = None, name = 'h2_conv')
        h2_norm = instance_norm_layer(inputs = h2, activation_fn = tf.nn.leaky_relu, name = 'h2_norm')
        h3 = conv2d_layer(inputs = h2_norm, filters = num_filters * 8, strides = [1, 1], activation = None, name = 'h3_conv')
        h3_norm = instance_norm_layer(inputs = h3, activation_fn = tf.nn.leaky_relu, name = 'h3_norm')
        h4 = conv2d_layer(inputs = h3_norm, filters = 1, strides = [1, 1], activation = None, name = 'h4_conv')

        return h4


def generator_resnet(inputs, num_filters = 64, output_channels = 3, reuse = False, scope_name = 'generator_resnet'):

    with tf.variable_scope(scope_name) as scope:

        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        #output_channels = inputs.shape[-1]

        # Check tf.pad using 'REFLECT' mode
        # https://www.tensorflow.org/api_docs/python/tf/pad
        c0 = tf.pad(tensor = inputs, paddings = [[0, 0], [3, 3], [3, 3], [0, 0]], mode = 'REFLECT', name = 'c0_pad')

        c1 = conv2d_layer(inputs = c0, filters = num_filters, kernel_size = [7, 7], strides = [1, 1], padding = 'valid', activation = None, name = 'c1_conv')

        c1_norm = instance_norm_layer(inputs = c1, activation_fn = tf.nn.relu, name = 'c1_norm')

        c2 = conv2d_layer(inputs = c1_norm, filters = num_filters * 2, kernel_size = [3, 3], strides = [2, 2], activation = None, name = 'c2_conv')
        c2_norm = instance_norm_layer(inputs = c2, activation_fn = tf.nn.relu, name = 'c2_norm')
        c3 = conv2d_layer(inputs = c2_norm, filters = num_filters * 4, kernel_size = [3, 3], strides = [2, 2], activation = None, name = 'c3_conv')
        c3_norm = instance_norm_layer(inputs = c3, activation_fn = tf.nn.relu, name = 'c3_norm')


        r1 = residual_block(inputs = c3_norm, filters = num_filters * 4, name_prefix = 'residual1_')
        r2 = residual_block(inputs = r1, filters = num_filters * 4, name_prefix = 'residual2_')
        r3 = residual_block(inputs = r2, filters = num_filters * 4, name_prefix = 'residual3_')
        r4 = residual_block(inputs = r3, filters = num_filters * 4, name_prefix = 'residual4_')
        r5 = residual_block(inputs = r4, filters = num_filters * 4, name_prefix = 'residual5_')
        r6 = residual_block(inputs = r5, filters = num_filters * 4, name_prefix = 'residual6_')
        r7 = residual_block(inputs = r6, filters = num_filters * 4, name_prefix = 'residual7_')
        r8 = residual_block(inputs = r7, filters = num_filters * 4, name_prefix = 'residual8_')
        r9 = residual_block(inputs = r8, filters = num_filters * 4, name_prefix = 'residual9_')

        d1 = conv2d_transpose_layer(inputs = r9, filters = num_filters * 2, kernel_size = [3, 3], strides = [2, 2], name = 'd1_deconv')
        d1_norm = instance_norm_layer(inputs = d1, activation_fn = tf.nn.relu, name = 'd1_norm')
        d2 = conv2d_transpose_layer(inputs = d1_norm, filters = num_filters, kernel_size = [3, 3], strides = [2, 2], name = 'd2_deconv')
        d2_norm = instance_norm_layer(inputs = d2, activation_fn = tf.nn.relu, name = 'd2_norm')
        d2_pad = tf.pad(tensor = d2_norm, paddings = [[0, 0], [3, 3], [3, 3], [0, 0]], mode = 'REFLECT', name = 'd2_pad')
        d3 = conv2d_layer(inputs = d2_pad, filters = output_channels, kernel_size = [7, 7], strides = [1, 1], padding = 'valid', activation = tf.nn.tanh, name = 'd3_conv')


        return d3





