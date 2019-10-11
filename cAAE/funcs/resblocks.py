import tensorflow as tf
from tensorlayer.layers import Conv2d, BatchNormLayer, set_name_reuse, InputLayer, DeConv2d


# same image size
def ResBlock(inputs, filter_in, filter_out, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        input_layer = InputLayer(inputs, name='inputs')
        conv1 = Conv2d(input_layer, filter_in, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="conv1")
        conv1 = BatchNormLayer(conv1, act=tf.nn.leaky_relu, is_train=phase_train,
                               gamma_init=gamma_init, name='bn1')
        conv2 = Conv2d(conv1, filter_out, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init, name="conv2")
        conv2 = BatchNormLayer(conv2, act=tf.nn.leaky_relu, is_train=phase_train,
                               gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(input_layer, filter_out, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="conv3")
        conv3 = BatchNormLayer(conv3, act=tf.nn.leaky_relu, is_train=phase_train,
                               gamma_init=gamma_init, name='bn3')

        conv_out = conv2.outputs + conv3.outputs
    return conv_out


# image size /2
def ResBlockDown(inputs, filters, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        input_layer = InputLayer(inputs, name='inputs')
        conv1 = Conv2d(input_layer, filters, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="conv1")
        conv1 = BatchNormLayer(conv1, act=tf.nn.leaky_relu, is_train=phase_train,
                               gamma_init=gamma_init, name='bn1')
        conv2 = Conv2d(conv1, filters * 2, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init, name="conv2")
        conv2 = BatchNormLayer(conv2, act=tf.nn.leaky_relu, is_train=phase_train,
                               gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(input_layer, filters * 2, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="conv3")
        conv3 = BatchNormLayer(conv3, act=tf.nn.leaky_relu, is_train=phase_train,
                               gamma_init=gamma_init, name='bn3')

        conv_out = conv2.outputs + conv3.outputs
    return conv_out


# image size *2
def ResBlockUp(inputs, input_size, batch_size, filters, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        input_layer = InputLayer(inputs, name='inputs')
        conv1 = DeConv2d(input_layer, filters, (3, 3), (input_size * 2, input_size * 2), (2, 2),
                         batch_size=batch_size, act=None, padding='SAME',
                         W_init=w_init, b_init=b_init, name="deconv1")
        conv1 = BatchNormLayer(conv1, act=tf.nn.leaky_relu, is_train=phase_train,
                               gamma_init=gamma_init, name='bn1')
        conv2 = DeConv2d(conv1, filters / 2, (3, 3), (input_size * 2, input_size * 2), (1, 1), act=None, padding='SAME',
                         batch_size=batch_size, W_init=w_init, b_init=b_init, name="deconv2")
        conv2 = BatchNormLayer(conv2, act=tf.nn.leaky_relu, is_train=phase_train,
                               gamma_init=gamma_init, name='bn2')

        conv3 = DeConv2d(input_layer, filters / 2, (3, 3), (input_size * 2, input_size * 2), (2, 2), act=None,
                         padding='SAME',
                         batch_size=batch_size, W_init=w_init, b_init=b_init, name="conv3")
        conv3 = BatchNormLayer(conv3, act=tf.nn.leaky_relu, is_train=phase_train,
                               gamma_init=gamma_init, name='bn3')

        conv_out = conv2.outputs + conv3.outputs
    return conv_out
