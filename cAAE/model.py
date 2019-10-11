from tensorlayer.layers import *
from funcs.resblocks import ResBlock, ResBlockDown, ResBlockUp

# The autoencoder network
def encoder(x, z_dim, reuse=False, is_train=True):
    """
    Encode part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    input_dim = x.shape[-1]
    image_size = input_dim
    s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16)
    gf_dim = 16  # Dimension of gen filters in first conv layer. [64]
    ft_size = 3
    with tf.variable_scope("Encoder", reuse=reuse):
        # x,y,z,_ = tf.shape(input_images)
        set_name_reuse(reuse)

        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.01)

        inputs = InputLayer(x, name='e_inputs')
        conv1 = Conv2d(inputs, gf_dim, (ft_size, ft_size), act=tf.nn.leaky_relu(x, 0.2),
                       padding='SAME',W_init=w_init, b_init=b_init, name="e_conv1")
        conv1 = BatchNormLayer(conv1, act=tf.nn.leaky_relu(x, 0.2), is_train=is_train,
                               gamma_init=gamma_init, name='e_bn1')
        # image_size * image_size
        res1 = ResBlockDown(conv1.outputs, gf_dim, "res1", reuse, is_train)

        # s2*s2
        res2 = ResBlockDown(res1, gf_dim * 2, "res2", reuse, is_train)

        # s4*s4
        res3 = ResBlockDown(res2, gf_dim * 4, "res3", reuse, is_train)

        # s8*s8
        res4 = ResBlockDown(res3, gf_dim * 8, "res4", reuse, is_train)

        # s16*s16
        h_flat = tf.reshape(res4, shape=[-1, s16 * s16 * gf_dim * 16])
        h_flat = InputLayer(h_flat, name='e_reshape')
        net_h = DenseLayer(h_flat, n_units=z_dim, act=tf.identity, name="e_dense_mean")
    return net_h.outputs


def decoder(x, reuse=False, is_train=True):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
    """
    input_dim = x.shape[-1]
    image_size = input_dim
    s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16)
    gf_dim = 16  # Dimension of gen filters in first conv layer. [64]
    c_dim = 1  # n_color 3
    ft_size = 3
    batch_size = 16  # 64
    with tf.variable_scope("Decoder", reuse=reuse):
        set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        inputs = InputLayer(x, name='g_inputs')

        # s16*s16
        z_develop = DenseLayer(inputs, s16 * s16 * gf_dim * 16, act=tf.nn.leaky_relu(x, 0.2),
                               name='g_dense_z')
        z_develop = tf.reshape(z_develop.outputs, [-1, s16, s16, gf_dim * 16])
        z_develop = InputLayer(z_develop, name='g_reshape')
        conv1 = Conv2d(z_develop, gf_dim * 8, (ft_size, ft_size), act=tf.nn.leaky_relu(x, 0.2),
                       padding='SAME', W_init=w_init, b_init=b_init, name="g_conv1")

        # s16*s16
        res1 = ResBlockUp(conv1.outputs, s16, batch_size, gf_dim * 8, "gres1", reuse, is_train)

        # s8*s8
        res2 = ResBlockUp(res1, s8, batch_size, gf_dim * 4, "gres2", reuse, is_train)

        # s4*s4
        res3 = ResBlockUp(res2, s4, batch_size, gf_dim * 2, "gres3", reuse, is_train)

        # s2*s2
        res4 = ResBlockUp(res3, s2, batch_size, gf_dim, "gres4", reuse, is_train)

        # image_size*image_size
        res_inputs = InputLayer(res4, name='res_inputs')
        conv2 = Conv2d(res_inputs, c_dim, (ft_size, ft_size), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="g_conv2")
        conv2_std = Conv2d(res_inputs, c_dim, (ft_size, ft_size), act=None, padding='SAME', W_init=w_init,
                           b_init=b_init,
                           name="g_conv2_std")
    return conv2.outputs, conv2_std.outputs


def discriminator(x, reuse=False):
    """
    Discriminator that is used to match the posterior distribution with a given prior distribution.
    :param x: tensor of shape [batch_size, z_dim]
    :param reuse: True -> Reuse the discriminator variables,
                  False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    n_l1=200
    n_l2=400
    w_init = tf.random_normal_initializer(stddev=0.01)
    with tf.variable_scope("Discriminator", reuse=reuse):
        set_name_reuse(reuse)
        net_in = InputLayer(x, name='dc/in')
        net_h0 = DenseLayer(net_in, n_units=n_l1,
                            W_init=w_init,
                            act=tf.nn.leaky_relu(x, 0.2), name='dc/h0/lin')
        net_h1 = DenseLayer(net_h0, n_units=n_l2,
                            W_init=w_init,
                            act=tf.nn.leaky_relu(x, 0.2), name='dc/h1/lin')
        net_h2 = DenseLayer(net_h1, n_units=1,
                            W_init=w_init,
                            act=tf.identity, name='dc/h2/lin')
        logits = net_h2.outputs
        net_h2.outputs = tf.nn.sigmoid(net_h2.outputs)
        return net_h2.outputs, logits
