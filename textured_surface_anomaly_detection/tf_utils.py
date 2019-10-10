#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def conv2d(data,
           kernel_shape,
           strides,
           channel,
           activation_fn,
           scope,
           bn = False,
           bn_decay = None,
           is_training=None): # default参数必须位于非default参数之后


    with tf.variable_scope(scope) as sc:
        in_channel = data.get_shape()[-1]
        filter_shape = [kernel_shape[0], kernel_shape[1], in_channel, channel]
        # filter = tf.get_variable(name='filter', shape=filter_shape)

        filter = _variable_with_weight_decay('weights',
                                             shape=filter_shape,
                                             use_xavier=True,
                                             stddev=1e-3,
                                             wd=0.0)

        # filter接收的是一个实实在在的tensor，有内容，不仅仅是一个shape!!
        # 要搞清楚filter/kernel和它们shape的区别!!
        outputs = tf.nn.conv2d(input=data,
                            filter=filter,
                            strides=[1, strides, strides, 1],
                            padding="SAME",
                            use_cudnn_on_gpu=None,
                            data_format=None,
                            name=None)
        # tf.nn.conv2d只有卷积操作（乘法运算）， 没有加bias！！手动加上
        biases = _variable_on_cpu('biases', [channel], tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def maxpool2d(data, kernel_shape, strides, padding):
    return tf.nn.max_pool(value=data,
                          ksize=[1, kernel_shape[0], kernel_shape[1], 1],
                          strides=[1, strides, strides, 1],
                          padding=padding,
                          name=None)


def avgpool2d(data, kernel_shape, strides, padding):
    return tf.nn.avg_pool(value=data,
                          ksize=[1, kernel_shape[0], kernel_shape[1], 1],
                          strides=[1, strides, strides, 1],
                          padding=padding,
                          name=None)


def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        # [-1] + sh[-dim:] 列表list的扩充

        for i in range(dim, 0, -1):
            # out = tf.concat([out, tf.zeros_like(out)], i)
            out = tf.concat([out, out], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out

# value = tf.get_variable(name='value', shape=[3,2,2,4,5], dtype=tf.int32)
# unpool(value=value)


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
      Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

      Args:
          inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
          is_training:   boolean tf.Variable, true indicates training phase
          scope:         string, variable scope
          *********************************************************************************
          * moments_dims:  a list of ints, indicating dimensions for moments calculation  *
          *                如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width]   *
          *                的均值/方差, 注意不要加入 channel 维度                              *
          *********************************************************************************
          bn_decay:      float or float tensor variable, controlling moving average weight
      Return:
          normed:        batch-normalized maps
      """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')

        # 我们使用batch进行每次的更新, 那每个batch的mean/var都会不同, 所以我们可以使用moving average
        # 的方法记录并慢慢改进mean/var的值，尽可能的让mean/var建立在所有数据的基础上；然后将修改提升后的
        # mean/var放入tf.nn.batch_normalization().而且在test阶段, 我们就可以直接调用最后一次修改的
        # mean/var值进行测试, 而不是采用test时的mean/var.
        # 采用加权滑动平均，权重系数以decay衰减。
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        # Operator that maintains moving average variables
        # 那如何确定我们是在train阶段还是在test阶段呢, 想办法传入is_training参数
        # 条件函数tf.cond（true_or_false, true_op, false_op）
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),  # is_training=true
                               lambda: tf.no_op())   # is_training=false

        # Update moving average and return current batch's avg and var
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var
        # 那如何确定我们是在train阶段还是在test阶段呢, 想办法传入is_training参数
        mean, var = tf.cond(is_training,           # is_training值为True/False。
                            mean_var_with_update,  # 若True, 更新mean/var;
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
                                                   # 若False，返回之前mean/var的滑动平均值

        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
        # 最后一步tf.nn.batch_normalization, 在做如下事情, inputs = Wx_plus_b:
        # Wx_plus_b = (Wx_plus_b - mean) / tf.sqrt(var + 1e-3)
        # Wx_plus_b = Wx_plus_b * gamma + beta
    return normed


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
    return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)


























