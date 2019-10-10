# !/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tf_utils


def placeholder_inputs(batch_size, img_w, img_h):
    input_img_pl = tf.placeholder(tf.float32, shape=(batch_size, img_w, img_h))
    label_seg_pl = tf.placeholder(tf.int32, shape=(batch_size, img_w, img_h))
    label_cls_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return input_img_pl, label_seg_pl, label_cls_pl



def cls_model(seg_out, seg_out_former, is_training, bn_decay):
    out_1 = tf_utils.conv2d(seg_out_former, kernel_shape=[1,1], strides=1, channel=32,
                            activation_fn=tf.nn.relu, scope='conv11')

    out_2 = tf_utils.maxpool2d(seg_out, kernel_shape=[128,128], strides=1, padding='VALID')
    out_3 = tf_utils.avgpool2d(seg_out, kernel_shape=[128,128], strides=1, padding='VALID')
    out_4 = tf_utils.maxpool2d(out_1, kernel_shape=[128, 128], strides=1, padding='VALID')
    out_5 = tf_utils.avgpool2d(out_1, kernel_shape=[128, 128], strides=1, padding='VALID')

    out_6 = tf.concat([out_2, out_3, out_4, out_5], -1)

    out_7 = tf_utils.conv2d(out_6, kernel_shape=[1,1], strides=1, channel=1,
                            bn=True, bn_decay=bn_decay, is_training=is_training,
                            activation_fn=tf.nn.sigmoid, scope='conv12')

    return out_7


def seg_model(input, is_training, bn_decay):
    extend_image = tf.expand_dims(input, axis=-1)

    out_1 = tf_utils.conv2d(extend_image, kernel_shape=[11,11], strides=2, channel=32,
                            bn=True, bn_decay=bn_decay, is_training=is_training,
                            activation_fn=tf.nn.relu, scope='conv1')
    out_2 = tf_utils.conv2d(out_1, kernel_shape=[11,11], strides=1, channel=32,
                            bn=True, bn_decay=bn_decay, is_training=is_training,
                            activation_fn=tf.nn.relu, scope='conv2')
    out_3 = tf_utils.conv2d(out_2, kernel_shape=[11,11], strides=1, channel=32,
                            bn=True, bn_decay=bn_decay, is_training=is_training,
                            activation_fn=tf.nn.relu, scope='conv3')

    out_4 = tf_utils.conv2d(out_3, kernel_shape=[7, 7], strides=2, channel=64,
                            bn=True, bn_decay=bn_decay, is_training=is_training,
                            activation_fn=tf.nn.relu, scope='conv4')
    out_5 = tf_utils.conv2d(out_4, kernel_shape=[7, 7], strides=1, channel=64,
                            bn=True, bn_decay=bn_decay, is_training=is_training,
                            activation_fn=tf.nn.relu, scope='conv5')
    out_6 = tf_utils.conv2d(out_5, kernel_shape=[7, 7], strides=1, channel=64,
                            bn=True, bn_decay=bn_decay, is_training=is_training,
                            activation_fn=tf.nn.relu, scope='conv6')

    out_7 = tf_utils.conv2d(out_6, kernel_shape=[3, 3], strides=1, channel=128,
                            bn=True, bn_decay=bn_decay, is_training=is_training,
                            activation_fn=tf.nn.relu, scope='conv7')
    out_8 = tf_utils.conv2d(out_7, kernel_shape=[3, 3], strides=1, channel=128,
                            bn=True, bn_decay=bn_decay, is_training=is_training,
                            activation_fn=tf.nn.relu, scope='conv8')
    out_9 = tf_utils.conv2d(out_8, kernel_shape=[3, 3], strides=1, channel=128,
                            bn=True, bn_decay=bn_decay, is_training=is_training,
                            activation_fn=tf.nn.relu, scope='conv9')

    seg_layer = tf_utils.conv2d(out_9, kernel_shape=[1, 1], strides=1, channel=1,
                                bn=True, bn_decay=bn_decay, is_training=is_training,
                                activation_fn=tf.nn.relu, scope='conv10')

    return seg_layer, out_9


def get_loss(label, model_out, stage):
    label = tf.to_float(label)
    if stage == 'seg':
        # label or loss: shape = [batch_size, pixel_w, pixel_h]

        # shape = model_out.get_shape()
        # # tf.get_variable函数，变量名称name是一个必填的参数，它会根据变量名称去创建或者获取变量
        # linear_W = tf.get_variable(name='W', shape=shape)
        # linear_b = tf.get_variable(name='b', shape=shape)
        # pred = tf.multiply(model_out, linear_W) + linear_b
        # # tf.multiply对应位置点乘，tf.matmul矩阵数学乘法

        pred = tf_utils.unpool(model_out)
        pred = tf_utils.unpool(pred)
        pred = tf.squeeze(pred, -1)

        loss = tf.reduce_mean(tf.square(label - pred))
    else:
        # stage == cls
        # label or loss: shape = [batch_size], 0 or 1
        pred = model_out
        loss = - tf.reduce_mean(label*tf.log(pred) + (1-label)*tf.log(1-pred))
    return pred, loss