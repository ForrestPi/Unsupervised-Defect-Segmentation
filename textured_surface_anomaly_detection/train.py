# !/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import sys
import argparse

import network
import provider
import tf_utils
import math

BASE_DIR = os.path.dirname(__file__)
DATA_ROOT = os.path.join(BASE_DIR, '../data/Class1/')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--base_learning_rate', type=float, default=0.001)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--max_epoch_seg', type=int, default=25)
parser.add_argument('--max_epoch_cls', type=int, default=10)
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--decay_step', type=int, default=1000)
parser.add_argument('--decay_rate', type=float, default=0.7)
FLAGS = parser.parse_args()

GPU_IDX = FLAGS.gpu
BATCH_SIZE = FLAGS.batch_size
BASE_LEARNING_RATE = FLAGS.base_learning_rate
MAX_EPOCH = FLAGS.max_epoch
MAX_EPOCH_seg = FLAGS.max_epoch_seg
MAX_EPOCH_cls = FLAGS.max_epoch_cls
OPTIMIZER = FLAGS.optimizer
IMAGE_SIZE = FLAGS.image_size
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

LOG_DIR = os.path.join(BASE_DIR, '../log/')
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def log_string(out_str): # 把每次print的东西都写入log
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(global_step):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE, global_step*BATCH_SIZE, DECAY_STEP,
        DECAY_RATE, staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate


def get_bn_decay(global_step):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY, global_step*BATCH_SIZE, BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_STEP, staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1-bn_momentum)
    return bn_decay


def train():
    with tf.device('/gpu:'+str(GPU_IDX)):
        # placeholder的作用是什么，我们现在需要用到一些参数构建graph，但这些参数要到后面才能给
        # 而且不同的阶段这些参数不同，因此不能直接将它们给定，先用placeholder占个位
        # 在后面向graph喂数据的时候再把这些数据具体输进去
        img_pl, label_seg_pl, label_cls_pl = \
            network.placeholder_inputs(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        print(is_training_pl)

        global_step = tf.Variable(0)
        bn_decay = get_bn_decay(global_step)

        seg_out, seg_out_former = network.seg_model(img_pl, is_training_pl, bn_decay)
        seg_pred, loss_seg = network.get_loss(label_seg_pl, seg_out, 'seg')

        cls_out = network.cls_model(seg_out, seg_out_former, is_training_pl, bn_decay)
        cls_pred, loss_cls = network.get_loss(label_cls_pl, cls_out, 'cls')

        learning_rate = get_learning_rate(global_step)
        tf.summary.scalar('learning_rate', learning_rate)

        if OPTIMIZER == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum='MOMENTUM')
        elif OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)

        train_op_seg = optimizer.minimize(loss_seg, global_step=global_step)
        train_op_cls = optimizer.minimize(loss_cls, global_step=global_step)
        # global_step: Optional 'Variable' to increment by one after the
        # variables have been updated every time
        # 所有数据集训练完一次，称为一个epoch。
        # 在一个epoch内，每训练一个batchsize，参数更新一次，称为一个iteration/step，
        # 累计在所有epoch的训练过程中，iteration的数目成为global_step

        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    init = tf.global_variables_initializer()

    sess.run(init)

    ops_seg = {'stage': 'seg',
               'img_pl': img_pl,
               'label_pl': label_seg_pl,
               'is_training_pl': is_training_pl,
               'pred': seg_pred,
               'loss': loss_seg,
               'train_op': train_op_seg,
               'step': global_step}

    ops_cls = {'stage': 'cls',
               'img_pl': img_pl,
               'label_pl': label_cls_pl,
               'is_training_pl': is_training_pl,
               'pred': cls_pred,
               'loss': loss_cls,
               'train_op': train_op_cls,
               'step': global_step}

    for e in range(MAX_EPOCH):
        log_string('----- EPOCH %03d -----'% e)
        for epoch in range(MAX_EPOCH_seg):
            log_string('----- SEG EPOCH %03d -----'% epoch)
            train_one_epoch(sess, ops_seg)
            eval_one_epoch(sess, ops_seg)

        for epoch in range(MAX_EPOCH_cls):
            log_string('----- CLS EPOCH %03d -----' % epoch)
            train_one_epoch(sess, ops_cls)
            eval_one_epoch(sess, ops_cls)

        if e % 1 == 0:
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops):
    is_training = True
    img, seg_label, cls_label = provider.LOAD_DATA(DATA_ROOT + 'Train/')
    img, seg_label, cls_label = provider.shuffle_data(img, seg_label, cls_label)
    label = {'seg': seg_label, 'cls': cls_label}
    img_num = img.shape[0]
    batch_num = img_num / BATCH_SIZE

    loss_sum = 0

    for batch_idx in range(batch_num):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        current_imgs = img[start_idx:end_idx]
        current_label = label[ops['stage']][start_idx:end_idx]

        feed_dict = {ops['img_pl']: current_imgs,
                     ops['label_pl']: current_label,
                     ops['is_training_pl']: is_training}

        global_step, pred, loss, _ = sess.run([ops['step'], ops['pred'], ops['loss'], ops['train_op']],
                                               feed_dict=feed_dict)

        log_string('global_step: %d; iter: %d; loss: %f' % (global_step, batch_idx+1, loss))
        loss_sum += loss

    log_string('train mean loss: %f' % (loss_sum / float(batch_num)))
    # print('accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops):
    is_training = False
    img, seg_label, cls_label = provider.LOAD_DATA(DATA_ROOT + 'Test/')
    label = {'seg': seg_label, 'cls': cls_label}
    img_num = len(img)
    batch_num = img_num / BATCH_SIZE

    loss_sum = 0

    for batch_idx in range(batch_num):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        current_imgs = img[start_idx:end_idx]
        current_label = label[ops['stage']][start_idx:end_idx]

        feed_dict = {ops['img_pl']: current_imgs,
                     ops['label_pl']: current_label,
                     ops['is_training_pl']: is_training}

        global_step, pred, loss = sess.run([ops['step'], ops['pred'], ops['loss']],
                                            feed_dict=feed_dict)

        loss_sum += loss

    log_string('eval mean loss: %f' % (loss_sum / float(batch_num)))
    # print('accuracy: %f' % (total_correct / float(total_seen)))

if __name__ == "__main__":
    train()
    LOG_FOUT.close()