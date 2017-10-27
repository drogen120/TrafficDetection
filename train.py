#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train,py

from tensorpack import *
import tensorflow as tf
import argparse
import numpy as np
import os

import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import dataset
from kitti import KITTI
import basemodel

slim = tf.contrib.slim

def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC', is_training = True):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.99},
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format) as sc:

            return sc


class Model(ModelDesc):
    def __init__(self, class_num, max_objects=10):
        super(Model, self).__init__()
        self.class_num = class_num
        self.max_objects = max_objects

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 300, 300, 3), 'input'),
                InputDesc(tf.int32, (None, 3), 'shape'),
                InputDesc(tf.float32, (None, self.max_objects, 4), 'bboxes'),
                InputDesc(tf.float32, (None, self.max_objects, 1), 'labels'),
                InputDesc(tf.float32, (None, self.max_objects, 1), 'alphas'),
                InputDesc(tf.float32, (None, self.max_objects, 1), 'truncated'),
                InputDesc(tf.float32, (None, self.max_objects, 1), 'occluded')
                ]

    def _build_graph(self, inputs):
        image, shape, bboxes, labels, alphas, truncated, occluded = inputs
        is_training = get_current_tower_context().is_training
        raw_ground_trouth = tf.concat([bboxes, labels, alphas, truncated,
                                       occluded], axis=2)

        image = basemodel.image_preprocess(image)
        if is_training:
            tf.summary.image("train_image", image)
        if tf.test.is_gpu_available():
            image = tf.transpose(image, [0, 3, 1, 2])
            data_format = 'NCHW'
        else:
            data_format = 'NHWC'

        end_points = {}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'decay': 0.99},
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME',
                                data_format=data_format) as sc:

                # Original VGG-16 blocks.
                net = slim.repeat(image, 1, slim.conv2d, 16, [3, 3], scope='conv1')
                end_points['block1'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                # Block 2.
                net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], scope='conv2')
                end_points['block2'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                # Block 3.
                net = slim.repeat(net, 3, slim.conv2d, 32, [3, 3], scope='conv3')
                end_points['block3'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                # Block 4.
                net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], scope='conv4')
                end_points['block4'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                # Block 5.
                net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope='conv5')
                end_points['block5'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                net = slim.conv2d(net, 256, [3, 3], scope='conv6')
                end_points['block6'] = net

                self.output = slim.conv2d(net, 5 + self.class_num, [3, 3],
                                          scope = 'output')

        encoded_gt = self.ground_trouth_encode(raw_ground_trouth)
        cost = tf.losses.absolute_difference(encoded_gt,self.output)
        self.cost = cost
        summary.add_moving_summary(cost)

    def _get_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=5000,
            decay_rate = 0.3, staircase=True, name='learning_rate'
        )
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)

    def ground_trouth_encode(self, raw_gt):
        #todo implement
        return tf.ones_like(self.output)

def get_data():
    d1 = KITTI('./kitti_train/', 'train', shuffle = True)
    augmentors = [
        imgaug.RandomCrop(300)
    ]

    data_train = AugmentImageComponent(d1, augmentors)
    data_train = BatchData(data_train, 6)
    # data_train = PrefetchData(data_train, 3, 2)
    return data_train 

def get_config():
    logger.auto_set_dir()
    dataset_train = get_data()
    return TrainConfig(
        model=Model(9),
        dataflow=dataset_train,
        callbacks=[ModelSaver()],
        max_epoch = 100,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU to use')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIVLE_DEVICES'] = args.gpu

    config = get_config()
    SimpleTrainer(config).train()
