#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: basemodel.py

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    Conv2D, MaxPooling, BatchNorm, BNReLU, GlobalAvgPooling
)

def image_preprocess(image, bgr=True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        image = image * (1.0 / 255)

        mean = [0.485, 0.456, 0.406] # rgb
        std = [0.229, 0.224, 0.225]

        if bgr:
            mean = mean[::-1]
            std = std[::-1]

        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_std

        return image

def get_bn(zero_init=False):
    if zero_init:
        return lambda x, name: BatchNorm('bn', x,
                                         gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name: BatchNorm('bn', x)

            
