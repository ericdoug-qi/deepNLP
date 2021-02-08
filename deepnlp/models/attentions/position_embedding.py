# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: position_embedding.py
   Description : 
   Author : ericdoug
   date：2021/2/8
-------------------------------------------------
   Change Activity:
         2021/2/8: created
-------------------------------------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

# sys packages
import os

# third packages
import tensorflow as tf


# my packages

class Position_Embedding(tf.keras.layers.Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        super(Position_Embedding, self).__init__(**kwargs)
        self.size = size  # 必须为偶数
        self.mode = mode

    def call(self, x):

        if self.size is None or self.mode == 'sum':
            self.size = int(x.shape[-1])

        position_j = 1. / tf.keras.backend.pow(1000., 2 * tf.keras.backend.arange(self.size / 2, dtype='float32')/ self.size)
        position_j = tf.keras.backend.expand_dims(position_j, 0)

        # 按照x的1维度累计求和， 与arange一样， 生成序列， 只不过按照x的实际长度来
        position_i = tf.cumsum(tf.keras.backend.ones_like(x[:, :, 0]), 1) - 1
        position_i = tf.keras.backend.expand_dims(position_i, 2)
        position_ij = tf.keras.backend.dot(position_i, position_j)
        position_ij = tf.keras.backend.concatenate([tf.keras.backend.cos(position_ij), tf.keras.backend.sin(position_ij)], 2)

        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return tf.keras.backend.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):

        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)






