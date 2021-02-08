# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: attention.py
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
from tensorflow import keras
from tensorflow.keras import backend as K

# my packages


class Attention(tf.keras.layers.Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head  # 输出的维度


    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(int(input_shape[0][-1]), self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.WK = self.add_weight(name='WK',
                                  shape=(int(input_shape[1][-1]), self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.WV = self.add_weight(name='WV',
                                  shape=(int(input_shape[2][-1]), self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)

        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):

        if seq_len is None:
            return inputs
        else:
            mask = tf.keras.backend.one_hot(seq_len[:, 0], tf.keras.backend.shape(inputs)[1])
            mask = 1 - tf.keras.backend.cumsum(mask, 1)

            for _ in range(len(inputs.shape) - 2):
                mask = tf.keras.backend.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):

        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x

        print("Q_seq", Q_seq)
        Q_seq = tf.keras.backend.dot(Q_seq, self.WQ)
        Q_seq = tf.keras.backend.reshape(Q_seq, (-1, tf.keras.backend.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = tf.keras.backend.permute_dimensions(Q_seq, (0, 2, 1, 3))

        K_seq = tf.keras.backend.dot(K_seq, self.WK)
        K_seq = tf.keras.backend.reshape(K_seq,
                                     (-1, tf.keras.backend.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = tf.keras.backend.permute_dimensions(K_seq, (0, 2, 1, 3))

        V_seq = tf.keras.backend.dot(V_seq, self.WV)
        V_seq = tf.keras.backend.reshape(V_seq,
                                         (-1, tf.keras.backend.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = tf.keras.backend.permute_dimensions(V_seq, (0, 2, 1, 3))

        # 计算内积，然后mask，然后softmax
        # A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5#attention_11/Shape_12:0", shape=(5,)
        A = tf.matmul(Q_seq, K_seq, transpose_b=True) / self.size_per_head ** 0.5  # tf2.1用这个

        ########上句报错
        ########ValueError: Dimension must be 5 but is 4 for 'attention_11/transpose_7'
        #####在TF1中，A形状为shape=(4,),到了TF2中，A形状变成了(5,)
        A = tf.keras.backend.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = tf.keras.backend.permute_dimensions(A, (0, 3, 2, 1))
        A = tf.keras.backend.softmax(A)
        # 输出并mask
        # O_seq = K.batch_dot(A, V_seq, axes=[3,2])#tf2.0用这个
        O_seq = tf.matmul(A, V_seq)  # tf2.1用这个
        O_seq = tf.keras.backend.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = tf.keras.backend.reshape(O_seq, (-1, tf.keras.backend.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')

        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


class TargetedDropout(keras.layers.Layer):  # Targeted Dropout层

    def __init__(self, drop_rate, target_rate, **kwargs):
        super(TargetedDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.drop_rate = drop_rate
        self.target_rate = target_rate

    def get_config(self):
        config = {
            'drop_rate': self.drop_rate,
            'target_rate': self.target_rate,
        }
        base_config = super(TargetedDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _compute_target_mask(self, inputs, mask=None):
        input_shape = K.shape(inputs)
        input_type = K.dtype(inputs)

        mask_threshold = K.constant(1e8, dtype=input_type)

        channel_num = int(inputs.shape[-1])
        channel_dim = K.prod(input_shape[:-1])
        masked_inputs = inputs
        if mask is not None:
            masked_inputs = K.switch(
                K.cast(mask, K.floatx()) > 0.5,
                masked_inputs,
                K.ones_like(masked_inputs, dtype=input_type) * mask_threshold
            )
        norm = K.abs(masked_inputs)
        channeled_norm = K.transpose(K.reshape(norm, (channel_dim, channel_num)))
        weight_num = K.sum(
            K.reshape(K.cast(masked_inputs < mask_threshold, K.floatx()), (channel_dim, channel_num)),
            axis=0,
        )
        indices = K.stack(
            [
                K.arange(channel_num, dtype='int32'),
                K.cast(self.target_rate * weight_num, dtype='int32') - 1,
            ],
            axis=-1,
        )
        threshold = -tf.gather_nd(tf.nn.top_k(-channeled_norm, k=K.max(indices[:, 1]) + 1).values, indices)

        threshold = K.reshape(tf.tile(threshold, [channel_dim]), input_shape)
        target_mask = K.switch(
            norm <= threshold,
            K.ones_like(inputs, dtype=K.floatx()),
            K.zeros_like(inputs, dtype=K.floatx()),
        )
        return target_mask

    def call(self, inputs, mask=None, training=None):
        target_mask = self._compute_target_mask(inputs, mask=mask)

        def dropped_mask():
            drop_mask = K.switch(
                K.random_uniform(K.shape(inputs)) < self.drop_rate,
                K.ones_like(inputs, K.floatx()),
                K.zeros_like(inputs, K.floatx()),
            )
            return target_mask * drop_mask

        def pruned_mask():
            return target_mask

        mask = K.in_train_phase(dropped_mask, pruned_mask, training=training)
        outputs = K.switch(
            mask > 0.5,
            K.zeros_like(inputs, dtype=K.dtype(inputs)),
            inputs,
        )
        return outputs


