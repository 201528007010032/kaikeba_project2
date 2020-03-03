# -*- coding:utf-8 -*-
# Created by XiangYunwu at 20/03/01


import tensorflow as tf


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]    # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    tf.linalg.band_part(
        input,
        num_lower,
        num_upper,
        name=None
    )
    input:输入的张量.
    num_lower:下三角矩阵保留的副对角线数量，从主对角线开始计算，相当于下三角的带宽。取值为负数时，则全部保留。
    num_upper:上三角矩阵保留的副对角线数量，从主对角线开始计算，相当于上三角的带宽。取值为负数时，则全部保留。
    """
    mask = 1 - tf.linalg.band_part(tf.ones(size, size), -1, 0)    # (seq_len, seq_len)
    return mask