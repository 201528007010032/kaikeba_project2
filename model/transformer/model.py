# -*- coding:utf-8 -*-
# Created by XiangYunwu at 20/03/01


import tensorflow as tf
from model.transformer.layers import Encoder


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 output_dim, maximum_position_encoding, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, maximum_position_encoding, dropout_rate)

        self.x_flatten = tf.keras.layers.Flatten()

        self.final_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid')

    def __call__(self, inp, training, enc_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        flatten_output = self.x_flatten(enc_output)

        final_output = self.final_layer(flatten_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output


if __name__ == '__main__':
    sample_transformer = Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500,
                                     output_dim=95, maximum_position_encoding=300)

    temp_input = tf.random.uniform((64, 62))
    temp_target = tf.random.uniform((64, 26))

    fn_out = sample_transformer(temp_input, training=True, enc_padding_mask=None)

    print(fn_out.shape)
