# -*- coding:utf-8 -*-
# Created by XiangYunwu at 20/03/03


import logging
import tensorflow as tf


def TextCNN(max_sequence_length,
            max_token_num,
            embedding_dim,
            output_dim,
            kernel_sizes,
            num_filters,
            model_img_path=None,
            embedding_matrix=None):
    x_input = tf.keras.Input(shape=(max_sequence_length,))
    logging.info("x_input.shape: %s" % str(x_input.shape))
    if embedding_matrix is None:
        x_emb = tf.keras.layers.Embedding(input_dim=max_token_num,
                                          output_dim=embedding_dim,
                                          input_length=max_sequence_length)(x_input)
    else:
        x_emb = tf.keras.layers.Embedding(input_dim=max_token_num,
                                          output_dim=embedding_dim,
                                          input_length=max_sequence_length,
                                          weights=[embedding_matrix],
                                          trainable=True)(x_input)
    logging.info("x_emb.shape: %s" % str(x_emb.shape))
    pool_output = []
    for kernel_size in kernel_sizes:
        c = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, strides=1)(x_emb)
        p = tf.keras.layers.MaxPool1D(pool_size=int(c.shape[1]))(c)
        pool_output.append(p)
        logging.info("kernel_size: %s \t c.shape: %s \t p.shape: %s" % (kernel_size, str(c.shape), str(p.shape)))

    pool_output = tf.keras.layers.concatenate([p for p in pool_output])
    logging.info("pool_output.shape: %s" % str(pool_output.shape))

    x_flatten = tf.keras.layers.Flatten()(pool_output)
    y = tf.keras.layers.Dense(output_dim, activation='sigmoid')(x_flatten)

    logging.info("y.shape: %s \n" % str(y.shape))

    model = tf.keras.Model([x_input], outputs=[y])

    if model_img_path:
        tf.keras.utils.plot_model(model, to_file=model_img_path, show_shapes=True, show_layer_names=False)
    # model.summary()

    return model
