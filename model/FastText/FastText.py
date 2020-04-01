# -*- coding:utf-8 -*-
# Created by XiangYunwu at 20/03/03


import tensorflow as tf
import numpy as np


class FastText(object):
    def __init__(self, maxlen, max_features, embedding_dims, class_num, last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        inputs = tf.keras.Input((self.maxlen,), name='inputs')
        embedding = tf.keras.layers.Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(inputs)
        x = tf.keras.layers.GlobalAveragePooling1D()(embedding)
        output = tf.keras.layers.Dense(self.class_num, activation=self.last_activation)(x)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    # >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    # >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return np.array(new_sequences)


def add_ngram_features(ngram_range, x_train, x_test, max_features):
    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        # x_train = add_ngram(x_train.tolist(), token_indice, ngram_range)
        # x_test = add_ngram(x_test.tolist(), token_indice, ngram_range)
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
    return x_train, x_test, max_features
