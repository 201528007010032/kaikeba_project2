# -*- coding:utf-8 -*-
# Created by XiangYunwu at 20/03/03


import os
import tensorflow as tf
from pprint import pprint

from utils.data_helper import data_loader
from utils.params import get_params
from utils.metrics import micro_f1, macro_f1

from model.TextCNN.TextCNN import TextCNN
import model.TextCNN.TextCNN_params as TextCNN_params

from model.FastText.FastText import FastText
import model.FastText.FastText_params as FastText_params
from model.FastText.FastText import add_ngram_features

from model.transformer.model import Transformer
import model.transformer.transformer_params as transformer_params
from model.transformer.train_helper import train_model


def train(x_train, x_test, y_train, y_test, params, model_params):
    model = build_model(params, model_params)

    print('Train...')
    if params['model'] == 'TextCNN' or params['model'] == 'FastText':
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_micro_f1', patience=10, mode='max')
        history = model.fit(x_train, y_train,
                            batch_size=model_params['batch_size'],
                            epochs=model_params['epochs'],
                            workers=model_params['workers'],
                            use_multiprocessing=True,
                            callbacks=[early_stopping],
                            validation_data=(x_test, y_test))

        print("\nSaving model...")
        tf.keras.models.save_model(model, model_params['save_path'])
        pprint(history.history)
    elif params['model'] == 'transformer':
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # 将数据集缓存到内存中以加快读取速度。
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(model_params['buffer_size'], reshuffle_each_iteration=True).batch(
            model_params['batch_size'], drop_remainder=True)

        test_dataset = test_dataset.batch(params['batch_size'])
        # 流水线技术 重叠训练的预处理和模型训练步骤。当加速器正在执行训练步骤 N 时，CPU 开始准备步骤 N + 1 的数据。这样做可以将步骤时间减少到模型训练与抽取转换数据二者所需的最大时间（而不是二者时间总和）。
        # 没有流水线技术，CPU 和 GPU/TPU 大部分时间将处于闲置状态:
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        train_model(model, train_dataset, test_dataset, model_params)
    else:
        pass


def build_model(params, model_params):
    if params['model'] == 'TextCNN':
        model = TextCNN(max_sequence_length=model_params['padding_size'],
                        max_token_num=model_params['vocab_size'],
                        embedding_dim=model_params['embedding_dim'],
                        output_dim=model_params['num_classes'],
                        kernel_sizes=model_params['filter_sizes'],
                        num_filters=model_params['num_filters'])
        model.compile(tf.optimizers.Adam(learning_rate=model_params['learning_rate']),
                      loss='binary_crossentropy',
                      metrics=[micro_f1, macro_f1])
        model.summary()
    elif params['model'] == 'FastText':
        fasttext = FastText(model_params['maxlen'],
                            model_params['vocab_size'],
                            model_params['embedding_dim'],
                            model_params['num_classes'])
        model = fasttext.get_model()
        model.compile(tf.optimizers.Adam(learning_rate=model_params['learning_rate']),
                      loss='binary_crossentropy',
                      metrics=[micro_f1, macro_f1])
        model.summary()
    elif params['model'] == 'transformer':
        model = Transformer(num_layers=model_params['num_layers'],
                            d_model=model_params['d_model'],
                            num_heads=model_params['num_heads'],
                            dff=model_params['dff'],
                            input_vocab_size=model_params['input_vocab_size'],
                            output_dim=model_params['output_dim'],
                            maximum_position_encoding=model_params['maximum_position_encoding'])
    else:
        pass
    return model


if __name__ == '__main__':
    params = get_params()
    if params['model'] == 'TextCNN':
        model_params = TextCNN_params.get_params()
    elif params['model'] == 'FastText':
        model_params = FastText_params.get_params()
    elif params['model'] == 'transformer':
        model_params = transformer_params.get_params()
    else:
        pass

    x_train, x_test, y_train, y_test, vocab, mlb = data_loader(params, is_rebuild_dataset=False)

    if params['model'] == 'FastText':
        x_train, x_test, vocab_size = add_ngram_features(model_params['ngram_range'], x_train, x_test,
                                                         model_params['vocab_size'])
        model_params['vocab_size'] = vocab_size

    train(x_train, x_test, y_train, y_test, params, model_params)
