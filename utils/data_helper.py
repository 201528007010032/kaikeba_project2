# -*- coding:utf-8 -*-
# Created by XiangYunwu at 20/03/01

import os
import argparse
import sys

sys.path.append('..')

import re
import jieba
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer

from utils.config import root
from utils.multi_proc_utils import parallelize
from utils import config


def load_stop_words(stop_word_path):
    """
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    """
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


stop_words = load_stop_words(config.stopwords_path)


def clean_sentence(line):
    s = '题目|答案|知识点|解析|A|B|C|D'
    line = re.sub(
        "[a-zA-Z0-9]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+|" + s, '', line)
    words = jieba.cut(line, cut_all=False)
    return words


def sentence_proc(sentence):
    """
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    """
    # 清除无用词
    words = clean_sentence(sentence)
    # 过滤停用词
    words = [word for word in words if word not in stop_words]
    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)


def proc(df):
    df['content'] = df['content'].apply(sentence_proc)
    return df


def data_loader(params, is_rebuild_dataset=False):
    if os.path.exists(config.train_x_path) and not is_rebuild_dataset:
        x_train = np.load(config.train_x_path)
        x_test = np.load(config.test_x_path)
        y_train = np.load(config.train_y_path)
        y_test = np.load(config.test_y_path)

        with open(config.vocab_save_path, 'r', encoding='utf-8') as f:
            vocab = {}
            for content in f.readlines():
                k, v = content.strip().split('\t')
                vocab[k] = int(v)
        label_df = pd.read_csv(config.data_label_path)
        # 多标签编码
        mlb = MultiLabelBinarizer()
        mlb.fit([label_df['label']])

        return x_train, x_test, y_train, y_test, vocab, mlb

    df = pd.read_csv(config.data_path, header=None).rename(columns={0: 'label', 1: 'content'})
    df = parallelize(df, proc)

    text_preprocesser = tf.keras.preprocessing.text.Tokenizer(num_words=params['vocab_size'], oov_token="<UNK>")
    text_preprocesser.fit_on_texts(df['content'])

    vocab = text_preprocesser.word_index
    with open(config.vocab_save_path, 'w', encoding='utf-8') as f:
        for k, v in vocab.items():
            f.write(f'{k}\t{str(v)}\n')

    x = text_preprocesser.texts_to_sequences(df['content'])
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=params['padding_size'], padding='post', truncating='post')

    # label_df = pd.read_csv(config.data_label_path)

    mlb = MultiLabelBinarizer()
    df['label'] = df['label'].apply(lambda x: x.split())
    mlb.fit(df['label'])
    y = mlb.transform(df['label'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    np.save(config.train_x_path, x_train)
    np.save(config.test_x_path, x_test)
    np.save(config.train_y_path, y_train)
    np.save(config.test_y_path, y_test)

    return x_train, x_test, y_train, y_test, vocab, mlb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the Bert test project.')

    parser.add_argument('-d', '--data_path', default='data/baidu_95.csv', type=str, help='data path')
    parser.add_argument('-v', '--vocab_save_dir', default='data/', type=str, help='data path')
    parser.add_argument('-vocab_size', default=50000, type=int, help='Limit vocab size.(default=50000)')
    parser.add_argument('-p', '--padding_size', default=200, type=int, help='Padding size of sentences.(default=128)')

    params = parser.parse_args()

    vocab_size = 50000
    padding_size = 300
    print('Parameters:', params.__dict__)
    x_train, x_test, y_train, y_test, vocab, mlb = data_loader(vocab_size, padding_size)
    print(x_train)
