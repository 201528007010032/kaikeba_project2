# -*- coding:utf-8 -*-
# Created by XiangYunwu at 20/03/01

import os
import pathlib

root = pathlib.Path(os.path.abspath(__file__)).parent.parent


stopwords_path = os.path.join(root, 'data/stopwords/哈工大停用词表.txt')

# source data
data_path = os.path.join(root, 'data/baidu_95.csv')
data_label_path = os.path.join(root, 'data/label_baidu_95.csv')

# train and test data
train_x_path = os.path.join(root, 'data', 'x_train.npy')
test_x_path = os.path.join(root, 'data', 'x_test.npy')
train_y_path = os.path.join(root, 'data', 'y_train.npy')
test_y_path = os.path.join(root, 'data', 'y_test.npy')

vocab_save_path = os.path.join(root, 'data/vocab.txt')
