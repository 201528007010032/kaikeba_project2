# -*- coding:utf-8 -*-
# Created by XiangYunwu at 20/03/03

import argparse
import os
import time


def get_params():
    parser = argparse.ArgumentParser(description='This is the transformer train project.')

    parser.add_argument('--model', default='transformer')

    parser.add_argument('--maximum_position_encoding', default=300, type=int,
                        help='Padding size of sentences.(default=128)')
    parser.add_argument('--num_heads', default=8, type=int, help='Word embedding size.(default=512)')
    parser.add_argument('--dff', default=2048, type=int, help='Word embedding size.(default=512)')
    parser.add_argument('--buffer_size', default=20000, type=int, help='Word embedding size.(default=512)')
    parser.add_argument('--num_layers', default=6, help='Convolution kernel sizes.(default=3,4,5)')
    parser.add_argument('--d_model', default=512, type=int, help='Number of each convolution kernel.(default=128)')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='Dropout rate in softmax layer.(default=0.5)')
    parser.add_argument('--output_dim', default=95, type=int, help='Number of target classes.(default=18)')
    parser.add_argument('--regularizers_lambda', default=0.01, type=float,
                        help='L2 regulation parameter.(default=0.01)')
    parser.add_argument('--batch_size', default=64, type=int, help='Mini-Batch size.(default=64)')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Learning rate.(default=0.005)')
    parser.add_argument('-input_vocab_size', default=50000, type=int, help='Limit vocab size.(default=50000)')
    parser.add_argument('-target_vocab_size', default=50000, type=int, help='Limit vocab size.(default=50000)')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs.(default=10)')
    parser.add_argument('--fraction_validation', default=0.05, type=float,
                        help='The fraction of validation.(default=0.05)')
    parser.add_argument('--results_dir', default='results/transformer', type=str,
                        help='The results dir including log, model, vocabulary and some images.(default=./results/)')

    parser.add_argument('--data_path', default='data/baidu_95.csv', type=str, help='data path')
    parser.add_argument('--vocab_save_dir', default='data/', type=str, help='data path')

    parser.add_argument('--workers', default=32, type=int, help='use worker count')

    args = parser.parse_args()
    params = vars(args)

    if not os.path.exists(params['results_dir']):
        os.mkdir(params['results_dir'])
    timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
    if not os.path.exists(os.path.join(params['results_dir'], timestamp)):
        os.mkdir(os.path.join(params['results_dir'], timestamp))

    params['save_path'] = os.path.join(params['results_dir'], timestamp)

    return params
