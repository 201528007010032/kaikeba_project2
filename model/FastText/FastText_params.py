# -*- coding:utf-8 -*-
# Created by XiangYunwu at 20/03/03

import argparse
import os
import time

from utils.config import root


def get_params():
    parser = argparse.ArgumentParser(description='This is the FastText train project.')

    parser.add_argument('--model', default='FastText')

    parser.add_argument('--maxlen', default=300, type=int, help='Padding size of sentences.(default=128)')
    parser.add_argument('--embedding_dim', default=512, type=int, help='Word embedding size.(default=512)')
    parser.add_argument('--num_classes', default=95, type=int, help='Number of target classes.(default=18)')
    parser.add_argument('--batch_size', default=256, type=int, help='Mini-Batch size.(default=64)')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Learning rate.(default=0.005)')
    parser.add_argument('--vocab_size', default=50000, type=int, help='Limit vocab size.(default=50000)')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs.(default=10)')
    parser.add_argument('--results_dir', default=os.path.join(root, 'results/FastText'), type=str,
                        help='The results dir including log, model, vocabulary and some images.(default=./results/)')
    parser.add_argument('--ngram_range', default=1, type=int, help='ngram range')

    parser.add_argument('--workers', default=32, type=int, help='use worker count')

    args = parser.parse_args()
    params = vars(args)

    if not os.path.exists(params['results_dir']):
        os.mkdirs(params['results_dir'])
    timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
    os.mkdir(os.path.join(params['results_dir'], timestamp))

    params['save_path'] = os.path.join(params['results_dir'], timestamp)

    return params
