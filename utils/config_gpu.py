# -*- coding:utf-8 -*-
# Created by XiangYunwu at 20/03/09

import tensorflow as tf
import pynvml


def config_gpu():
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if gpus:
        try:
            tmp = 0
            m = 1
            pynvml.nvmlInit()
            for i in range(8):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 这里的0是GPU id
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if meminfo.used / meminfo.total < m:
                    tmp = i
                    m = meminfo.used / meminfo.total
            tf.config.experimental.set_visible_devices(devices=gpus[tmp], device_type='GPU')
            print('Set Visible Devices:', tmp)
        except RuntimeError as e:
            print(e)


def config_gpu_growth():
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
