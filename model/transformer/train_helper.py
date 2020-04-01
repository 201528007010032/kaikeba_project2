# -*- coding:utf-8 -*-
# Created by XiangYunwu at 20/03/03

import tensorflow as tf
import time

from model.transformer.padding_mask import create_padding_mask
from model.transformer.layers import CustomSchedule
from utils.metrics import micro_f1, macro_f1


def train_model(transformer, train_dataset, test_dataset, params):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')

    learning_rate = CustomSchedule(params['d_model'])
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def train_step(inp, tar):
        enc_padding_mask = create_padding_mask(inp)

        with tf.GradientTape() as tape:
            predictions = transformer(inp, True, enc_padding_mask=enc_padding_mask)
            loss = loss_function(tar, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar, predictions)

        mi_f1 = micro_f1(tar, predictions)
        ma_f1 = macro_f1(tar, predictions)
        return mi_f1, ma_f1

    def predict(inp, tar, enc_padding_mask):
        predictions = transformer(inp, False, enc_padding_mask=enc_padding_mask)
        mi_f1 = micro_f1(tar, predictions)
        ma_f1 = macro_f1(tar, predictions)
        return mi_f1, ma_f1

    for epoch in range(params['epochs']):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            mic_f1, mac_f1 = train_step(inp, tar)

            if batch % 10 == 0:
                test_input, test_target = next(iter(test_dataset))
                enc_padding_mask = create_padding_mask(test_input)
                val_mic_f1, val_mac_f1 = predict(test_input, test_target, enc_padding_mask)

                print(
                    'Epoch {} Batch {} Loss {:.4f} micro_f1 {:.4f} macro_f1 {:.4f} val_micro_f1 {:.4f} val_macro_f1 {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), mic_f1, mac_f1, val_mic_f1, val_mac_f1))

        # if (epoch + 1) % 5 == 0:
        #     # ckpt_save_path = ckpt_manager.save()
        #     # print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
        #     #                                                     ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
