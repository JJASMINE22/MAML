# -*- coding: UTF-8 -*-
'''
@Project ：MAML
@File    ：predict.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''

import cv2
import numpy as np
import tensorflow as tf
import config as cfg
from models.maml import MAML


if __name__ == '__main__':

    maml = MAML(input_shape=cfg.input_shape, class_num=cfg.class_num,
                lr_rate=cfg.learning_rate)

    ckpt = tf.train.Checkpoint(maml=maml.model,
                               optimizer=maml.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, cfg.ckpt_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点，加载模型
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    while True:
        try:
            file_path = input('file path: ')
        except FileNotFoundError:
            print('No such file')
        else:
            image = cv2.imread(file_path)
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
            image = np.array(image) / 127.5 - 1
            image = np.clip(image, -1., 1.)
            image = np.expand_dims(image, axis=0)
            pred_logits = maml.model.predict(image)
            pred_label = np.argmax(pred_logits[0], axis=0)
            print('predict label: ', pred_label)