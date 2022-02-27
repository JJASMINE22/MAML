# -*- coding: UTF-8 -*-
'''
@Project ：MAML
@File    ：predict_lstm.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import numpy as np
import tensorflow as tf
import config as cfg
import matplotlib.pyplot as plt
from models.maml_lstm import MAML
from _utils import DataGenerator

if __name__ == '__main__':

    gen = DataGenerator(txt_path=cfg.text_path, time_seq=cfg.time_seq,
                        train_ratio=cfg.seq_train_ratio, task_num=cfg.seq_task_num,
                        query_ratio=cfg.seq_query_ratio, support_query_size=cfg.seq_sq_size)

    maml = MAML(input_shape=(cfg.time_seq, cfg.feature_dims),
                feature_dims=cfg.feature_dims,
                lr_rate=cfg.seq_learning_rate)

    if not os.path.exists(cfg.seq_ckpt_path):
        os.makedirs(cfg.seq_ckpt_path)

    ckpt = tf.train.Checkpoint(maml_lstm=maml.model,
                               optimizer=maml.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, cfg.seq_ckpt_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点，加载模型
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    total_src, total_tgt = gen.preprocess()
    last_seq = total_src[-1]
    preds = []
    for i in range(20):
        pred_seq = maml.model.predict(np.expand_dims(last_seq, axis=0))
        last_seq = np.concatenate([last_seq, pred_seq], axis=0)[1:]
        preds.append(pred_seq[0])

    preds = np.array(preds)

    for j in range(preds.shape[-1]):
        plt.subplot(5, 1, j+1)
        plt.plot(preds[:, j], color='r', linewidth=2)

    plt.show()
