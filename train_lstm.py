# -*- coding: UTF-8 -*-
'''
@Project ：MAML
@File    ：train_lstm.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''

import tensorflow as tf
import config as cfg
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

    train_func = gen.generate(training=True)
    test_func = gen.generate(training=False)
    for epoch in range(cfg.seq_epoches):
        maml.train(next(train_func))

        for i in range(gen.get_val_len()):
            maml.test(*next(test_func))

        print(
            f'Epoch {epoch + 1}, '
            f'support_Loss: {maml.support_loss.result()}, '
            f'query_Loss:  {maml.query_loss.result()}, '
            f'val_Loss:  {maml.val_loss.result()}, '
        )

        ckpt_save_path = ckpt_manager.save()

        maml.support_loss.reset_states()
        maml.query_loss.reset_states()
        maml.val_loss.reset_states()
