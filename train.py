# -*- coding: UTF-8 -*-
'''
@Project ：MAML
@File    ：train.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import tensorflow as tf
import config as cfg
from models.maml import MAML
from _utils import Generator

if __name__ == '__main__':

    gen = Generator(file_path=cfg.file_path, classify_tasks=cfg.classify_tasks,
                    single_task_class=cfg.single_task_class, support_query_size=cfg.support_query_size,
                    query_ratio=cfg.query_ratio, val_len=cfg.query_ratio, thresh=cfg.thresh)

    maml = MAML(input_shape=cfg.input_shape, class_num=cfg.class_num,
                lr_rate=cfg.learning_rate)

    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)

    ckpt = tf.train.Checkpoint(maml=maml.model,
                               optimizer=maml.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, cfg.ckpt_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点，加载模型
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_func = gen.generate(training=True)
    test_func = gen.generate(training=False)
    for epoch in range(cfg.epoches):
        maml.train(next(train_func))

        for i in range(gen.get_val_len()):
            maml.test(*next(test_func))

        print(
            f'Epoch {epoch + 1}, '
            f'support_Loss: {maml.support_loss.result()}, '
            f'support_Acc: {maml.support_acc.result() * 100}, '
            f'query_Loss:  {maml.query_loss.result()}, '
            f'query_Acc: {maml.query_acc.result() * 100}',
            f'val_Loss:  {maml.val_loss.result()}, '
            f'val_Acc: {maml.val_acc.result() * 100}'
        )

        ckpt_save_path = ckpt_manager.save()

        maml.support_loss.reset_states()
        maml.support_acc.reset_states()
        maml.query_loss.reset_states()
        maml.query_acc.reset_states()
        maml.val_loss.reset_states()
        maml.val_acc.reset_states()