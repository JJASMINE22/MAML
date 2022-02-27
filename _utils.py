# -*- coding: UTF-8 -*-
'''
@Project ：MAML
@File    ：_utils.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''

import os
import cv2
import numpy as np
import pandas as pd

class Generator:
    """
    数据生成器, 图像分类
    """
    def __init__(self,
                 file_path,
                 classify_tasks,
                 single_task_class,
                 support_query_size,
                 query_ratio,
                 val_len,
                 thresh):

        self.file_path = file_path
        self.dirs = np.array(os.listdir(self.file_path))
        self.classify_nums = len(self.dirs)
        assert single_task_class < self.classify_nums

        self.classify_tasks = classify_tasks
        self.single_task_class = single_task_class
        self.support_query_size = support_query_size
        self.query_ratio = query_ratio
        self.val_len = val_len
        self.thresh = thresh
        self.tasks_dir_idx = self.create_task()

    def create_task(self):
        """
        分配task_num个任务
        """
        while True:
            tasks_index = np.array([[*np.random.choice(self.classify_nums, self.single_task_class, replace=False)]
                                    for i in range(self.classify_tasks)])
            total_index = np.reshape(tasks_index, [-1])

            if np.all(np.array([list(total_index).count(i) for i in range(self.classify_nums)]) > self.thresh):
                return tasks_index

    def get_val_len(self):

        return len(self.dirs) * self.val_len // (self.support_query_size//2)

    def generate(self, training=True):
        """
        训练过程包含了task中的query、support成分划分
        """
        while True:
            if training:
                task_files = np.array([[np.random.choice(os.listdir(os.path.join(self.file_path, task_dir)),
                                                         self.support_query_size//4, replace=False)
                                        for i, task_dir in enumerate(self.dirs[task_dir_idx])]
                                       for task_dir_idx in self.tasks_dir_idx])

                targets, sources, labels = [], [], []
                for i, task_dir_idx in enumerate(self.tasks_dir_idx):
                    for j, task_dir in enumerate(self.dirs[task_dir_idx]):
                        for file in task_files[i, j]:
                            image = cv2.imread(os.path.join(os.path.join(self.file_path, task_dir), file))
                            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
                            image = np.array(image) / 127.5 - 1
                            image = np.clip(image, -1., 1.)
                            sources.append(image)
                            labels.append(task_dir_idx[j])

                    support_targets = np.array(sources)[:int(self.support_query_size*self.query_ratio)]
                    query_targets = np.array(sources)[int(self.support_query_size*self.query_ratio):]
                    support_labels = np.array(labels)[:int(self.support_query_size*self.query_ratio)]
                    query_labels = np.array(labels)[int(self.support_query_size*self.query_ratio):]
                    targets.append([support_targets, support_labels, query_targets, query_labels])
                    sources.clear()
                    labels.clear()

                annotation_targets = targets.copy()
                targets.clear()
                yield annotation_targets

            else:
                sources, labels = [], []
                val_files = np.array([np.random.choice(os.listdir(os.path.join(self.file_path, dir)),
                                                       self.val_len, replace=False) for dir in self.dirs])
                for i in range(val_files.shape[-1]):
                    for j, val_dir in enumerate(self.dirs):
                        image = cv2.imread(os.path.join(os.path.join(self.file_path, val_dir), val_files[j, i]))
                        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
                        image = np.array(image) / 127.5 - 1
                        image = np.clip(image, -1., 1.)
                        sources.append(image)
                        labels.append(list(self.dirs).index(val_dir))

                        if np.logical_and(np.equal(len(sources), self.support_query_size//2),
                                          np.equal(len(labels), self.support_query_size//2)):
                            annotation_sources, annotation_labels = sources.copy(), labels.copy()
                            sources.clear()
                            labels.clear()
                            yield np.array(annotation_sources), np.array(annotation_labels)


class DataGenerator():
    """
    数据生成器, 数据预测
    """
    def __init__(self,
                 txt_path,
                 time_seq,
                 train_ratio,
                 task_num,
                 query_ratio,
                 support_query_size):
        self.txt_path = txt_path
        self.time_seq = time_seq
        self.train_ratio = train_ratio
        self.task_num = task_num
        self.query_ratio = query_ratio
        self.support_query_size = support_query_size
        self.train_src, self.train_tgt, self.val_src, self.val_tgt = self.split_train_val()
        self.tasks_idx = self.create_task()

    def create_task(self):
        """
        分配task_num个任务
        """
        main_part_idx = np.arange(len(self.train_src))
        residue_part_idx = np.random.choice(len(self.train_src),
                                            self.support_query_size * self.task_num - len(main_part_idx),
                                            replace=False)
        tasks_idx = np.concatenate([main_part_idx, residue_part_idx])

        return tasks_idx

    def split_train_val(self):

        total_src, total_tgt = self.preprocess()
        index = np.arange(len(total_src))
        np.random.shuffle(index)
        train_src = total_src[index[:int(self.train_ratio * len(index))]]
        train_tgt = total_tgt[index[:int(self.train_ratio * len(index))]]
        val_src = total_src[index[int(self.train_ratio * len(index)):]]
        val_tgt = total_tgt[index[int(self.train_ratio * len(index)):]]

        return train_src, train_tgt, val_src, val_tgt

    def preprocess(self):
        """
        读取数据, 恢复时间戳, 线性插值(高阶多项式拟合亦可)
        :return: 预处理数据
        """
        df = pd.DataFrame(pd.read_csv(self.txt_path)).iloc[:630, :]
        df[df.keys()[0]] = pd.to_datetime(df.Date)
        df = df.set_index(df.keys()[0])

        idx = pd.date_range(start=df.index[0], end=df.index[-1], freq='D')
        new_df = df.reindex(idx, fill_value=0)
        data_len = np.arange(len(new_df))
        for key in new_df.keys():
            bool = new_df[key] > 0
            state = np.where(bool)

            interped_data = np.interp(data_len, state[0], np.array(new_df[key])[state])
            new_df[key] = interped_data
        new_df = (new_df - new_df.min(axis=0)) / (new_df.max(axis=0) - new_df.min(axis=0))

        assign_source = np.array([np.array(new_df)[i:i + self.time_seq] for i in range(len(new_df) - self.time_seq)])
        assign_target = np.array([np.array(new_df)[i + self.time_seq] for i in range(len(new_df) - self.time_seq)])

        return assign_source, assign_target

    def get_val_len(self):

        return len(self.val_src)//(self.support_query_size//2)

    def generate(self, training=True):
        """
        训练过程包含了task中的query、support成分划分
        """
        while True:
            if training:
                tasks_idx = self.tasks_idx
                np.random.shuffle(tasks_idx)
                tasks_idx = np.reshape(tasks_idx, [-1, self.support_query_size])
                targets = []
                for task_idx in tasks_idx:
                    support_sources = self.train_src[task_idx][:int(self.support_query_size*self.query_ratio)]
                    support_targets = self.train_tgt[task_idx][:int(self.support_query_size*self.query_ratio)]
                    query_sources = self.train_src[task_idx][int(self.support_query_size*self.query_ratio):]
                    query_targets = self.train_tgt[task_idx][int(self.support_query_size*self.query_ratio):]
                    targets.append([support_sources, support_targets, query_sources, query_targets])
                    # yield [support_sources, support_targets, query_sources, query_targets]
                annotation_targets = targets.copy()
                targets.clear()
                yield annotation_targets

            else:
                idx = np.arange(len(self.val_src))
                np.random.shuffle(idx)
                val_src, val_tgt = self.val_src[idx], self.val_tgt[idx]
                sources, targets = [], []
                for src, tgt in zip(val_src, val_tgt):
                    sources.append(src)
                    targets.append(tgt)

                    if np.logical_and(np.equal(len(sources), self.support_query_size//2),
                                      np.equal(len(targets), self.support_query_size//2)):
                        annotation_sources, annotation_targets = sources.copy(), targets.copy()
                        sources.clear()
                        targets.clear()

                        yield np.array(annotation_sources), np.array(annotation_targets)
