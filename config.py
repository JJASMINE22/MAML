# -*- coding: UTF-8 -*-
'''
@Project ：MAML
@File    ：config.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''

# ===sequence prediction===

# data_generator
text_path = '数据文件绝对路径'
time_seq = 7
seq_train_ratio = 0.7
seq_task_num = 12
seq_query_ratio = 0.5
seq_sq_size = 64

# training
seq_epoches = 150
feature_dims = 5
seq_learning_rate = {'sub_lr': 1e-3, 'meta_lr': 1e-3},
seq_ckpt_path = '.\\tf_lstm_models\\checkpoint'


# ===label classification===

# data_generator
file_path = '图像文件绝对文件'
classify_tasks = 50
single_task_class = 4  # 每个任务允许的样本类别上限
support_query_size = 16
query_ratio = 0.5
val_len = 8
thresh = 6

# training
epoches = 300
input_shape=(28, 28, 3)
class_num=20
learning_rate={'sub_lr': 1e-3, 'meta_lr': 1e-3}
ckpt_path = '.\\tf_models\\checkpoint'
