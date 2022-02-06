# -*- coding: UTF-8 -*-
'''
@Project ：MAML
@File    ：maml_lstm.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input,
                                     LSTM,
                                     Dense,
                                     Conv2D,
                                     Dropout,
                                     Flatten,
                                     Attention,
                                     LeakyReLU,
                                     Activation,
                                     BatchNormalization,
                                     GlobalAveragePooling2D
                                     )
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import GlorotNormal, Zeros
from CustomLayers import MyLSTM

class MAML:

    """
    主要应用于小样本模型训练
    maml原论文仅用于分类, 此处改造为数据预测
    """
    def __init__(self, input_shape, feature_dims, lr_rate, resume_train=False):
        """
        :param input_shape: 输入形状
        :param feature_dims: 时序数据特征维度
        :param lr_rate: 学习率
        :param resume_train: 是否继续训练
        """

        assert isinstance(lr_rate, dict)

        self.input_shape = input_shape
        self.feature_dims = feature_dims

        self.lr_rate = lr_rate
        self.sub_lr = self.lr_rate['sub_lr']
        self.meta_lr = self.lr_rate['meta_lr']
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.meta_lr)  # meta-optimizer

        self.loss = tf.keras.losses.MeanSquaredError()

        self.total_loss = []
        self.support_loss = tf.keras.metrics.Mean()
        self.query_loss = tf.keras.metrics.Mean()
        self.val_loss = tf.keras.metrics.Mean()

        self.resume_train = resume_train

        self.model = self.CreateModel()
        self.initial_weights = self.model.get_weights()

        if self.resume_train:
            self.model.load_weights(".\\模型文件")

    def CreateModel(self):
        """
        使用自定义长短期记忆网络搭建模型
        注意: tensorflow提供的LSTM层在CUDA加速状态下无法实现MAML, 该BUG与cudnn有关
        """
        input = Input(shape=self.input_shape)
        x = MyLSTM(units=64,
                   use_bias=True,
                   return_state=True,
                   return_sequences=True,
                   kernel_initializer=GlorotNormal(),
                   recurrent_initializer=GlorotNormal(),
                   bias_initializer=Zeros())(input)
        x = Attention()(x)
        x = MyLSTM(units=128,
                   use_bias=True,
                   return_state=True,
                   return_sequences=True,
                   kernel_initializer=GlorotNormal(),
                   recurrent_initializer=GlorotNormal(),
                   bias_initializer=Zeros())(x)
        x = Attention()(x)
        x = MyLSTM(units=256,
                   use_bias=True,
                   return_state=False,
                   return_sequences=False,
                   kernel_initializer=GlorotNormal(),
                   recurrent_initializer=GlorotNormal(),
                   bias_initializer=Zeros())(x)
        # x = Dropout(rate=0.3)(x)
        x = Dense(units=self.feature_dims,
                  use_bias=False,
                  kernel_initializer=GlorotNormal())(x)
        # x = Dropout(rate=0.3)(x)
        output = Activation('sigmoid')(x)

        return Model(input, output)

    def forward(self, source, real):
        pred = self.model(source)
        loss = tf.reduce_mean(self.loss(real,pred))
        return loss, pred

    def train(self, targets):
        """
        创建两个tape, 控制sub模型与support模型的误差loss作用域
        """
        with tf.GradientTape() as query_tape:
            for target in targets:
                support_src, support_tgt, query_src, query_tgt = target
                with tf.GradientTape() as support_tape:
                    support_loss, support_logits = self.forward(support_src, support_tgt)  # Compute loss of Ti

                # Create temporary model to compute θ` - applying gradients
                sub_gradients = support_tape.gradient(support_loss, self.model.trainable_variables)

                sub_Model = MAML(self.input_shape, self.feature_dims, self.lr_rate, self.resume_train)
                sub_Model.model.set_weights(self.model.get_weights())

                """
                使用手动运算更新模型参数, Var→Constant
                模型将丢失可训练属性(主动丢失梯度),
                因此sub模型的loss可作用于support模型
                """
                z = 0
                for k in range(len(sub_Model.model.layers)):
                    if sub_Model.model.layers[k].name.split('_')[0] not in ['lstm', 'dense']:
                        continue
                    else:
                        if sub_Model.model.layers[k].name.split('_')[0] == 'dense':
                            sub_Model.model.layers[k].kernel = tf.subtract(self.model.layers[k].kernel,
                                                                           tf.multiply(self.sub_lr, sub_gradients[z]))
                            z += 1
                        else:
                            sub_Model.model.layers[k].kernel = tf.subtract(self.model.layers[k].kernel,
                                                                                tf.multiply(self.sub_lr, sub_gradients[z]))
                            sub_Model.model.layers[k].recurrent_kernel = tf.subtract(self.model.layers[k].recurrent_kernel,
                                                                                          tf.multiply(self.sub_lr, sub_gradients[z+1]))
                            sub_Model.model.layers[k].bias = tf.subtract(self.model.layers[k].bias,
                                                                              tf.multiply(self.sub_lr, sub_gradients[z+2]))
                            z += 3

                query_loss, query_logits = sub_Model.forward(query_src, query_tgt)
                self.support_loss(support_loss)
                self.query_loss(query_loss)
                self.total_loss.append(query_loss)
            # 计算所有task的平均误差
            avg_query_loss = tf.reduce_mean(self.total_loss)
            meta_gradients = query_tape.gradient(avg_query_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(meta_gradients, self.model.trainable_variables))
            self.total_loss.clear()

    def test(self, source, target):
        """
        Performs a prediction on new datapoints and evaluates the prediction (loss)
        """
        with tf.GradientTape() as tape:
            loss, logits = self.forward(source, target)
        self.val_loss(loss)
