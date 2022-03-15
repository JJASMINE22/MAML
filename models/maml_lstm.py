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
    maml原文用于图像分类, 此处改造为数据预测
    """
    def __init__(self, input_shape, feature_dims, lr_rate):
        """
        :param input_shape: 输入形状
        :param feature_dims: 时序数据特征维度
        :param lr_rate: 学习率
        """

        assert isinstance(lr_rate, dict)

        self.input_shape = input_shape
        self.feature_dims = feature_dims

        self.lr_rate = lr_rate
        self.sub_lr = self.lr_rate['sub_lr']
        self.meta_lr = self.lr_rate['meta_lr']
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.meta_lr) # adam优化器

        self.loss = tf.keras.losses.MeanSquaredError() # 均方误差

        self.total_loss = []
        self.support_loss = tf.keras.metrics.Mean()
        self.query_loss = tf.keras.metrics.Mean()
        self.val_loss = tf.keras.metrics.Mean()

        self.model = self.CreateModel()

    def CreateModel(self):
        """
        使用自定义长短期记忆网络搭建模型
        注意: tensorflow所提供的LSTM模块在CUDA硬件加速下无法实现MAML, 与cudnn接口bug有关
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
        """
        This method executes a forward pass of the model using input x (model prediction).
        It uses the lossFunction to calculate the loss and returns both the loss and the predictions
        """
        pred = self.model(source)
        loss = tf.reduce_mean(self.loss(real,pred))
        return loss, pred

    def train(self, targets):
        """
        创建两个tape, 均用于附着meta模型的梯度
        """
        with tf.GradientTape() as query_tape:
            for target in targets:
                support_src, support_tgt, query_src, query_tgt = target
                with tf.GradientTape() as support_tape:
                    support_loss, support_logits = self.forward(support_src, support_tgt)  # Compute loss of Ti

                # 循环计算各sub模型在其task下的梯度
                sub_gradients = support_tape.gradient(support_loss, self.model.trainable_variables)

                sub_Model = MAML(self.input_shape, self.feature_dims, self.lr_rate, self.resume_train)
                sub_Model.model.set_weights(self.model.get_weights())

                """
                使用手动运算更新模型参数, Var→带meta模型梯度的Constant
                致使sub模型丢失训练属性(主动丢失梯度), 反而附着meta模型的梯度
                因此sub模型输出的loss可梯度下降作用于meta模型
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
                # 统计各附着meta模型梯度的loss
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
