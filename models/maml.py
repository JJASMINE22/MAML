# -*- coding: UTF-8 -*-
'''
@Project ：MAML
@File    ：maml.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Dense,
                                     Conv2D,
                                     Flatten,
                                     LeakyReLU,
                                     Activation,
                                     BatchNormalization,
                                     GlobalAveragePooling2D
                                     )
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import GlorotNormal, Zeros


class MAML:

    """
    主要应用于小样本模型训练
    maml-tf2版本
    """
    def __init__(self, input_shape, class_num, lr_rate):
        """
        :param input_shape: 输入形状
        :param class_num: 类别数目
        :param lr_rate: 学习率
        """
        assert isinstance(lr_rate, dict)

        self.input_shape = input_shape
        self.class_num = class_num

        self.lr_rate = lr_rate
        self.sub_lr = self.lr_rate['sub_lr']
        self.meta_lr = self.lr_rate['meta_lr']
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.meta_lr) # meta-optimizer

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

        self.total_loss = []
        self.support_loss = tf.keras.metrics.Mean()
        self.query_loss = tf.keras.metrics.Mean()

        self.support_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.query_acc = tf.keras.metrics.SparseCategoricalAccuracy()

        self.val_loss = tf.keras.metrics.Mean()
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

        self.model = self.CreateModel()

    def CreateModel(self):
        """
        创建分类模型
        针对小样本训练, 模型务必轻量化
        :return: 模型对象
        """
        model = Sequential([
            Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                   kernel_initializer=GlorotNormal(), bias_initializer=Zeros(),
                   input_shape=self.input_shape),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                   kernel_initializer=GlorotNormal(), bias_initializer=Zeros()),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(filters=256, kernel_size=3, strides=1, padding='same',
                   kernel_initializer=GlorotNormal(), bias_initializer=Zeros()),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            # Flatten(),
            GlobalAveragePooling2D(),
            Dense(units=self.class_num, kernel_initializer=GlorotNormal(),
                  bias_initializer=Zeros()),
            Activation('softmax')
        ])
        return model

    def forward(self, source, real_labels):
        """
        This method executes a forward pass of the model using input x (model prediction).
        It uses the lossFunction to calculate the loss and returns both the loss and the predictions
        """
        pred_labels = self.model(source)
        loss = self.loss(real_labels,pred_labels)
        return loss, pred_labels

    def train(self, targets):
        """
        创建两个tape, 控制sub模型与support模型的误差loss作用域
        """
        with tf.GradientTape() as query_tape:
            for target in targets:
                support_src, support_tgt, query_src, query_tgt = target
                with tf.GradientTape() as support_tape:
                    support_loss, support_logits = self.forward(support_src, support_tgt)  # Compute loss of Ti

                # 循环计算各sub模型在其task下的梯度
                sub_gradients = support_tape.gradient(support_loss, self.model.trainable_variables)

                sub_Model = MAML(self.input_shape, self.class_num, self.lr_rate, self.resume_train)
                sub_Model.model.set_weights(self.model.get_weights())

                """
                使用手动运算更新模型参数, Var→Constant
                模型将丢失可训练属性(主动丢失梯度),
                因此sub模型的loss可作用于support模型
                """
                z = 0
                for k in range(len(sub_Model.model.layers)):
                    if sub_Model.model.layers[k].name.split('_')[0] not in ['batch', 'conv2d', 'dense']:
                        continue
                    else:
                        if sub_Model.model.layers[k].name.split('_')[0] == 'batch':
                            sub_Model.model.layers[k].gamma = tf.subtract(self.model.layers[k].gamma,
                                                                          tf.multiply(self.sub_lr, sub_gradients[z]))
                            sub_Model.model.layers[k].beta = tf.subtract(self.model.layers[k].beta,
                                                                         tf.multiply(self.sub_lr, sub_gradients[z + 1]))
                        else:
                            sub_Model.model.layers[k].kernel = tf.subtract(self.model.layers[k].kernel,
                                                                           tf.multiply(self.sub_lr, sub_gradients[z]))
                            sub_Model.model.layers[k].bias = tf.subtract(self.model.layers[k].bias,
                                                                         tf.multiply(self.sub_lr, sub_gradients[z + 1]))
                        z += 2

                query_loss, query_logits = sub_Model.forward(query_src, query_tgt)
                self.support_loss(support_loss)
                self.query_loss(query_loss)
                self.support_acc(support_tgt, support_logits)
                self.query_acc(query_tgt, query_logits)
                self.total_loss.append(query_loss)
            # 计算所有task的平均误差
            avg_query_loss = tf.reduce_mean(self.total_loss)
            meta_gradients = query_tape.gradient(avg_query_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(meta_gradients, self.model.trainable_variables))
            self.total_loss.clear()

    def test(self, source, label):
        """
        Performs a prediction on new datapoints and evaluates the prediction (loss)
        """
        with tf.GradientTape() as tape:
            loss, logits = self.forward(source, label)
        self.val_loss(loss)
        self.val_acc(label, logits)
