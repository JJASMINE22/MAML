# -*- coding: UTF-8 -*-
'''
@Project ：MAML
@File    ：CustomLayers.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
from tensorflow.keras.layers import (Add,
                                     Dense,
                                     Layer,
                                     Activation
                                     )
from tensorflow.keras import backend as K
import tensorflow as tf


class MyLSTM(Layer):
    """
    完整的自定义长短期记忆网络
    目前无法实现动态seq_len的输入训练
    """
    def __init__(self,
                 units=None,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 recurrent_initializer=None,
                 recurrent_regularizer=None,
                 recurrent_constraint=None,
                 use_bias=False,
                 bias_initializer=None,
                 bias_regularizer=None,
                 bias_constraint=None,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 states=[],
                 **kwargs
                 ):
        super(MyLSTM, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.recurrent_initializer = recurrent_initializer
        self.recurrent_regularizer = recurrent_regularizer
        self.recurrent_constraint = recurrent_constraint
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.states = states

    def get_config(self):
        config = super(MyLSTM, self).get_config()
        config.update({
            'units': self.units,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'recurrent_initializer': self.recurrent_initializer,
            'recurrent_regularizer': self.recurrent_regularizer,
            'recurrent_constraint': self.recurrent_constraint,
            'use_bias': self.use_bias,
            'bias_initializer': self.bias_initializer,
            'bias_regularizer': self.bias_regularizer,
            'bias_constraint': self.bias_constraint,
            'activation': self.activation,
            'recurrent_activation': self.recurrent_activation,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'go_backwards': self.go_backwards,
            'states': self.states
        })
        return config

    def build(self, input_shape):
        """
        切勿于build中拆分i、o、c、f的参数, 否则模型会丢失训练属性
        """
        input_dim = input_shape[-1]
        # self.kernel处理传入本层的输入
        self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        # self.recurrent_kernel处理本层不同时间步的输入
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)

        self.built = True

    @staticmethod
    def multiple_dot_compute(input, h, kernel, recurrent_kernel, bias, activation):

        f = tf.matmul(input, kernel) + tf.matmul(h, recurrent_kernel)
        if bias is not None:
            f = tf.nn.bias_add(f, bias)
        if activation is not None:
            f = Activation(activation)(f)
        return f

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, **kwargs):

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = (
            self.recurrent_kernel[:, self.units: self.units * 2])
        self.recurrent_kernel_c = (
            self.recurrent_kernel[:, self.units * 2: self.units * 3])
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        out_put = []
        h_t_ = tf.zeros(shape=(1, self.units), dtype=tf.float32)  # h(t-1)
        c_t_ = tf.zeros(shape=(1, self.units), dtype=tf.float32)  # C(t-1)

        for i in range(inputs.shape[1]):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(h_t_, tf.TensorShape([None, self.units]))])
            tf.autograph.experimental.set_loop_options(shape_invariants=[(c_t_, tf.TensorShape([None, self.units]))])

            f_t = self.multiple_dot_compute(inputs[:, i, :], h_t_, self.kernel_f,
                                            self.recurrent_kernel_f, self.bias_f,
                                            self.recurrent_activation)

            i_t = self.multiple_dot_compute(inputs[:, i, :], h_t_, self.kernel_i,
                                            self.recurrent_kernel_i, self.bias_i,
                                            self.recurrent_activation)

            _c_t = self.multiple_dot_compute(inputs[:, i, :], h_t_, self.kernel_c,
                                             self.recurrent_kernel_c, self.bias_c,
                                             self.activation)

            c_t_ = f_t * c_t_ + i_t * _c_t

            o_t = self.multiple_dot_compute(inputs[:, i, :], h_t_, self.kernel_o,
                                            self.recurrent_kernel_o, self.bias_o,
                                            self.recurrent_activation)

            if self.activation is not None:
                c_t_ = Activation(self.activation)(c_t_)

            h_t_ = o_t * c_t_

            out_put.append(tf.expand_dims(h_t_, 1))

        out_put = tf.concat(out_put, axis=1)
        self.states.extend([h_t_, c_t_])
        if self.return_sequences:
            if self.return_state:
                out_put = [out_put] + self.states.copy()
        else:
            if self.return_state:
                out_put = [out_put[:, -1, :]] + self.states.copy()
            else:
                out_put = out_put[:, -1, :]
        self.states.clear()
        return out_put

    def compute_output_shape(self, input_shape):

        if self.return_sequences:
            if self.return_state:
                return tuple([(*input_shape[:2], self.units)] + [((input_shape[0], self.units), ) * 2])
            return (*input_shape[:2], self.units)

        if self.return_state:
            return ((input_shape[0], self.units), ) * 3

        return (input_shape[0], self.units)


class LSTMCell(Layer):
    """
    长短期记忆网络单元
    """
    def __init__(self,
                 units=None,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 recurrent_initializer=None,
                 recurrent_regularizer=None,
                 recurrent_constraint=None,
                 use_bias=False,
                 bias_initializer=None,
                 bias_regularizer=None,
                 bias_constraint=None,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 **kwargs):
        super(LSTMCell, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.recurrent_initializer = recurrent_initializer
        self.recurrent_regularizer = recurrent_regularizer
        self.recurrent_constraint = recurrent_constraint
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint
        self.activation = activation
        self.recurrent_activation = recurrent_activation

    def get_config(self):

        config = super(LSTMCell, self).get_config()
        config.update({
            'units': self.units,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'recurrent_initializer': self.recurrent_initializer,
            'recurrent_regularizer': self.recurrent_regularizer,
            'recurrent_constraint': self.recurrent_constraint,
            'use_bias': self.use_bias,
            'bias_initializer': self.bias_initializer,
            'bias_regularizer': self.bias_regularizer,
            'bias_constraint': self.bias_constraint,
            'activation': self.activation,
            'recurrent_activation': self.recurrent_activation,
        })

        return config

    def build(self, input_shape):
        assert len(input_shape) == 2

        feature_dim = input_shape[-1]
        # self.kernel处理传入本层的输入
        self.kernel = self.add_weight(shape=(feature_dim, self.units*4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # self.recurrent_kernel处理本层不同时间步的输入
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units*4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units*4,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.built = True

    @staticmethod
    def multiple_dot_compute(x, h, kernel, recurrent_kernel, bias, activation):

        f = tf.matmul(x, kernel) + tf.matmul(h, recurrent_kernel)
        if bias is not None:
            f = tf.nn.bias_add(f, bias)
        if activation is not None:
            f = Activation(activation)(f)
        return f

    def call(self, input, states=None, *args, **kwargs):

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units*2]
        self.kernel_c = self.kernel[:, self.units*2: self.units*3]
        self.kernel_o = self.kernel[:, self.units*3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units*2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units*2: self.units*3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units*3:]

        if states:
            assert isinstance(states, list)
            h_t, c_t = states
        else:
            h_t = tf.zeros(shape=(tf.shape(input)[0], self.units), dtype=tf.float32)
            c_t = tf.zeros(shape=(tf.shape(input)[0], self.units), dtype=tf.float32)

        f_t = self.multiple_dot_compute(input, h_t, self.kernel_f,
                                        self.recurrent_kernel_f, self.bias_f,
                                        self.recurrent_activation)

        i_t = self.multiple_dot_compute(input, h_t, self.kernel_i,
                                        self.recurrent_kernel_i, self.bias_i,
                                        self.recurrent_activation)

        _c_t = self.multiple_dot_compute(input, h_t, self.kernel_c,
                                         self.recurrent_kernel_c, self.bias_c,
                                         self.activation)

        c_t = f_t * c_t + i_t * _c_t

        o_t = self.multiple_dot_compute(input, h_t, self.kernel_o,
                                        self.recurrent_kernel_o, self.bias_o,
                                        self.recurrent_activation)

        if self.activation is not None:
            c_t = Activation(self.activation)(c_t)

        h_t = o_t * c_t

        return h_t, c_t

class RNN(Layer):
    """
    循环神经网络
    """
    def __init__(self,
                 cell,
                 go_backwards=False,
                 return_state=False,
                 return_sequences=False,
                 states=[],
                 **kwargs):
        super(RNN, self).__init__(**kwargs)
        assert isinstance(cell, object)
        self.cell = cell
        self.states = states
        self.go_backwards = go_backwards
        self.return_state = return_state
        self.return_sequences = return_sequences

    def get_config(self):
        config = super(RNN, self).get_config()
        config.update({
            'cell': self.cell,
            'states': self.states,
            'go_backwards': self.go_backwards,
            'return_state': self.return_state,
            'return_sequences': self.return_sequences
        })
        return config

    def call(self, input, states=None, *args, **kwargs):

        assert len(input.shape) == 3

        input_len = input.shape[1]
        if self.go_backwards:
            input = input[:, ::-1, :]

        out_put = []
        for i in range(input_len):
            h_t, c_t = self.cell(input[:, i, :], states)
            states = [h_t, c_t]
            out_put.append(h_t)

        self.states = [h_t, c_t]
        out_put = tf.concat([tf.expand_dims(_, axis=1) for _ in out_put], axis=1)
        if self.return_sequences:
            if self.return_state:
                out_put = [out_put] + self.states.copy()
        else:
            if self.return_state:
                out_put = [out_put[:, -1, :]] + self.states.copy()
            else:
                out_put = out_put[:, -1, :]
        self.states.clear()
        return out_put
