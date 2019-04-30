#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
    """
    convlstm
    """
    def __init__(self, shape, filters, kernel, dtype, name, rate_list=[1,2],
                forget_bias=1.0,
                peephole=True,
                show=False,
                reuse=tf.AUTO_REUSE):
        super(ConvLSTMCell, self).__init__(_reuse=reuse)
        self._filters = filters
        self._size = tf.TensorShape(shape + [self._filters])
        self._kernel = kernel
        self._dtype  = dtype
        self._name   = name
        self._rate_list   = rate_list
        self._forget_bias = forget_bias
        self._peephole = peephole
        self._reuse = reuse
        self._show  = show
        self.is_in_decoder = False
        self._activate = tf.tanh

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

    @property
    def output_size(self):
          return self._size
    
    def _is_show(self, *args):
        print(*args) if self._show else None
    
    def _conv(self, name, x, c):
        # 这里写法有误，不应该是 output += conv(x, rate)
        with tf.variable_scope(name, reuse=self._reuse):
            for _index, _rate in enumerate(self._rate_list):
                indim  = x.shape.as_list()[-1]
                outdim = self._filters
                _kernel_shape = self._kernel + [indim, outdim]
                w = tf.get_variable(f'weight_rate_{_rate}_conv_{_index}', shape=_kernel_shape, dtype=self._dtype, initializer=tf.glorot_normal_initializer())
                b = tf.get_variable(f'bias_rate_{_rate}_conv_{_index}'  , shape=[outdim]     , dtype=self._dtype, initializer=tf.glorot_normal_initializer())
                x = tf.nn.atrous_conv2d(value=x, filters=w, rate=_rate, padding='SAME') + b
                print(w, '\n', b, '\n', x) if self._show else None
            if self._peephole:
                # 为了节省参数，没有用 shape = c.shape[1:]，而是一个标量代替
                w = tf.get_variable('peephole', shape=[], initializer=tf.glorot_normal_initializer())
                x += w * c
                self._is_show('peephole is on, ', w, '\n', x)
        return x
    
    def call(self, x, state):
        '''
        这里的 x, c, h 都是(N=batch,H,W,C)的样式，input_placeholder 是(N=batch,T,H,W,C)的样式
        '''
        c, h = state
        self._is_show('--------------------')
        self._is_show(c, '\n', '\n', h, '\n', c)
        self._is_show('c , h and x shape is:', c.shape, h.shape, x.shape)
        if self.is_in_decoder == True:
            x = h
        elif self.is_in_decoder == False:
            x = tf.concat([x, h], axis=3)
        else:
            print('error')
        with tf.variable_scope(self._name, reuse=self._reuse):
            i = self._activate(self._conv('Door__i__', x, c))
            f = self._activate(self._conv('Door__f__', x, c) + self._forget_bias)
            c = f * c + i * tf.tanh(self._conv('Door__c__', x, c))
            o = self._activate(self._conv('Door__o__', x, c))
        h = o * tf.tanh(c)
        self._is_show(i, '\n', f, '\n', c, '\n', o, '\n', h)
#         c = c + 1
#         h = h - 1 + x
        state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
        return h, state
