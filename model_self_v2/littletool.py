#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from cell import ConvLSTMCell
from HEADio import BATCH, TIMESTEPS, SHAPE, IMG_input_C, IMG_C, KERNEL, FILTERS, DTYPE
from HEADio import logger

# activate = tf.nn.leaky_relu
activate = tf.tanh


def conv(name, x, filter_shape, outdim, rate):
    '''
    对于 (N,T,H,W,C)的tensor，用 filter_shape 的filter 做卷积，返回形状为(N,T,H,W,C')
    '''
    with tf.variable_scope(name_or_scope='conv', reuse=tf.AUTO_REUSE):
        _shape = x.shape.as_list()
        indim = _shape[-1]
        _shape[-1] = outdim
        w = tf.get_variable(name, shape=[filter_shape[0], filter_shape[1], indim, outdim], dtype=DTYPE, initializer=tf.glorot_normal_initializer())
        b = tf.get_variable(name=name+'__bias', shape=outdim)
        x = tf.nn.atrous_conv2d(value=x[0], filters=w, rate=rate, padding='SAME') + b
        return tf.reshape(activate(x), _shape)

def pool(x, pooling_type='MAX'):
    '''
    only for (N,T,H,W,C) tensor,
    用 2*2 的 max pool，返回 (N,T,H/2,W/2,C)
    '''
    _shape = x.shape.as_list()
    _shape[2], _shape[3] = int(_shape[2]/2), int(_shape[3]/2)
    x = tf.nn.pool(input=x[0], window_shape=[2,2], pooling_type=pooling_type, padding='SAME', strides=[2,2])
    return tf.reshape(x, shape=_shape)

def avg(x):
    '''
    对于 (N,T,H,W,C)的tensor，全局平均，返回形状不变
    '''
    _output_shape = x.shape.as_list()
    _shape = x.shape.as_list()
    _shape[-2], _shape[-3] = 1, 1
    x = tf.reshape(tf.reduce_mean(x, [2, 3]), shape=_shape)[0]
    x = tf.image.resize_nearest_neighbor(x, size=_output_shape[-3:-1], name='avg')
    return tf.cast(tf.reshape(x, shape=_output_shape), dtype=DTYPE)

def avg_8_8(x):
    '''
    对于 (N,T,H,W,C)的tensor，8*8平均，返回形状不变
    '''
    _shape = x.shape.as_list()
    _output_shape = x.shape.as_list()
    x = tf.nn.pool(input=x[0], window_shape=[8,8], pooling_type='AVG', padding='SAME', strides=[8,8])
    x = tf.image.resize_nearest_neighbor(x, size=_output_shape[-3:-1], name='avg_8_8')
    return tf.cast(tf.reshape(x, shape=_output_shape), dtype=DTYPE)

def downsample(name, x, sample_technique):
    '''
    对于(N,T,H,W,C)的tensor，，返回(N,T,2*H,2*W,C)
    upsample 方法为 Dense Upsampling Convolution (DUC), 3*3 的 filter
    或 tf.image.resize_bilinear
    '''
    _shape = x.shape.as_list()
    _shape[2], _shape[3] = int(2*_shape[2]), int(2*_shape[3])
    if sample_technique == 'DUC':
        indim  = _shape[-1]
        outdim = _shape[-1]*4
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = conv(name='DUC_downsample', x=x, filter_shape=(3,3), outdim=outdim, rate=1)
    elif sample_technique == 'bilinear':
        x = tf.image.resize_bilinear(x[0], size=_shape[-3:-1])
        x = tf.cast(x, dtype=DTYPE)
    return tf.reshape(x, shape=_shape)

def creat(name, cell_input, divide, rate_list, _init=None, is_in_decoder=False):
    assert divide in (1,2,4,8,16)
    _shape = [int(SHAPE[0]/divide), int(SHAPE[1]/divide)]
    _cell = ConvLSTMCell(shape=_shape, filters=FILTERS, kernel=KERNEL, dtype=DTYPE, name=name, rate_list=rate_list)
    _cell.is_in_decoder = is_in_decoder
    _outputs, _state = tf.nn.dynamic_rnn(cell=_cell, inputs=cell_input, initial_state=_init, dtype=DTYPE)
    return _init, _outputs, _state

def aspp(name, x, feed=None, is_downsample=True):
    '''
    x.shape = (N,T,H,W,C), 输出形状 (N,T,H=,W=,C=FILTERS)
    '''
    # 鉴于main的结构已经*2/4/8了，这里要不要改成=indim/2呢
    # aspp的输出也改成indim/2可以吗，这样最后输出也是 200*200*FILTERS/2
    _cccccc = int(FILTERS/4)
    _aspp_list = []
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        _aspp_list.append(conv('_rate1', x=x, outdim=_cccccc, rate=1, filter_shape=(1,1)))
        _aspp_list.append(conv('_rate5', x=x, outdim=_cccccc, rate=5, filter_shape=(3,3)))
        _aspp_list.append(conv('_rate9', x=x, outdim=_cccccc, rate=9, filter_shape=(3,3)))
        _aspp_list.append(conv('_rate13', x=x, outdim=_cccccc, rate=13, filter_shape=(3,3)))
        _aspp_list.append(conv('_avg'  , x=avg(x), outdim=_cccccc, rate=1, filter_shape=(1,1)))
        # _aspp_list.append(conv('_avg_8_8', x=avg_8_8(x), outdim=_cccccc, rate=1, filter_shape=(1,1)))
        _aspp_list.append(x)
        if feed != None:
            _aspp_list.append( conv('_feed', x=feed, outdim=_cccccc, rate=1, filter_shape=(3,3)) )
        _aspp = tf.concat(_aspp_list, axis=4)
        _aspp = conv('channel_rectification_1_3_5_9_13_avg_feed', _aspp, outdim=FILTERS, rate=1, filter_shape=(1,1))
        if is_downsample:
#             _aspp = downsample(name, _aspp, sample_technique='bilinear')
            _aspp = downsample(name, _aspp, sample_technique='DUC')
    return _aspp

def SENet(x):
    '''
    Sequeeze and Excitation block，channel attention的典型代表，ImageNet2017冠军。
    对于 (N,T,H,W,C)的tensor，先对(H, W)做全局平均，再对 C 全卷积，再点乘，返回(N,T,H,W,C)。
    '''
    _shape = x.shape.as_list()
    _shape[-2], _shape[-3] = 1, 1
    _cache = tf.reshape(tf.reduce_mean(x, [2, 3]), shape=_shape)
    _cache = conv(name='SENet_fc_on_channel', x=_cache, filter_shape=(1,1), outdim=_shape[-1], rate=1)
    x = x*_cache
    # logger.info('----------, ' + str(x) + ', ' + str(_cache))
    return x

def res_block(x, outdim):
    '''
    (N, T, H, W, C) -> (N, T, H, W, outdim)
    Res2Net, 2019.04.03新鲜出炉的paper，"更细颗粒度表示多尺度信息，增加感受野"
    https://arxiv.org/abs/1904.01169
    https://zhuanlan.zhihu.com/p/61407825
    '''
    assert int(outdim/4.0) == outdim/4.0, '为了顺滑使用Res2Net，输入通道需能被4整除'
    identical = x
    quarter_C = int(outdim/4.0)
    x = conv(name=f'res2net_pre_process', x=x, filter_shape=(1,1), outdim=outdim, rate=1)
    y1 = x[:,:,:,:,0:1*quarter_C]
    y2 = conv(name='res2net_y2', x=x[:,:,:,:,1*quarter_C:2*quarter_C] + y1, filter_shape=(3,3), outdim=quarter_C, rate=1)
    y3 = conv(name='res2net_y3', x=x[:,:,:,:,2*quarter_C:3*quarter_C] + y2, filter_shape=(3,3), outdim=quarter_C, rate=1)
    y4 = conv(name='res2net_y4', x=x[:,:,:,:,3*quarter_C:4*quarter_C] + y3, filter_shape=(3,3), outdim=quarter_C, rate=1)
    x = tf.concat(name='concat_y1_y2_y3_y4', values=(y1, y2, y3, y4), axis=4)
    x = conv(name=f'res2net_post_process', x=x, filter_shape=(1,1), outdim=outdim, rate=1)
    if identical.shape != x.shape:
        identical = conv(name=f'upsample_for_identical', x=identical, filter_shape=(1,1), outdim=outdim, rate=1)
    x = SENet(x)
    x = x + identical
    return x

def res_attention_block(x, outdim):
    '''
    Residual Attention Network，别人的论文笔记https://blog.csdn.net/wspba/article/details/73727469
    参考 https://github.com/KejieLyu/Residual-Attention-Network 实现
    限于前面写的upsample只能2倍扩张，可能有奇偶问题，又不想重写upsample，所以没完全按这个repository来
    嗯，速度，舍取。
    (N, T, H, W, C) -> (N, T, H, W, outdim)
    '''
    # Pre-processing Residual Units
    # with tf.variable_scope('res_attention_pre_processing', reuse=tf.AUTO_REUSE):
    #     x = res_block(x, outdim)
    # Trunk branch
    with tf.variable_scope('res_attention_Trunk_branch', reuse=tf.AUTO_REUSE):
        trunks = res_block(x, outdim)
    # Mask branch
    mask1 = conv(name=f'res_attention_mask1_rate_{1}', x=x, filter_shape=(3,3), outdim=int(1), rate=1)
    # mask2 = conv(name=f'res_attention_mask2_rate_{2}', x=mask1, filter_shape=(3,3), outdim=int(FILTERS/4), rate=2)
    # mask3 = conv(name=f'res_attention_mask3_rate_{1}', x=mask2, filter_shape=(3,3), outdim=int(FILTERS/4), rate=1)
    # mask3 = mask3 + mask1
    # mask4 = conv(name=f'res_attention_mask4_1_rate_{1}', x=mask3, filter_shape=(3,3), outdim=int(FILTERS/4), rate=2)
    # mask4 = conv(name=f'res_attention_mask4_2_rate_{1}', x=mask3, filter_shape=(1,1), outdim=int(1), rate=1)
    # mask4 = tf.nn.sigmoid(mask4)
    # logger.info('----------' + str(mask4) + str(trunks))
    # logger.info('----------------------*******************-------------------')
    output = tf.multiply(trunks, mask1) + trunks
    with tf.variable_scope('res_attention_post_processing', reuse=tf.AUTO_REUSE):
        output = res_block(output, outdim)
    return output

def conv_list(name, x, outdim):
    '''
    输入：(N, T, W, H, C)
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = SENet(x)
        x = res_attention_block(x, outdim)
    return x

def predicting(name, x):
    '''
    (N=1,T,H,W,C) to (T,H,W,C=2) with 3*3 filter
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = conv(name=f'predict_conv', x=x, filter_shape=(3,3), outdim=IMG_input_C, rate=1)
    return tf.cast(x[0], dtype=DTYPE)

def make_alphalist(alpha, _for_):
    alphalist = [alpha]
    for i in range(len(_for_)-1):
        alphalist.append(alphalist[i]*(1-alpha))
    alphalist = list(reversed(alphalist))
    alphalist[0] = alphalist[0]*0.1
    alphalist[1] = alphalist[1]*0.3
    alphalist[2] = alphalist[2]*0.5
    alphalist = np.array(alphalist)
    alphalist = alphalist/alphalist.sum()
    assert len(alphalist)  == len(_for_)
    assert alphalist.sum() - 1.0 < 1e-8
    logger.info(f'alphalist information, len(alphalist)={len(alphalist)}, the actul rate is {alphalist[-1]/alphalist[-2]}. :')
    _for_ = [ _w * _loss_item for _w, _loss_item in zip(alphalist, _for_)]
    assert len(alphalist)  == len(_for_)
    for _w, _loss_item in zip(alphalist, _for_):
        # logger.info('_w, _loss_item: ' + str((_w, _loss_item)))
        pass
    
    return _for_



