#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from PIL import Image
from tqdm import tqdm
from datetime import timedelta

from cell import ConvLSTMCell
from littletool import conv, pool, avg, avg_8_8, downsample, predicting, creat, aspp, conv_list, make_alphalist
from HEADio import BATCH, TIMESTEPS, SHAPE, VIEW, IMG_input_C, IMG_C, KERNEL, FILTERS, DTYPE, IO_version
from HEADio import logger, encoding, decoding, read_as_array, axis_off_imshow, fast_submit, make_iterator
from HEADio import RESHAPE_spacing, LOOP_num, SPACING
import argparse
parser = argparse.ArgumentParser(description="used for control")
parser.add_argument("control")
parser.add_argument("lamda", type=float)
args = parser.parse_args()


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# model building
logger.info('building mode ...')
# -----------------------------------------
# -------------- placeholder --------------
input_placeholder    = tf.placeholder(shape=[BATCH, TIMESTEPS] + SHAPE + [IMG_C], dtype=DTYPE, name='input_placeholder')
lr_placeholder       = tf.placeholder(shape=[], dtype=DTYPE, name='lr_placeholder')
lambda_placeholder   = tf.placeholder(shape=[], dtype=DTYPE, name='lambda_placeholder')
momentum_placeholder = tf.placeholder(shape=[], dtype=DTYPE, name='momentum_placeholder')
# --------------------------------------
# -------------- encoding --------------
# conv_outputs_0 = conv_list(name='conv_list_0', x=input_placeholder, outdim=FILTERS)
conv_outputs_0 = conv(name='conv_0', x=input_placeholder, filter_shape=(3,3), outdim=FILTERS, rate=1)
eq_fft_a = downsample(name='eq_fft_a', x=pool(pool(input_placeholder)), sample_technique='bilinear')
eq_fft_b = downsample(name='eq_fft_b', x=pool(pool(eq_fft_a)), sample_technique='bilinear')

# --- pool ---
# 鉴于 FILTERS 不大，为了充分利用 feature，在同一层pool之内用 DenseNet 结构
pool_outputs_o_i = pool(conv_outputs_0[:,:,:,:,0:3])

conv_outputs_1 = conv_list(name='conv_list_1', x=pool_outputs_o_i, outdim=FILTERS)

densenet_input = tf.concat([eq_fft_a, pool_outputs_o_i, conv_outputs_1], axis=4)
init_a, cell_outputs_a, state_a = creat(name='cell_a', cell_input=densenet_input, divide=2, rate_list=[1])

# --- pool ---
pool_outputs_o_ii = pool(cell_outputs_a)

conv_outputs_2 = conv_list(name='conv_list_2', x=pool_outputs_o_ii, outdim=FILTERS)

densenet_input = tf.concat([eq_fft_b, pool_outputs_o_ii, conv_outputs_2], axis=4)
init_b, cell_outputs_b, state_b = creat(name='cell_b', cell_input=densenet_input, divide=4, rate_list=[1])

# --- pool ---
pool_outputs_o_iii = pool(cell_outputs_b)

conv_outputs_3 = conv_list(name='conv_list_3', x=pool_outputs_o_iii, outdim=FILTERS)

densenet_input = tf.concat([pool_outputs_o_iii, conv_outputs_3], axis=4)
init_c, cell_outputs_c, state_c = creat(name='cell_c', cell_input=densenet_input, divide=8, rate_list=[1])

# -------------------------------------------
# -------------- aspp decoding --------------
aspp_a = aspp(name='aspp_a_0', x=cell_outputs_c, feed=None, is_downsample=False)
aspp_a = aspp(name='aspp_a_1', x=conv_outputs_3, feed=aspp_a, is_downsample=True)

aspp_b = aspp(name='aspp_b_0', x=cell_outputs_b, feed=aspp_a, is_downsample=False)
aspp_b = aspp(name='aspp_b_1', x=conv_outputs_2, feed=aspp_b, is_downsample=True)

aspp_c = aspp(name='aspp_c_0', x=cell_outputs_a, feed=aspp_b, is_downsample=False)
aspp_c = aspp(name='aspp_c_1', x=conv_outputs_1, feed=aspp_c, is_downsample=True)

aspp_d = aspp(name='aspp_d_0', x=conv_outputs_0, feed=aspp_c, is_downsample=False)

aspp_e = tf.concat(name='concat_with_input', values=(aspp_d, input_placeholder), axis=4)
aspp_e = aspp(name='aspp_e_0', x=aspp_e, feed=None, is_downsample=False)
# aspp_e = conv(name='cc', x=aspp_e, filter_shape=(3,3), outdim=int(FILTERS/4), rate=1)

predict_in_encoder = predicting(name='main_aspp_e', x=aspp_e)
assert predict_in_encoder.shape.as_list() == [TIMESTEPS]+SHAPE+[IMG_input_C]

# 留待备用
# init_list  = [init_a, init_b, init_c]
# state_list = [state_a, state_b, state_c]

# predict
predict_base_on_img_i_list = []
for i in range(TIMESTEPS):
    _ = predict_in_encoder[i,:,:,:]
    predict_base_on_img_i_list.append(_)
    # logger.info( 'predict_base_on_img_i_list: ' + str(i) + ', ' + str(_) )
assert len(predict_base_on_img_i_list) == TIMESTEPS
assert predict_base_on_img_i_list[0].shape.as_list() == SHAPE+[IMG_input_C]

logger.info('model have been built')
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

# 輔助loss
logger.info('make help_loss op:')
def make_help_loss(name, aspp_, pp, w):
    '''
    把 lambda(predicting, [asppa, asppb, asppc]) 分别于 input 不同尺寸的pool对比。
    '''
    cache = input_placeholder
    for _ in range(pp):
        cache = pool(cache, pooling_type='AVG')
    aspp_predict_ = predicting(name=name, x=aspp_)
    cache_loss_list = []
    for i in range(TIMESTEPS-3):
        _1 = tf.sqrt(tf.sqrt(tf.sqrt(tf.reduce_mean( tf.math.squared_difference(aspp_predict_[i,:,:,0:1], cache[0,i+1,:,:,0:1]) ))))
        _2 = tf.sqrt(tf.sqrt(tf.sqrt(tf.reduce_mean( tf.math.squared_difference(aspp_predict_[i,:,:,1:2], cache[0,i+1,:,:,1:2]) ))))
        _3 = tf.sqrt(tf.sqrt(tf.sqrt(tf.reduce_mean( tf.math.squared_difference(aspp_predict_[i,:,:,2:3], cache[0,i+1,:,:,2:3]) ))))
        cache_loss_list.append(0.3333*_1 + 0.3333*_2 + 0.3333*_3)
    assert len(cache_loss_list) == TIMESTEPS - 3
    return w*tf.reduce_mean(cache_loss_list)

help_loss = 0
help_loss += make_help_loss(name='help_loss_aspp_a', aspp_=aspp_a, pp=2, w=0.005)
help_loss += make_help_loss(name='help_loss_aspp_b', aspp_=aspp_b, pp=1, w=0.010)
help_loss += make_help_loss(name='help_loss_aspp_c', aspp_=aspp_c, pp=0, w=0.015)
help_loss += make_help_loss(name='help_loss_aspp_d', aspp_=aspp_d, pp=0, w=0.020)
logger.info('help_loss op have been made')

# make loss
logger.info('make loss_list ...')
loss_list = []
for i in range(TIMESTEPS - 1):
    # logger.info(f'---{i}-------------------------------------****************-------------------------------------')

    # _1 = tf.sqrt(tf.sqrt(tf.sqrt(tf.reduce_mean( tf.math.squared_difference(predict_base_on_img_i_list[i][:,:,1:2], input_placeholder[0,i+1,:,:,1:2]) ))))
    # _2 = tf.sqrt(tf.sqrt(tf.sqrt(tf.reduce_mean( tf.math.squared_difference(predict_base_on_img_i_list[i][:,:,2:3], input_placeholder[0,i+1,:,:,2:3]) ))))
    # _  = 0.5*_1 + 0.5*_2

    _1 = tf.sqrt(tf.reduce_mean( tf.math.squared_difference(predict_base_on_img_i_list[i][:,:,1:2], input_placeholder[0,i+1,:,:,1:2]) ))
    _2 = tf.sqrt(tf.reduce_mean( tf.math.squared_difference(predict_base_on_img_i_list[i][:,:,2:3], input_placeholder[0,i+1,:,:,2:3]) ))
    _  = tf.sqrt(0.5 * tf.sqrt(_1) + 0.5 * tf.sqrt(_2))

    loss_list.append(_)

assert len(loss_list) == TIMESTEPS - 1
loss_list_len = len(loss_list)

logger.info('make loss_L2, loss, val, test and submit op:')
loss_L2 = tf.add_n([tf.nn.l2_loss(_) for _ in tf.trainable_variables() if 'bias' not in _.name])

_for_train = loss_list[: loss_list_len-2]
_for_train = make_alphalist(alpha=0.025, _for_=_for_train)
loss = tf.add_n(_for_train) + lambda_placeholder*loss_L2 + help_loss
assert len(_for_train) == loss_list_len - 2

_for_val   = loss_list[loss_list_len-2]
val_see  = tf.reduce_mean(_for_val)

_for_test  = loss_list[loss_list_len-1]
test_see = tf.reduce_mean(_for_test)

_for_submit  = loss_list[: loss_list_len]
_for_submit  = make_alphalist(alpha=0.025, _for_=_for_submit)
submit = tf.add_n(_for_submit) + lambda_placeholder*loss_L2 + help_loss
assert len(_for_submit) == loss_list_len

logger.info('loss_L2, loss, val, test have been made.')


# -------------------------------------------------------
# just a test
logger.info('')
logger.info('just a test')
with tf.variable_scope(name_or_scope='opt', reuse=tf.AUTO_REUSE):
    # adam
    optimizer_adam = tf.train.AdamOptimizer(lr_placeholder)
    optimizer_step_adam = optimizer_adam.minimize(loss)
    optimizer_adam_in_val = tf.train.AdamOptimizer(lr_placeholder)
    optimizer_step_adam_in_val = optimizer_adam.minimize(0.5*loss+0.5*val_see)
    optimizer_adam_in_submit = tf.train.AdamOptimizer(lr_placeholder)
    optimizer_step_adam_in_submit = optimizer_adam.minimize(submit)
    # moment
    optimizer_moment = tf.train.MomentumOptimizer(learning_rate=lr_placeholder, momentum=momentum_placeholder)
    optimizer_step_moment = optimizer_moment.minimize(loss)
    optimizer_moment_in_val = tf.train.MomentumOptimizer(learning_rate=lr_placeholder, momentum=momentum_placeholder)
    optimizer_step_moment_in_val = optimizer_moment.minimize(0.5*loss+0.5*val_see)
    optimizer_moment_in_submit = tf.train.MomentumOptimizer(learning_rate=lr_placeholder, momentum=momentum_placeholder)
    optimizer_step_moment_in_submit = optimizer_moment.minimize(submit)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_dict = {lr_placeholder:0.001, lambda_placeholder:8e-6, momentum_placeholder:0.9}
    feed_dict[input_placeholder] = np.random.randn(BATCH, TIMESTEPS, VIEW, VIEW, IMG_C)
    _ = sess.run([optimizer_step_adam, loss_list, loss_L2, help_loss, loss, val_see, test_see], feed_dict=feed_dict)
    start_time = time.time()
    _ = sess.run([optimizer_step_adam, loss_list, loss_L2, help_loss, loss, val_see, test_see], feed_dict=feed_dict)
    end_time   = time.time()
logger.info(f'a step need {end_time - start_time} s')
logger.info(f'see if nan: {_[2:]}')
logger.info('test over, everything is ok.')
logger.info('')

# -------------------------------------------------------
# training
def train(_optimizer_step, lamda, lr, momentum, loop_num, spacing, zone):
    global recoding_list
    logger.info(f'zone: {zone}, true_lamda: {lamda*1.5}, lr: {lr} ---> {lr*(0.7 + 0.3 * (1 - (loop_num-1)/loop_num))}')
    _list_in_run = [_optimizer_step, loss_list, loss_L2, help_loss, loss, val_see, test_see]
    start_time = time.time()
    # 真正的lamda
    feed_dict[lambda_placeholder] = lamda*1.5
    feed_dict[momentum_placeholder] = momentum
    __loss_list = []
    for i in range(loop_num):
        feed_dict[lr_placeholder] = lr*(0.7 + 0.3 * (1 - i/loop_num))
        img_data, img_label = sess.run(ITERATOR_train_next_element)
        img_label = str(img_label[0], encoding='utf8')
        feed_dict[input_placeholder] = img_data
        _, _loss_list, _loss_L2, _help_loss, _loss, _val, _test = sess.run(_list_in_run, feed_dict=feed_dict)
        __loss_list.append(_loss_list)
        _pure_loss = _loss - lamda*_loss_L2 - _help_loss
        recoding_list.append([_pure_loss, _val, _test])
        if i % spacing == spacing - 1:
            msg1_1 = f'iter_{i}:' + ' lossL2:' + f'{_loss_L2:{6}.{5}}' + ', _pure_loss:' + f'{_pure_loss:{6}.{5}}'
            msg1_2 = ', val_see:'+ f'{_val:{6}.{5}}' + ', test_see:'+ f'{_test:{6}.{5}}'
            # msg1 = msg1_1 + msg1_2 + ', time:' + str(timedelta(seconds=int(round(time.time()-start_time))))
            logger.info( msg1_1 + msg1_2 + ', selected img_label:' + img_label)
    msg2 = f'iter_{i}:'+ ' loss_list:'+ ''.join([str(_)+', ' for _ in np.power(np.array(__loss_list).mean(axis=0), 4) ])
    logger.info(msg2)
    logger.info( f'show recoding_list[-{loop_num}:].mean(axis=0), [loss, val, test]: ' + str( np.power( np.array(recoding_list[-loop_num:]).mean(axis=0), 4) ) )

def pre_train_for_all_zone(lamda, zone, is_go_on=None):

    global recoding_list
    set_iterator(zone)

    logger.warning(f'ZONE: {zone}, base_LAMDA: {lamda}')
    sess.run(tf.global_variables_initializer())
    logger.info('model have been random initing')
    if not os.path.isdir(f'../.checkpoints'):
        os.mkdir(f'../.checkpoints')
    if not os.path.isdir(f'../.checkpoints/convlstm_deeplav3_{IO_version}'):
        os.mkdir(f'../.checkpoints/convlstm_deeplav3_{IO_version}')
    model_path = f'../.checkpoints/convlstm_deeplav3_{IO_version}/mymodel__init__.ckpt'
    saver.save(sess, model_path)
    logger.info('model have been saved at ' + model_path)

    if is_go_on == True:
        model_path = f'../.checkpoints/convlstm_deeplav3_{IO_version}/mymodel_lamda_{lamda}_all_zone_train__.ckpt'
        saver.restore(sess, model_path)
        logger.info('model have been restored from ' + model_path)

    recoding_list = []
    if is_go_on != True:
        train(_optimizer_step=optimizer_step_adam, lamda=lamda, lr=0.0004000, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
        train(_optimizer_step=optimizer_step_adam, lamda=lamda, lr=0.0003000, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
        train(_optimizer_step=optimizer_step_adam, lamda=lamda, lr=0.0002000, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)

    train(_optimizer_step=optimizer_step_adam, lamda=lamda, lr=0.0001500, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
    train(_optimizer_step=optimizer_step_adam, lamda=lamda, lr=0.0001000, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
    train(_optimizer_step=optimizer_step_adam, lamda=lamda, lr=0.0000500, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)

    _ = f'zone: {zone} \n, lamda: {lamda} have been trained, the result:'
    logger.critical(_ + '\n' + str( np.power(np.array(recoding_list).reshape(-1,RESHAPE_spacing,len(recoding_list[-1])).mean(axis=1), 4) ) )
    logger.info('pre_train_for_all_zone() have been done')
    model_path = f'../.checkpoints/convlstm_deeplav3_{IO_version}/mymodel_lamda_{lamda}_all_zone_train__.ckpt'
    saver.save(sess, model_path)
    logger.info('model have been saved at ' + model_path)


def over_a_case(lamda, zone, is_in_train=False, is_in_val=False, is_in_submit=False, adam_or_moment='moment'):

    global recoding_list
    set_iterator(zone)

    logger.warning(f'ZONE: {zone}, base_LAMDA: {lamda}')

    if is_in_train:
        # 此处手动控制
        logger.info(r'读取公共训练的模型用于min(loss)')
        model_path = f'../.checkpoints/convlstm_deeplav3_{IO_version}/mymodel_lamda_{lamda}_all_zone_train__.ckpt'
        # logger.info(r'读取上一次单独训练的模型用于min(loss), 淬火, 如果改变lamda的话注意更新f-string哦')
        # model_path = f'../.checkpoints/convlstm_deeplav3_{IO_version}/mymodel_lamda_{lamda}_train_' + str(zone[0]) + '.ckpt'
        saver.restore(sess, model_path)
        
        recoding_list = []
        if adam_or_moment == 'adam':
            _opt_here = optimizer_step_adam
            logger.info('...... sess.run(opt.min(loss)) ...... adam ..................')
        elif adam_or_moment == 'moment':
            _opt_here = optimizer_step_moment
            logger.info('...... sess.run(opt.min(loss)) ...... moment ..................')
        else:
            logger.error('ERROR !!!')

        # 手动、反复、淬火
        train(_optimizer_step=_opt_here, lamda=lamda, lr=0.0001600, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
        train(_optimizer_step=_opt_here, lamda=lamda, lr=0.0000800, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
        train(_optimizer_step=_opt_here, lamda=lamda, lr=0.0000400, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)

        train(_optimizer_step=_opt_here, lamda=lamda, lr=0.0000200, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
        train(_optimizer_step=_opt_here, lamda=lamda, lr=0.0000100, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
        train(_optimizer_step=_opt_here, lamda=lamda, lr=0.0000050, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)

        _ = f'zone: {zone}, lamda: {lamda} have been trained, the following is the result:'
        logger.critical(_ + '\n' + str( np.power(np.array(recoding_list).reshape(-1,RESHAPE_spacing,len(recoding_list[-1])).mean(axis=1), 4) ) )

        model_path = f'../.checkpoints/convlstm_deeplav3_{IO_version}/mymodel_lamda_{lamda}_train_' + str(zone[0]) + '.ckpt'
        saver.save(sess, model_path)
        logger.info('model have been saved at ' + model_path)

    if is_in_val:
        logger.info(r'读取已经为某一单独地区train好了的模型用于min(0.5*loss+0.5*val_see)')
        model_path = f'../.checkpoints/convlstm_deeplav3_{IO_version}/mymodel_lamda_{lamda}_train_' + str(zone[0]) + '.ckpt'
        saver.restore(sess, model_path)

        recoding_list = []
        if adam_or_moment == 'adam':
            _opt_here = optimizer_step_adam_in_val
            logger.info('...... sess.run(opt.min(0.5*loss+0.5*val_see)) ...... adam ..................')
        elif adam_or_moment == 'moment':
            _opt_here = optimizer_step_moment_in_val
            logger.info('...... sess.run(opt.min(0.5*loss+0.5*val_see)) ...... moment ..................')
        else:
            logger.error('ERROR !!!')
        train(_optimizer_step=_opt_here, lamda=lamda, lr=0.0000200, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
        train(_optimizer_step=_opt_here, lamda=lamda, lr=0.0000100, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
        train(_optimizer_step=_opt_here, lamda=lamda, lr=0.0000050, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
        _ = f'zone: {zone}, lamda: {lamda} have been val, the following is the result:'
        logger.critical(_ + '\n' + str( np.power(np.array(recoding_list).reshape(-1,RESHAPE_spacing,len(recoding_list[-1])).mean(axis=1), 4) ) )

        model_path = f'../.checkpoints/convlstm_deeplav3_{IO_version}/mymodel_lamda_{lamda}_val_' + str(zone[0]) + '.ckpt'
        saver.save(sess, model_path)
        logger.info('model have been saved at ' + model_path)

    if is_in_submit:
        # 此处手动控制
        # logger.info(r'读取已经为某一单独地区val好了的模型用于min(submit)')
        # model_path = f'../.checkpoints/convlstm_deeplav3_{IO_version}/mymodel_lamda_{lamda}_val_' + str(zone[0]) + '.ckpt'
        logger.info(r'读取已经为某一单独地区train好了的模型用于min(submit)')
        model_path = f'../.checkpoints/convlstm_deeplav3_{IO_version}/mymodel_lamda_{lamda}_train_' + str(zone[0]) + '.ckpt'
        # logger.info(r'读取已经为某一单独地区submit好了的模型用于min(submit)')
        # model_path = f'../.checkpoints/convlstm_deeplav3_{IO_version}/mymodel_lamda_{lamda}_submit_' + str(zone[0]) + '.ckpt'
        saver.restore(sess, model_path)
        
        recoding_list = []
        if adam_or_moment == 'adam':
            _opt_here = optimizer_step_adam_in_submit
            logger.info('...... sess.run(opt.min(submit)) ...... adam ..................')
        elif adam_or_moment == 'moment':
            _opt_here = optimizer_step_moment_in_submit
            logger.info('...... sess.run(opt.min(submit)) ...... moment ..................')
        else:
            logger.error('ERROR !!!')
        train(_optimizer_step=_opt_here, lamda=lamda, lr=0.0000200, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
        train(_optimizer_step=_opt_here, lamda=lamda, lr=0.0000100, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
        train(_optimizer_step=_opt_here, lamda=lamda, lr=0.0000050, momentum=0.95, loop_num=LOOP_num, spacing=SPACING, zone=zone)
        _ = f'zone: {zone}, lamda: {lamda} have been run(submit), the result:'
        logger.critical(_ + '\n' + str( np.power(np.array(recoding_list).reshape(-1,RESHAPE_spacing,len(recoding_list[-1])).mean(axis=1), 4) ) )
        
        # model_path = f'../.checkpoints/convlstm_deeplav3_{IO_version}/mymodel_lamda_{lamda}_submit_' + str(zone[0]) + '.ckpt'
        # saver.save(sess, model_path)
        logger.info('model have not been saved at ' + model_path)

    # 输出预测
    if is_in_train == False and is_in_val == False and is_in_submit == False:
        logger.info('-------------------------------- submit start ! --------------------------------')
        model_path = f'../.checkpoints/convlstm_deeplav3_{IO_version}/mymodel_lamda_{lamda}_submit_' + str(zone[0]) + '.ckpt'
        saver.restore(sess, model_path)
    fast_submit(ITERATOR_test_next_element, input_placeholder, predict_base_on_img_i_list, sess, lamda, zone[0], show=False)

def set_iterator(zone_list):
    global ITERATOR_train_next_element
    global ITERATOR_test_next_element
    tfrecords_list = []
    for _ in zone_list:
        tfrecords_list.append('../data/tfrecords-' + IO_version + '/' + _ + '.tfrecords')
    iterator_train = make_iterator(tfrecords_list=tfrecords_list, is_random=True)
    sess.run(iterator_train.initializer)
    ITERATOR_train_next_element = iterator_train.get_next()
    iterator_test = make_iterator(tfrecords_list=tfrecords_list, is_random=False)
    sess.run(iterator_test.initializer)
    ITERATOR_test_next_element = iterator_test.get_next()

lamda = args.lamda

with tf.Session() as sess:
    if args.control == 'pre_train_for_all_zone':
        pre_train_for_all_zone(lamda=lamda, zone=['Z'+str(i) for i in range(1,25)], is_go_on=False)
    else:
        over_a_case(lamda=lamda, zone=[args.control], is_in_train=True, is_in_val=False, is_in_submit=True, 
                    adam_or_moment='adam')

sess.close()


'''

with tf.Session() as sess:
    set_iterator(['Z1'])
    img_data, img_label = sess.run(ITERATOR_test_next_element)
    print(img_label)
    assert img_data.shape == (BATCH, TIMESTEPS) + tuple(SHAPE) + (IMG_C,)
    pp = decoding(img_data[0,7,:,:,2])
    Image.fromarray(pp.astype(np.int32)).save('_.tif')

import os
os.system("python ./main-z1.py")

print('over...')

gen = ((i,j) for j in cache2 if f(j) for i in cache1 if f(i))

for item in gen:
    pass

from itertools import product
for x, y, z in product(xlist, ylist, zlist):
    pass
尽量使用生成器

不推荐
reduce(rf, filter(ff, map(mf, a_list)))

推荐
中间结果尽量用 ifilter, imap 代替 filter, map
from itertools import ifilter, imap
reduce(rf, ifilter(ff, imap(mf, a_list)))
layz evaluation会有更高效的内存利用率尤其是大数据时候

'''

