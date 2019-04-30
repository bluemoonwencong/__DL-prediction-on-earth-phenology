#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import logging
import itertools
from skimage.measure import block_reduce


# basic param
IO_version = 'io2'
BATCH     = 1
TIMESTEPS = 10
WIDTH   = 240
CROSS   = 24
VIEW    = WIDTH + 2 * CROSS
SHAPE   = [VIEW, VIEW]
IMG_input_C = 3
IMG_C   = 2 * (IMG_input_C + 3 + 3 + 3)
KERNEL  = [3, 3]
FILTERS = 24
DTYPE   = tf.float32
assert int(FILTERS/4.0) == FILTERS/4.0
assert BATCH == 1
assert int(VIEW/8.0) == VIEW/8.0
assert int(1200/WIDTH) == 1200/WIDTH

# 训练，只能动以下参数。
EPOCH = int(1200/WIDTH)*int(1200/WIDTH)*20 + 0
EPOCH = int(EPOCH/2.0)*2
RESHAPE_spacing   = EPOCH
LOOP_num, SPACING = EPOCH, EPOCH
assert int(LOOP_num/RESHAPE_spacing) == LOOP_num/RESHAPE_spacing
assert int(RESHAPE_spacing/2.0) == RESHAPE_spacing/2.0


# set logger
logger = logging.getLogger()
logger.setLevel('DEBUG')
BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y/%m/%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
handler1 = logging.StreamHandler()
handler1.setFormatter(formatter)
handler1.setLevel('INFO')#控制台就info级别吧
handler2 = logging.FileHandler('logging.log')#日志文件debug级别
handler2.setFormatter(formatter)
logger.addHandler(handler1)
logger.addHandler(handler2)
logger.debug('')
logger.debug('')
logger.debug('debug message, RUN, RUN, RUN!!!')
logger.info('info message')
logger.warning('warning message')
logger.error('error message')
# logger.critical('critical message')

# window table
assert WIDTH == 240
assert VIEW  == 240 + 48
_cutpoints_a = 0, WIDTH + 2*CROSS
_cutpoints_b = 1*WIDTH - CROSS, 2*WIDTH + CROSS
_cutpoints_c = 2*WIDTH - CROSS, 3*WIDTH + CROSS
_cutpoints_d = 3*WIDTH - CROSS, 4*WIDTH + CROSS
_cutpoints_e = 4*WIDTH - 2*CROSS, 5*WIDTH

assert len(range(*_cutpoints_a)) == VIEW
assert len(range(*_cutpoints_b)) == VIEW
assert len(range(*_cutpoints_c)) == VIEW
assert len(range(*_cutpoints_d)) == VIEW
assert len(range(*_cutpoints_e)) == VIEW
window_11 = _cutpoints_a + _cutpoints_a
window_12 = _cutpoints_a + _cutpoints_b
window_13 = _cutpoints_a + _cutpoints_c
window_14 = _cutpoints_a + _cutpoints_d
window_15 = _cutpoints_a + _cutpoints_e

window_21 = _cutpoints_b + _cutpoints_a
window_22 = _cutpoints_b + _cutpoints_b
window_23 = _cutpoints_b + _cutpoints_c
window_24 = _cutpoints_b + _cutpoints_d
window_25 = _cutpoints_b + _cutpoints_e

window_31 = _cutpoints_c + _cutpoints_a
window_32 = _cutpoints_c + _cutpoints_b
window_33 = _cutpoints_c + _cutpoints_c
window_34 = _cutpoints_c + _cutpoints_d
window_35 = _cutpoints_c + _cutpoints_e

window_41 = _cutpoints_d + _cutpoints_a
window_42 = _cutpoints_d + _cutpoints_b
window_43 = _cutpoints_d + _cutpoints_c
window_44 = _cutpoints_d + _cutpoints_d
window_45 = _cutpoints_d + _cutpoints_e

window_51 = _cutpoints_e + _cutpoints_a
window_52 = _cutpoints_e + _cutpoints_b
window_53 = _cutpoints_e + _cutpoints_c
window_54 = _cutpoints_e + _cutpoints_d
window_55 = _cutpoints_e + _cutpoints_e

decode_a_i, decode_a_ii = 0, WIDTH
decode_b_i, decode_b_ii = CROSS, CROSS + WIDTH
decode_c_i, decode_c_ii = CROSS, CROSS + WIDTH
decode_d_i, decode_d_ii = CROSS, CROSS + WIDTH
decode_e_i, decode_e_ii = 2*CROSS, 2*CROSS + WIDTH

assert len(range(decode_a_i, decode_a_ii)) == WIDTH
assert len(range(decode_b_i, decode_b_ii)) == WIDTH
assert len(range(decode_c_i, decode_c_ii)) == WIDTH
assert len(range(decode_d_i, decode_d_ii)) == WIDTH
assert len(range(decode_e_i, decode_e_ii)) == WIDTH

def view_recover_to_width(x, window_select):
    if   window_select == str(window_11): return x[decode_a_i:decode_a_ii, decode_a_i:decode_a_ii]
    elif window_select == str(window_12): return x[decode_a_i:decode_a_ii, decode_b_i:decode_b_ii]
    elif window_select == str(window_13): return x[decode_a_i:decode_a_ii, decode_c_i:decode_c_ii]
    elif window_select == str(window_14): return x[decode_a_i:decode_a_ii, decode_d_i:decode_d_ii]
    elif window_select == str(window_15): return x[decode_a_i:decode_a_ii, decode_e_i:decode_e_ii]
    
    elif window_select == str(window_21): return x[decode_b_i:decode_b_ii, decode_a_i:decode_a_ii]
    elif window_select == str(window_22): return x[decode_b_i:decode_b_ii, decode_b_i:decode_b_ii]
    elif window_select == str(window_23): return x[decode_b_i:decode_b_ii, decode_c_i:decode_c_ii]
    elif window_select == str(window_24): return x[decode_b_i:decode_b_ii, decode_d_i:decode_d_ii]
    elif window_select == str(window_25): return x[decode_b_i:decode_b_ii, decode_e_i:decode_e_ii]
    
    elif window_select == str(window_31): return x[decode_c_i:decode_c_ii, decode_a_i:decode_a_ii]
    elif window_select == str(window_32): return x[decode_c_i:decode_c_ii, decode_b_i:decode_b_ii]
    elif window_select == str(window_33): return x[decode_c_i:decode_c_ii, decode_c_i:decode_c_ii]
    elif window_select == str(window_34): return x[decode_c_i:decode_c_ii, decode_d_i:decode_d_ii]
    elif window_select == str(window_35): return x[decode_c_i:decode_c_ii, decode_e_i:decode_e_ii]
    
    elif window_select == str(window_41): return x[decode_d_i:decode_d_ii, decode_a_i:decode_a_ii]
    elif window_select == str(window_42): return x[decode_d_i:decode_d_ii, decode_b_i:decode_b_ii]
    elif window_select == str(window_43): return x[decode_d_i:decode_d_ii, decode_c_i:decode_c_ii]
    elif window_select == str(window_44): return x[decode_d_i:decode_d_ii, decode_d_i:decode_d_ii]
    elif window_select == str(window_45): return x[decode_d_i:decode_d_ii, decode_e_i:decode_e_ii]
    
    elif window_select == str(window_51): return x[decode_e_i:decode_e_ii, decode_a_i:decode_a_ii]
    elif window_select == str(window_52): return x[decode_e_i:decode_e_ii, decode_b_i:decode_b_ii]
    elif window_select == str(window_53): return x[decode_e_i:decode_e_ii, decode_c_i:decode_c_ii]
    elif window_select == str(window_54): return x[decode_e_i:decode_e_ii, decode_d_i:decode_d_ii]
    elif window_select == str(window_55): return x[decode_e_i:decode_e_ii, decode_e_i:decode_e_ii]
    
    else: raise TypeError

def encoding(x):
    return x/10000.0 - 0.0

def decoding(x):
    return (x + 0.0)*10000.0

def read_as_array(path, low_row, hi_row, low_column, hi_column):

    _d   = path.split('-')[0]
    _int = path[-7:-4]
    path = '../data/' + _d + '/' + path
    _img = np.array(Image.open(path)).astype(np.float32)

    # base
    _int = int(_int)
    _int = _int % 12
    if _int == 0: _int = 12
    _base_path = '../data/' + _d + 'base/' + _d + '-' + str(_int) + '.tif'
    _base = np.array(Image.open(_base_path)).astype(np.float32)
    
    _img = _img - _base
    
    _img_origin_subtract_base_window = _img[low_row:hi_row, low_column:hi_column]
    _img_origin_subtract_base_pool   = np.zeros([VIEW, VIEW]).astype(np.float32) + np.mean(_img)
    _img_origin_subtract_base_pool[CROSS:CROSS+WIDTH, CROSS:CROSS+WIDTH] = block_reduce(_img, (int(1200/WIDTH), int(1200/WIDTH)), np.mean)
    assert _img_origin_subtract_base_pool.shape == _img_origin_subtract_base_window.shape

    logger.info('this will be write into tfrecordes: window@(' + path + ' - the base: ' + _base_path + ')')
    logger.info('this will be write into tfrecordes:   pool@(' + path + ' - the base: ' + _base_path + ')')
    
    return encoding(_img_origin_subtract_base_window), encoding(_img_origin_subtract_base_pool)

def axis_off_imshow(x, title=None, axis_off='off'):
    plt.title(title)
    plt.axis(axis_off)
    plt.imshow(x)

def _save_img(_img_pred__, str_id, dd_select, window_select, _pp, show):
    save_name = 'predict_' + dd_select + '-' + str_id + '---' + window_select + '.tif'
    cache = view_recover_to_width(_img_pred__, window_select)
    submit = Image.fromarray(decoding(cache).astype(np.int32))
    if show:
        logger.info(str((_img_pred__.shape, cache.shape, cache.min(), cache.max())) )
        logger.info('file have been saved at ' + _pp + save_name)
    submit.save(_pp + save_name)
    assert (np.array(Image.open(_pp + save_name)) == np.array(submit)).all()

def fast_submit(ITERATOR_test_next_element, input_placeholder, predict_base_on_img_i_list, sess, lamda, zone, show):
    logger.info('fast_submit() start!')
    if show: logger.info('')
    while True:
        if show: logger.info('......')
        try:
            img_data, img_label = sess.run(ITERATOR_test_next_element)
            img_label = str(img_label[0], encoding='utf8')
        except tf.errors.OutOfRangeError:
            logger.warning('tf.errors.OutOfRangeError')
            logger.warning('tf.errors.OutOfRangeError')
            logger.warning('while loop will break')
            break
        if show: logger.info('img_label: ' + img_label)
        dd_select = img_label.split('(')[0]
        window_select = '(' + img_label.split('(')[1]
        if show: logger.info( f'dd_select: {dd_select}, window_select: {window_select}' )
        
        _feed_dict = {}
        # 以后尽量不用 tuple 重载的加法运算符，容易出错，因为单个元素的时候必须加上 ","
        assert img_data.shape == (BATCH, TIMESTEPS) + tuple(SHAPE) + (IMG_C,)
        _feed_dict[input_placeholder] = img_data

        assert len(predict_base_on_img_i_list) == TIMESTEPS
        assert predict_base_on_img_i_list[0].shape.as_list() == SHAPE+[IMG_input_C]
        # 正常情况，把_feed_dict喂进去
        # logger.info(r'正常情况，把_feed_dict喂进去')
        _img_pred_all = sess.run(predict_base_on_img_i_list[-3:], feed_dict=_feed_dict)
        # 测试。末尾三年的输入是['196', '197', '208', '209', '220', '221']
        # 本应该得到下一年的，如果是sess.run()的话，但现在测试阶段只是为了看看是否精确对齐
        # logger.info(r'测试overlap有没对齐')
        # _img_pred_all = img_data[0,-3:,:,:,:]

        _img_pred_val_n_plus_1    = _img_pred_all[0][:,:,1]
        _img_pred_val_n_plus_2    = _img_pred_all[0][:,:,2]
        _img_pred_test_n_plus_1   = _img_pred_all[1][:,:,1]
        _img_pred_test_n_plus_2   = _img_pred_all[1][:,:,2]
        _img_pred_submit_n_plus_1 = _img_pred_all[2][:,:,1]
        _img_pred_submit_n_plus_2 = _img_pred_all[2][:,:,2]

        if not os.path.isdir('../data_submit'):
            os.mkdir('../data_submit')
        if not os.path.isdir(f'../data_submit/{IO_version}'):
            os.mkdir(f'../data_submit/{IO_version}')
        if not os.path.isdir(f'../data_submit/{IO_version}/fragments'):
            os.mkdir(f'../data_submit/{IO_version}/fragments')
        _pp = f'../data_submit/{IO_version}/fragments/'
        
        _save_img(_img_pred__=_img_pred_val_n_plus_1, str_id='208', dd_select=dd_select, window_select=window_select, _pp=_pp, show=show)
        _save_img(_img_pred__=_img_pred_val_n_plus_2, str_id='209', dd_select=dd_select, window_select=window_select, _pp=_pp, show=show)
        _save_img(_img_pred__=_img_pred_test_n_plus_1, str_id='220', dd_select=dd_select, window_select=window_select, _pp=_pp, show=show)
        _save_img(_img_pred__=_img_pred_test_n_plus_2, str_id='221', dd_select=dd_select, window_select=window_select, _pp=_pp, show=show)
        _save_img(_img_pred__=_img_pred_submit_n_plus_1, str_id='232', dd_select=dd_select, window_select=window_select, _pp=_pp, show=show)
        _save_img(_img_pred__=_img_pred_submit_n_plus_2, str_id='233', dd_select=dd_select, window_select=window_select, _pp=_pp, show=show)

    for cache_dd in [zone]:
        for cache_int in ['208', '209', '220', '221', '232', '233']:
            pp = np.zeros((1200,1200))
            cache_p = 'predict_' + cache_dd + '-' + cache_int

            f_list = [cache_p+'---'+str(window_11)+'.tif', cache_p+'---'+str(window_12)+'.tif', cache_p+'---'+str(window_13)+'.tif',
                      cache_p+'---'+str(window_14)+'.tif', cache_p+'---'+str(window_15)+'.tif']
            for i, fliename in enumerate(f_list):
                pp[0:240, i*240:(i+1)*240] = np.array(Image.open(_pp + fliename))

            f_list = [cache_p+'---'+str(window_21)+'.tif', cache_p+'---'+str(window_22)+'.tif', cache_p+'---'+str(window_23)+'.tif',
                      cache_p+'---'+str(window_24)+'.tif', cache_p+'---'+str(window_25)+'.tif']
            for i, fliename in enumerate(f_list):
                pp[240:480, i*240:(i+1)*240] = np.array(Image.open(_pp + fliename))

            f_list = [cache_p+'---'+str(window_31)+'.tif', cache_p+'---'+str(window_32)+'.tif', cache_p+'---'+str(window_33)+'.tif',
                      cache_p+'---'+str(window_34)+'.tif', cache_p+'---'+str(window_35)+'.tif']
            for i, fliename in enumerate(f_list):
                pp[480:720, i*240:(i+1)*240] = np.array(Image.open(_pp + fliename))

            f_list = [cache_p+'---'+str(window_41)+'.tif', cache_p+'---'+str(window_42)+'.tif', cache_p+'---'+str(window_43)+'.tif',
                      cache_p+'---'+str(window_44)+'.tif', cache_p+'---'+str(window_45)+'.tif']
            for i, fliename in enumerate(f_list):
                pp[720:960, i*240:(i+1)*240] = np.array(Image.open(_pp + fliename))

            f_list = [cache_p+'---'+str(window_51)+'.tif', cache_p+'---'+str(window_52)+'.tif', cache_p+'---'+str(window_53)+'.tif',
                      cache_p+'---'+str(window_54)+'.tif', cache_p+'---'+str(window_55)+'.tif']
            for i, fliename in enumerate(f_list):
                pp[960:1200, i*240:(i+1)*240] = np.array(Image.open(_pp + fliename))
            
            # residual_error (the direct target)
            _ = _pp + cache_dd + '-' + cache_int + '_residual_error.tif'
            Image.fromarray(pp.astype(np.int32)).save(_)
            logger.info(f'mean(file)={pp.mean()}, file have been saved at ' + _)
            
            # base
            _int = int(cache_int)
            _int = _int % 12
            if _int == 0: _int = 12
            _base_path = '../data/' + cache_dd + 'base/' + cache_dd + '-' + str(_int) + '.tif'
            _base = np.array(Image.open(_base_path)).astype(np.float32)

            _ = _pp + cache_dd + '-' + cache_int + '_base.tif'
            Image.fromarray(( 0 + _base).astype(np.int32)).save(_)
            logger.info(f'mean(flie)={( 0 + _base).mean()}, file have been saved at ' + _)

            # residual_error + base
            _ = _pp + cache_dd + '-' + cache_int + '.tif'
            Image.fromarray((pp + _base).astype(np.int32)).save(_)
            logger.info(f'mean(flie)={(pp + _base).mean()}, file have been saved at ' + _)

            if cache_int in ['208', '209']:
                val_test_submit = 'val'
            elif cache_int in ['220', '221']:
                val_test_submit = 'test'
            elif cache_int in ['232', '233']:
                val_test_submit = 'submit'
            else:
                logger.error('ERROR !!!')

            if not os.path.isdir(f'../data_submit/{IO_version}/{val_test_submit}'):
                os.mkdir(f'../data_submit/{IO_version}/{val_test_submit}')
            _ = f'../data_submit/{IO_version}/{val_test_submit}/' + cache_dd + '-' + cache_int + '.tif'
            Image.fromarray((pp + _base).astype(np.int32)).save(_)
            logger.info(f'mean(flie)={(pp + _base).mean()}, file have been saved at ' + _)

    pass


# ------------------------------------------------------------------------------------------
# 选择文件列表
def choose_file(choose, dd):
    _file_list = []
    _month = [0]
    _month += os.listdir('../data/' + dd)
    while choose>0 and len(_file_list)<TIMESTEPS:
        _file_list.append(_month[choose])
        choose = choose - 12
    return list(reversed(_file_list))


def make_input_img_array(window_select, dd):
    # 选定文件
    choose = 231
    _0 = choose_file(choose = choose - 12*1 + 0, dd=dd)
    _1 = choose_file(choose = choose - 12*1 + 1, dd=dd)
    _2 = choose_file(choose = choose - 12*1 + 2, dd=dd)

    _3 = choose_file(choose = choose - 12*0 - 1, dd=dd)
    _4 = choose_file(choose = choose - 12*0 - 2, dd=dd)
    _5 = choose_file(choose = choose - 12*0 - 3, dd=dd)

    _6 = choose_file(choose = choose - 12*1 - 1, dd=dd)
    _7 = choose_file(choose = choose - 12*1 - 2, dd=dd)
    _8 = choose_file(choose = choose - 12*1 - 3, dd=dd)

    _9 = choose_file(choose = choose - 12*1 + 3, dd=dd)
    _10 = choose_file(choose = choose - 12*1 + 4, dd=dd)
    _11 = choose_file(choose = choose - 12*1 + 5, dd=dd)

    train_file_list = [(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11) for i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11 in zip(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11)]
    assert len(train_file_list) == TIMESTEPS
    # 输入数据
    logger.info('read images:')
    input_img_array = np.zeros([BATCH, TIMESTEPS] + SHAPE + [IMG_C]).astype(np.float32)
    for i, path in enumerate(train_file_list):
        assert train_file_list[i] == path
        logger.info(f'train_file_list[{i}]: ' + str(path))
        for j in range(0, len(path)):
            _origin_subtract_base_window, _origin_subtract_base_pool = read_as_array(path[j], *window_select)
            input_img_array[0,i,:,:,j+0]         = _origin_subtract_base_window
            input_img_array[0,i,:,:,j+len(path)] = _origin_subtract_base_pool
            pass
    logger.info( str([BATCH, TIMESTEPS] + SHAPE + [IMG_C]) + str([input_img_array.shape, input_img_array.min(), input_img_array.max()]) )
    return train_file_list, input_img_array

window_list = [window_11, window_12, window_13, window_14, window_15,
               window_21, window_22, window_23, window_24, window_25,
               window_31, window_32, window_33, window_34, window_35,
               window_41, window_42, window_43, window_44, window_45,
               window_51, window_52, window_53, window_54, window_55
              ]

def decoding_func(serialized_example):
    features = tf.parse_single_example(serialized_example, features={
                                'img_label': tf.FixedLenFeature([], tf.string),
                                'img_data' : tf.FixedLenFeature([], tf.string)})
        # 踩坑 !!!
        # 如果下一句用 tf.uint8 不会报错, 但就是错的.尺寸会增加到 [H, W, 8*C]
        # 因为 tf.float64 占据的空间是 tf.uint8 的 8 倍, 而写入 *.tfrecords 文件的时候 img 用的是 np.float64
        # 所以应该统一改成 float32, *.tfrecords 和 .npy 文件都可以节省一半空间了
        # 不行,就算把 img 改成 float32, 在编码为 *.tfrecords 文件的时候还是会在内部改成 float64,
        # 或者是因为 float32 和 float64 在编码到字节码的时候所占空间相同 ?
    img   = tf.decode_raw(features['img_data'], tf.float32)
    # 下面 reshape 不用加 BATCH，会自动加的
    img   = tf.reshape(img, [TIMESTEPS] + SHAPE + [IMG_C] )
    label = tf.cast(features['img_label'], tf.string)
    return img, label

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_iterator(tfrecords_list, is_random):
    dataset  = tf.data.TFRecordDataset(tfrecords_list)
    dataset  = dataset.map(decoding_func)
    dataset  = dataset.batch(BATCH)
    # Dataset.shuffle() 转换会使用类似于 tf.RandomShuffleQueue 的算法随机重排输入数据集：
    # 它会维持一个固定大小的缓冲区，并从该缓冲区统一地随机选择下一个元素。
    if is_random:
        dataset  = dataset.repeat()
        dataset  = dataset.shuffle(buffer_size=int( 1.0*int(1200/WIDTH)*int(1200/WIDTH) ))
    iterator = dataset.make_initializable_iterator()
    return iterator

if __name__ == "__main__":
    _pp = '../data/' + 'tfrecords-' + IO_version + '/'
    if not os.path.isdir(_pp):
        os.mkdir(_pp)
    # make *.tfrecords files
    logger.info('make *.tfrecords files')
    for _zone in ['Z'+str(i) for i in range(1,25)]:
        writer = tf.python_io.TFRecordWriter(_pp + _zone + ".tfrecords")
        for _dd, _window in itertools.product([_zone], window_list[0:]):
            _, img_data = make_input_img_array(window_select=_window, dd=_dd)
            img_data  = img_data.astype(np.float32)
            img_label = bytes(_dd +str(_window), 'utf-8')
            # 竟然没影响,省不了空间了
            # img_npy.dtype = 'float32'
            feature = {"img_label": _bytes_feature(img_label), 'img_data': _bytes_feature(img_data.tobytes())}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            _msg = 'an example have been writed, img_label:', img_label, ' shape and mean: ', img_data.shape, img_data.mean()
            logger.info(_msg)


'''

for i in range(len(files_img)):

    # 这里可能有些教程里会使用skimage或者opencv来读取文件，但是我试了一下opencv的方法
    # 一个400MB左右的训练集先用cv2.imread转换成numpy数组再转成string，最后写入TFRecord
    # 得到的文件有17GB，所以这里推荐使用FastGFile方法，生成的tfrecord文件会变小，
    # 唯一不好的一点就是这种方法需要在读取之后再进行一次解码。
    img = tf.gfile.FastGFile(files_img[i], 'rb').read()
    label = tf.gfile.FastGFile(files_anno[i], 'rb').read()

    # 按照第一部分中Example Protocol Buffer的格式来定义要存储的数据格式
    example = tf.train.Example(features=tf.train.Features(feature={
        'raw_image': _bytes_feature(img),
        'label': _bytes_feature(label)
    }))
    # 最后将example写入TFRecord文件
    writer.write(example.SerializeToString())

# 因为FastGFile读取的是图片没有解码过的的原始数据，所以在使用存在tfrecord中的这些原始数据时，需要对读取出来的图片原始数据进行解码。

'''