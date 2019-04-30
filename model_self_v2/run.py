#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import time

for i in range(0):
    time.sleep(60)
    print(f'wait {i+1} min')

lamda = 101e-8

# os.system(f"python ./HEADio.py")

os.system(f"python ./main.py pre_train_for_all_zone {lamda}")

for _ in ['Z'+str(i) for i in range(1,25)]:
    os.system(f"python ./main.py {_} {lamda}")

print('over...')

'''

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

for f in os.listdir('./data_submit/io2/submit'):
    img = np.array(Image.open('./data_submit/io2/submit/' + f))
    print(f, img.min(), img.max())
    Image.fromarray(np.clip(img,0,10000).astype(np.int32)).save('./data_submit/io2/submit/_' + f)




'''
