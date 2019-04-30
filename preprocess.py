
import os
import numpy as np
from PIL import Image

def guess_func(dir_check, choose, end, alpha):
    choose = choose % 12 - 1
    if choose == -1: choose = 11
    flielist = os.listdir(dir_check)[choose:end:12]
#     print(flielist)
    guess = np.array(Image.open(dir_check+flielist[0]))
    for i, flie in enumerate(flielist):
        img = np.array(Image.open(dir_check+flie))
        if alpha == None:
            # simple average
            guess = guess*(i/(i+1)) + img/(i+1)
        else:
            # moving average
            guess = guess*(1-alpha) + img*alpha
    return guess


# 准备 base 数据
for dd in ['Z'+str(i) for i in range(1,25)]:
    print('base for ' + dd + ':')
    for choose in range(232, 232+12):
        guess = guess_func('./data/' + dd + '/', choose=choose, end=choose-1, alpha=None)
        if not os.path.isdir('./data/'+dd+'base'):
            os.mkdir('./data/'+dd+'base')
        choose = choose % 12
        if choose == 0: choose = 12
        Image.fromarray(np.round(guess).astype(np.int16)).save('./data/' + dd + 'base/' + dd + '-' + str(choose) + '.tif')


