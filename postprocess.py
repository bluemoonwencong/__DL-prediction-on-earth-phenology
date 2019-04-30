import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

for f in os.listdir('./data_submit/io2/submit'):
    img = np.array(Image.open('./data_submit/io2/submit/' + f))
    print(f, img.min(), img.max())
    Image.fromarray(np.clip(img,0,10000).astype(np.int32)).save('./data_submit/io2/submit/' + f)
