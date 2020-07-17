import os, cv2, random, h5py
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

#get file names for all train and test images
os.chdir('/Users/stevenhurwitt/Documents/Python/convnet/')
TRAIN_DIR = '/Users/stevenhurwitt/Documents/Python/convnet/train/'
TEST_DIR = '/Users/stevenhurwitt/Documents/Python/convnet/test/'

ROWS = 256
COLS = 256
CHANNELS = 3

#train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]


train_images = train_dogs + train_cats
random.shuffle(train_images)
test_images =  test_images

#read the image into a matrix of rgb values
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    b,g,r = cv2.split(img)
    img2 = cv2.merge([r,g,b])
    return cv2.resize(img2, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

#read in all images to data frame
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%10000 == 0: print('Processed {} of {}'.format(i, count))
    
    return data


train = prep_data(train_images)
test = prep_data(test_images)

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))

print("saving images as numpy arrays...")
np.savez("train", train)
np.savez("test", test)

labels = []
for i in train_images:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)

print("saving labels...")
with open('labels.data', 'wb') as filehandle:  
    # store the data as binary data stream
    pickle.dump(labels, filehandle)

