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
from keras.models import model_from_json
from keras.models import load_model


ROWS = 256
COLS = 256
CHANNELS = 3

os.chdir('/Users/stevenhurwitt/Documents/Python/convnet/')
TRAIN_DIR = '/Users/stevenhurwitt/Documents/Python/convnet/train/'
TEST_DIR = '/Users/stevenhurwitt/Documents/Python/convnet/test/'

#load training/test data
print("loading data...")

train_npz = np.load("train.npz")
test_npz = np.load("test.npz")

train = train_npz['arr_0']
test = test_npz['arr_0']

#load labels
print("loading labels...")

with open('labels.data', 'rb') as filehandle:  
    # read the data as binary data stream
    labels = pickle.load(filehandle)


#load model
print("loading model...")
json_file = open('catdog.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("catdog.h5")
print("loaded model from disk")

#model.save('catdog.hdf5')
loaded_model=load_model('catdog.hdf5')

print("loading predictions")

#predictions = loaded_model.predict(test)

with open('preds.data', 'rb') as filehandle:  
    predictions = pickle.load(filehandle)

with open('history.data', 'rb') as filehandle:  
    history = pickle.load(filehandle)


loss = history.losses
val_loss = history.val_losses

json_file = open('catdog.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("catdog.h5")
print("Loaded model from disk")

#model.save('catdog.hdf5')
loaded_model=load_model('catdog.hdf5')

#pred2 = loaded_model.predict(test)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,epochs)[0::2])
plt.legend()
plt.show()
   
#show sample of predictions
for i in range(0,10):
    if predictions[i, 0] >= 0.5: 
        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))
    else: 
        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))
        
    plt.imshow(test[i].T)
    plt.pause(0)

