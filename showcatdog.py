import matplotlib.pyplot as plt
import numpy as np
import os, cv2

os.chdir('/Users/stevenhurwitt/Documents/Python/convnet/')
TRAIN_DIR = '/Users/stevenhurwitt/Documents/Python/convnet/train/'

ROWS = 256
COLS = 256
CHANNELS = 3

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    b,g,r = cv2.split(img)
    img2 = cv2.merge([r,g,b])
    return cv2.resize(img2, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

for a in range(0,5):
    cat = read_image(train_cats[a])
    dog = read_image(train_dogs[a])
    pair = np.concatenate((cat, dog), axis=1)
    plt.figure(figsize=(10,5))
    plt.imshow(pair)
    plt.show()
