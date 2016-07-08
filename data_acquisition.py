'''
This file helps in extracting images, generating words, arranging
the data according  to the need of model and saving the data in .hdf5
format
'''

import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
import h5py

count = 0
words = []  # list to store the string of words
paths = []  # path to images
# Let us first load the file and extract the paths
file = open('path/to/file/annotation_test.txt', 'r')
for i, f in enumerate(file):
    if i == 10000:
        break
    splits = f.split(" ")
    path = path + [splits[0][1:]]
    splits = f.split("_")
    words += [splits[1]]
file.close()

X = np.zeros((1000, 32, 100))
for i, p in enumerate(path):
    img = imread(path, as_grey=True)
    img = resize(img, output_shape=(32, 100))
    X[i, :, :] = img

X = np.expand_dims(X, axis=1)

chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

Words = np.zeros((1000, 24), dtype=np.int) + 37  # 37 is addes for end of the word
# Words will be the input to the embedding layer
Words[:, 0] = 0  # this is for start of the word
print (char_indices)

for i, w in enumerate(words):
    for j, char in enumerate(w):
        Words[i, j + 1] = char_indices[char.lower()]

# Words = np_utils.to_categorical(Words, 38)
W = np.zeros((1000, 24, 38), dtype=np.bool)
for i in range(1000):
    for j in range(24):
        W[i, j, Words[i, j]] = 1

y = np.ones((1000, 1, 38), dtype=np.bool)
y[:, :, 37] = 1
y = np.concatenate((W, y), axis=1)

with h5py.File('test_data.hdf5', 'w') as hf:
    hf.create_dataset('X', data=X)  # Images,
    hf.create_dataset('y', data=y)  # Target layer of LSTMS, in categorical one-hot way
    hf.create_dataset('Words', data=Words)  # Words to be input to the embedding layer
    # hf.create_dataset('W', data=W)  # One-hot representation for the LSTM layer targets, not required if you have y
    hf.create_dataset('paths', data=paths)  # paths to the images
    hf.create_dataset('words_string', data=words)  # list of strings containing the words
