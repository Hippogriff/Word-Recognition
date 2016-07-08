'''
This file loads the training, validation, and testing data.
It also does the preprocessing

'''
#TODO Define the elements in .hdf5 file, including the dimensions

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import h5py
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils

def data_utils():
    path_pos = '/data4/gopal.sharma/datasets/deep_text/text-non-text/train_text_non_text_pos.h5'

    with h5py.File('test_data.hdf5', 'r') as hf:
        data = hf.get("X")
        X = np.array(data)
        data = hf.get("Words")
        Words = np.array(data)
        data = hf.get("y")
        y = np.array(data)

    X = X.astype('float32')
    X /= 255
    X -= X.mean()
    X /= X.std()
    X_train = X[0:700000]
    X_test = X[700001:]
    Words_train = Words[0:700000]
    Words_test = Words[700001:]
    y_train = y[0:700000]
    y_test = y[700001:]

    # Generate the split
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)

    print('shape of y_train: ', y_train.shape)
    return X_train, X_test, Words_train, Words_test, y_train, y_test