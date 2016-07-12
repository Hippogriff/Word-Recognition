'''
This file loads the training, validation, and testing data.
It also does the preprocessing

'''
# TODO Define the elements in .hdf5 file, including the dimensions

import matplotlib

matplotlib.use( 'Agg' )
import h5py
import numpy as np

def data_utils():
    with h5py.File( 'test_data.hdf5', 'r' ) as hf:
        data = hf.get( "X" )
        X = np.array( data )
        data = hf.get( "Words" )
        Words = np.array( data )
        data = hf.get( "y" )
        y = np.array( data )

    X = X.astype( 'float32' )
    X /= 255
    X -= X.mean( )
    X /= X.std( )
    X_train = X[0:7000]
    X_test = X[7001:]
    Words_train = Words[0:7000]
    Words_test = Words[7001:]
    y_train = y[0:7000]
    y_test = y[7001:]

    # Generate the split
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)

    print('shape of y_train: ', y_train.shape)
    return X_train, X_test, Words_train, Words_test, y_train, y_test
